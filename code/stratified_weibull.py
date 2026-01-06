import os
from pathlib import Path
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / 'eval-analysis-public' / 'data' / 'external' / 'all_runs.jsonl'
DATA_PATH = Path(os.environ.get('PETO_DATA_FILE', str(DEFAULT_DATA_PATH)))
RESULTS_ROOT = Path(os.environ.get('PETO_RESULTS_ROOT', str(PROJECT_ROOT / 'results_mle')))
OUTPUT_DIR = RESULTS_ROOT / 'stratified'

FRONTIER_MODELS = [
    "Claude 3.5 Sonnet (New)",
    "GPT-4o",
    "Claude 3.7 Sonnet",
    "o1",
]

MIN_FAMILY_COUNT = int(os.environ.get("MIN_FAMILY_COUNT", "50"))
DRAWS = int(os.environ.get("PM_DRAWS", "800"))
TUNE = int(os.environ.get("PM_TUNE", "800"))
CHAINS = int(os.environ.get("PM_CHAINS", "4"))
TARGET_ACCEPT = float(os.environ.get("PM_TARGET_ACCEPT", "0.9"))
RANDOM_SEED = int(os.environ.get("PM_SEED", "42"))


def load_stratified_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_json(path, lines=True)

    df = df[df["alias"].isin(FRONTIER_MODELS)].copy()

    if "task_family" not in df.columns:
        df["task_family"] = df["task_id"].apply(lambda x: x.split("__")[0])

    family_counts = df["task_family"].value_counts()
    valid_families = family_counts[family_counts > MIN_FAMILY_COUNT].index
    df = df[df["task_family"].isin(valid_families)].copy()

    families = sorted(df["task_family"].unique().tolist())
    fam_map = {name: idx for idx, name in enumerate(families)}
    df["fam_idx"] = df["task_family"].map(fam_map)

    df["t"] = df["human_minutes"] / 60.0
    df = df[df["t"] > 0].copy()

    return df, families, fam_map


def fit_stratified_weibull(df, families):
    n_families = len(families)
    fam_idx = df["fam_idx"].values
    t = df["t"].values
    y = df["score_binarized"].values

    with pm.Model() as model:
        mu_k = pm.Normal("mu_k", 1.0, 0.5)
        sigma_k = pm.HalfNormal("sigma_k", 0.5)

        k_fam = pm.Normal("k_fam", mu=mu_k, sigma=sigma_k, shape=n_families)
        k = pm.Deterministic("k", pm.math.exp(k_fam))

        mu_lam = pm.Normal("mu_lam", np.log(0.1), 1.0)
        lam = pm.math.exp(pm.Normal("lam_fam", mu=mu_lam, sigma=1.0, shape=n_families))

        c = pm.Beta("c", alpha=10, beta=1)

        hazard = pm.math.exp(-((lam[fam_idx] * t) ** k[fam_idx]))
        p = c * hazard

        pm.Bernoulli("y", p=p, observed=y)

        trace = pm.sample(
            DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
        )

    return trace


def write_outputs(trace, families):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = az.summary(trace, var_names=["k", "c"], hdi_prob=0.95)
    summary_path = os.path.join(OUTPUT_DIR, "stratified_posterior_summary.csv")
    summary.to_csv(summary_path)

    fam_map_path = os.path.join(OUTPUT_DIR, "family_index_map.csv")
    pd.DataFrame({"fam_idx": list(range(len(families))), "task_family": families}).to_csv(
        fam_map_path, index=False
    )

    az.plot_forest(trace, var_names=["k"], combined=True)
    plt.axvline(1.0, color="red", linestyle="--", label="Constant Hazard (k = 1)")
    plt.title("Weibull k by Task Family (Frontier Models)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stratified_k_forest.png"))
    plt.close()

    return summary_path, fam_map_path


def main():
    df, families, fam_map = load_stratified_data(DATA_PATH)
    print(f"Analyzing {len(df)} runs across {len(families)} task families.")

    trace = fit_stratified_weibull(df, families)
    summary_path, fam_map_path = write_outputs(trace, families)

    print(f"Saved summary: {summary_path}")
    print(f"Saved family map: {fam_map_path}")
    print(f"Saved plot: {os.path.join(OUTPUT_DIR, 'stratified_k_forest.png')}")


if __name__ == "__main__":
    main()
