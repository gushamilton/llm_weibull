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
OUTPUT_DIR = RESULTS_ROOT / 'stratified_task_groups'

TASK_GROUPS = ["SWAA", "HCAST", "RE-Bench"]

FRONTIER_MODELS = [
    "Claude 3.5 Sonnet (New)",
    "GPT-4o",
    "Claude 3.7 Sonnet",
    "o1",
]

DRAWS = int(os.environ.get("PM_DRAWS", "1500"))
TUNE = int(os.environ.get("PM_TUNE", "1500"))
CHAINS = int(os.environ.get("PM_CHAINS", "4"))
TARGET_ACCEPT = float(os.environ.get("PM_TARGET_ACCEPT", "0.97"))
RANDOM_SEED = int(os.environ.get("PM_SEED", "42"))


def load_grouped_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_json(path, lines=True)

    df = df[df["alias"].isin(FRONTIER_MODELS)].copy()

    if "task_source" not in df.columns:
        raise KeyError("Expected column 'task_source' not found in dataset.")

    df = df[df["task_source"].isin(TASK_GROUPS)].copy()

    df["t"] = df["human_minutes"] / 60.0
    df = df[df["t"] > 0].copy()

    groups = sorted(df["task_source"].unique().tolist())
    group_map = {name: idx for idx, name in enumerate(groups)}
    df["group_idx"] = df["task_source"].map(group_map)

    return df, groups, group_map


def fit_group_weibull(df, groups):
    n_groups = len(groups)
    group_idx = df["group_idx"].values
    t = df["t"].values
    y = df["score_binarized"].values

    with pm.Model() as model:
        mu_k = pm.Normal("mu_k", 1.0, 0.5)
        sigma_k = pm.HalfNormal("sigma_k", 0.5)

        k_group = pm.Normal("k_group", mu=mu_k, sigma=sigma_k, shape=n_groups)
        k = pm.Deterministic("k", pm.math.exp(k_group))

        mu_lam = pm.Normal("mu_lam", np.log(0.1), 1.0)
        lam = pm.math.exp(pm.Normal("lam_group", mu=mu_lam, sigma=1.0, shape=n_groups))

        c = pm.Beta("c", alpha=10, beta=1)

        hazard = pm.math.exp(-((lam[group_idx] * t) ** k[group_idx]))
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


def write_outputs(trace, groups):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = az.summary(trace, var_names=["k"], hdi_prob=0.95)
    summary_path = os.path.join(OUTPUT_DIR, "task_group_posterior_summary.csv")
    summary.to_csv(summary_path)

    group_map_path = os.path.join(OUTPUT_DIR, "task_group_index_map.csv")
    pd.DataFrame({"group_idx": list(range(len(groups))), "task_source": groups}).to_csv(
        group_map_path, index=False
    )

    az.plot_forest(trace, var_names=["k"], combined=True)
    plt.axvline(1.0, color="red", linestyle="--", label="Constant Hazard (k = 1)")
    plt.title("Weibull k by Task Group (Frontier Models)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task_group_k_forest.png"))
    plt.close()

    return summary_path, group_map_path


def main():
    df, groups, _ = load_grouped_data(DATA_PATH)
    print(f"Analyzing {len(df)} runs across {len(groups)} task groups: {groups}")

    trace = fit_group_weibull(df, groups)
    summary_path, group_map_path = write_outputs(trace, groups)

    print(f"Saved summary: {summary_path}")
    print(f"Saved group map: {group_map_path}")
    print(f"Saved plot: {os.path.join(OUTPUT_DIR, 'task_group_k_forest.png')}")


if __name__ == "__main__":
    main()
