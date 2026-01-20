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
OUTPUT_DIR = RESULTS_ROOT / 'bayes_posteriors'

DRAWS = int(os.environ.get("PM_DRAWS", "1000"))
TUNE = int(os.environ.get("PM_TUNE", "1000"))
CHAINS = int(os.environ.get("PM_CHAINS", "4"))
TARGET_ACCEPT = float(os.environ.get("PM_TARGET_ACCEPT", "0.95"))
RANDOM_SEED = int(os.environ.get("PM_SEED", "42"))


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_json(path, lines=True)
    df = df[df["alias"] != "human"].copy()

    df["t"] = df["human_minutes"] / 60.0
    df = df[df["t"] > 0].copy()

    models = sorted(df["alias"].unique().tolist())
    model_map = {name: idx for idx, name in enumerate(models)}
    df["model_idx"] = df["alias"].map(model_map)

    return df, models


def fit_hierarchical_weibull(df, models):
    n_models = len(models)
    model_idx = df["model_idx"].values
    t = df["t"].values
    y = df["score_binarized"].values

    with pm.Model() as model:
        mu_k = pm.Normal("mu_k", 0.0, 0.5)
        sigma_k = pm.HalfNormal("sigma_k", 0.5)
        k_model = pm.Normal("k_model", mu=mu_k, sigma=sigma_k, shape=n_models)
        k = pm.Deterministic("k", pm.math.exp(k_model))

        mu_lam = pm.Normal("mu_lam", np.log(0.1), 1.0)
        sigma_lam = pm.HalfNormal("sigma_lam", 1.0)
        lam_model = pm.Normal("lam_model", mu=mu_lam, sigma=sigma_lam, shape=n_models)
        lam = pm.Deterministic("lam", pm.math.exp(lam_model))

        mu_c = pm.Normal("mu_c", 2.944, 1.0)
        sigma_c = pm.HalfNormal("sigma_c", 1.0)
        c_model = pm.Normal("c_model", mu=mu_c, sigma=sigma_c, shape=n_models)
        c = pm.Deterministic("c", pm.math.invlogit(c_model))

        hazard = pm.math.exp(-((lam[model_idx] * t) ** k[model_idx]))
        p = c[model_idx] * hazard

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


def write_outputs(trace, models):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = az.summary(trace, var_names=["k", "lam"], hdi_prob=0.95)
    summary_path = os.path.join(OUTPUT_DIR, "posterior_k_lambda_summary.csv")
    summary.to_csv(summary_path)

    model_map_path = os.path.join(OUTPUT_DIR, "model_index_map.csv")
    pd.DataFrame({"model_idx": list(range(len(models))), "alias": models}).to_csv(
        model_map_path, index=False
    )

    # Build tidy data for plotting from saved files to avoid index surprises
    summary = pd.read_csv(summary_path, index_col=0)
    model_map = pd.read_csv(model_map_path)

    k_rows = summary[summary.index.str.startswith("k[")].copy()
    k_rows["model_idx"] = k_rows.index.str.extract(r"k\[(\d+)\]")[0].astype(int).to_numpy()
    k_rows = k_rows.merge(model_map, on="model_idx", how="left")

    lam_rows = summary[summary.index.str.startswith("lam[")].copy()
    lam_rows["model_idx"] = lam_rows.index.str.extract(r"lam\[(\d+)\]")[0].astype(int).to_numpy()
    lam_rows = lam_rows.merge(model_map, on="model_idx", how="left")

    if k_rows["alias"].isna().any() or lam_rows["alias"].isna().any():
        missing = k_rows[k_rows["alias"].isna()]["model_idx"].tolist()
        raise ValueError(f"Missing model aliases for indices: {missing}")

    # Plot k (forest-style points with 95% CI)
    k_rows = k_rows.sort_values(by="mean")
    plt.figure(figsize=(10, 7))
    plt.errorbar(
        k_rows["mean"],
        k_rows["alias"],
        xerr=[k_rows["mean"] - k_rows["hdi_2.5%"], k_rows["hdi_97.5%"] - k_rows["mean"]],
        fmt="o",
        color="#1f77b4",
        ecolor="#7fb3d5",
        capsize=3,
    )
    plt.axvline(1.0, color="#d62728", linestyle="--", linewidth=2, label="k = 1 (Constant Hazard)")
    plt.xlabel("Weibull Shape (k)")
    plt.ylabel("Model Alias")
    plt.title("Weibull Shape Parameter by Model (Posterior 95% CI, AI Only)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    k_plot_path = os.path.join(OUTPUT_DIR, "posterior_k_by_model.png")
    plt.savefig(k_plot_path)
    plt.close()

    # Plot lambda (log scale)
    lam_rows = lam_rows.sort_values(by="mean")
    plt.figure(figsize=(10, 7))
    plt.errorbar(
        lam_rows["mean"],
        lam_rows["alias"],
        xerr=[lam_rows["mean"] - lam_rows["hdi_2.5%"], lam_rows["hdi_97.5%"] - lam_rows["mean"]],
        fmt="o",
        color="#2ca02c",
        ecolor="#a3e4a7",
        capsize=3,
    )
    plt.xscale("log")
    plt.xlabel("Weibull Scale (lambda) [Log Scale]")
    plt.ylabel("Model Alias")
    plt.title("Weibull Lambda by Model (Posterior 95% CI, AI Only)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    lam_plot_path = os.path.join(OUTPUT_DIR, "posterior_lambda_by_model.png")
    plt.savefig(lam_plot_path)
    plt.close()

    return summary_path, k_plot_path, lam_plot_path, model_map_path


def main():
    df, models = load_data(DATA_PATH)
    print(f"Analyzing {len(df)} runs across {len(models)} models (human excluded).")

    trace = fit_hierarchical_weibull(df, models)
    summary_path, k_plot_path, lam_plot_path, model_map_path = write_outputs(trace, models)
    trace_path = OUTPUT_DIR / "posterior_trace.nc"
    trace.to_netcdf(trace_path)

    print(f"Saved summary: {summary_path}")
    print(f"Saved model map: {model_map_path}")
    print(f"Saved plot: {k_plot_path}")
    print(f"Saved plot: {lam_plot_path}")
    print(f"Saved trace: {trace_path}")


if __name__ == "__main__":
    main()
