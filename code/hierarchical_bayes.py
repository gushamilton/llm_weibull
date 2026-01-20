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
OUTPUT_DIR = RESULTS_ROOT / 'hierarchical'

TARGET_MODELS = [
    "Claude 3.5 Sonnet (New)",
    "Claude 3.5 Sonnet (Old)",
    "Claude 3 Opus",
    "GPT-4o",
    "GPT-4 Turbo",
    "o1",
    "Claude 3.7 Sonnet",
]

DRAWS = int(os.environ.get("PM_DRAWS", "1000"))
TUNE = int(os.environ.get("PM_TUNE", "1000"))
CHAINS = int(os.environ.get("PM_CHAINS", "4"))
TARGET_ACCEPT = float(os.environ.get("PM_TARGET_ACCEPT", "0.9"))
RANDOM_SEED = int(os.environ.get("PM_SEED", "42"))


def load_and_prep_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}")

    df = pd.read_json(path, lines=True)
    if "alias" not in df.columns:
        raise KeyError("Expected column 'alias' not found in dataset.")

    df = df[df["alias"].isin(TARGET_MODELS)].copy()

    if df.empty:
        raise ValueError("No rows found for TARGET_MODELS.")

    models = sorted(df["alias"].unique().tolist())
    model_map = {name: idx for idx, name in enumerate(models)}
    df["model_idx"] = df["alias"].map(model_map)

    df["time_hours"] = df["human_minutes"] / 60.0
    df = df[df["time_hours"] > 0].copy()

    return df, models, model_map


def run_bayesian_model(df, model_names):
    n_models = len(model_names)
    model_indices = df["model_idx"].values
    t_obs = df["time_hours"].values
    y_obs = df["score_binarized"].values

    print(f"Fitting hierarchical model on {len(df)} tasks across {n_models} models...")

    with pm.Model() as hierarchical_model:
        # Priors on log-scale so exp() centers k, lambda near desired values.
        mu_k = pm.Normal("mu_k", mu=0.0, sigma=0.5)
        sigma_k = pm.HalfNormal("sigma_k", sigma=0.5)

        mu_lam = pm.Normal("mu_lam", mu=np.log(0.1), sigma=1.0)
        sigma_lam = pm.HalfNormal("sigma_lam", sigma=1.0)

        mu_c = pm.Normal("mu_c", mu=2.944, sigma=1.0)
        sigma_c = pm.HalfNormal("sigma_c", sigma=1.0)

        k_model = pm.Normal("k_model", mu=mu_k, sigma=sigma_k, shape=n_models)
        k = pm.Deterministic("k", pm.math.exp(k_model))

        lam_model = pm.Normal("lam_model", mu=mu_lam, sigma=sigma_lam, shape=n_models)
        lam = pm.Deterministic("lam", pm.math.exp(lam_model))

        c_model = pm.Normal("c_model", mu=mu_c, sigma=sigma_c, shape=n_models)
        c = pm.Deterministic("c", pm.math.invlogit(c_model))

        hazard_term = pm.math.exp(-((lam[model_indices] * t_obs) ** k[model_indices]))
        p_success = c[model_indices] * hazard_term

        pm.Bernoulli("y", p=p_success, observed=y_obs)

        trace = pm.sample(
            DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
        )

    return trace


def plot_results(trace, models):
    summary = az.summary(trace, var_names=["k", "lam", "c"], hdi_prob=0.95)

    summary_path = os.path.join(OUTPUT_DIR, "posterior_summary.csv")
    summary.to_csv(summary_path)

    k_draws = trace.posterior["k"].values
    k_gt_1 = (k_draws > 1).mean(axis=(0, 1))

    prob_df = pd.DataFrame({"alias": models, "p_k_gt_1": k_gt_1})
    prob_path = os.path.join(OUTPUT_DIR, "p_k_gt_1_by_model.csv")
    prob_df.to_csv(prob_path, index=False)

    global_prob = (k_draws > 1).mean()
    with open(os.path.join(OUTPUT_DIR, "p_k_gt_1_global.txt"), "w") as f:
        f.write(f"{global_prob:.6f}\n")

    az.plot_forest(trace, var_names=["k"], combined=True)
    plt.axvline(1.0, color="red", linestyle="--", label="Constant Hazard (k = 1)")
    plt.title("Weibull Shape (k) by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "hierarchical_k_forest.png"))
    plt.close()

    az.plot_forest(trace, var_names=["c"], combined=True)
    plt.title("Reliability Ceiling (c) by Model")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "hierarchical_c_forest.png"))
    plt.close()

    return summary_path, prob_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df, models, _ = load_and_prep_data(DATA_PATH)
    print(f"Loaded {len(df)} runs for {len(models)} models.")

    trace = run_bayesian_model(df, models)
    summary_path, prob_path = plot_results(trace, models)
    trace_path = OUTPUT_DIR / "posterior_trace.nc"
    trace.to_netcdf(trace_path)

    print(f"Saved posterior summary: {summary_path}")
    print(f"Saved P(k>1) table: {prob_path}")
    print(f"Saved plots to: {OUTPUT_DIR}")
    print(f"Saved trace: {trace_path}")


if __name__ == "__main__":
    main()
