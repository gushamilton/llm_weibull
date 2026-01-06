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
OUTPUT_DIR = RESULTS_ROOT / 'bayes_compare'

DRAWS = int(os.environ.get("PM_DRAWS", "800"))
TUNE = int(os.environ.get("PM_TUNE", "800"))
CHAINS = int(os.environ.get("PM_CHAINS", "4"))
TARGET_ACCEPT = float(os.environ.get("PM_TARGET_ACCEPT", "0.95"))
RANDOM_SEED = int(os.environ.get("PM_SEED", "42"))
COMPARE_IC = os.environ.get("PM_COMPARE_IC", "loo")


def load_data(path):
    df = pd.read_json(path, lines=True)

    # Split: AI hierarchy vs human benchmark (human excluded from hierarchy)
    ai_df = df[df["alias"] != "human"].copy()

    ai_df["t"] = ai_df["human_minutes"] / 60.0
    ai_df = ai_df[ai_df["t"] > 0].copy()

    models = sorted(ai_df["alias"].unique())
    model_map = {name: idx for idx, name in enumerate(models)}
    ai_df["model_idx"] = ai_df["alias"].map(model_map)

    return ai_df, models


def fit_models(df, model_names):
    n_models = len(model_names)
    model_idx = df["model_idx"].values
    t = df["t"].values
    y = df["score_binarized"].values

    print(f"Fitting models on {len(df)} runs across {n_models} AI models...")

    results = {}

    # Weibull (varying hazard)
    with pm.Model() as m_weibull:
        mu_k = pm.Normal("mu_k", 1.0, 0.5)
        k = pm.math.exp(pm.Normal("k_log", mu_k, 0.5, shape=n_models))
        pm.Deterministic("k", k)

        mu_lam = pm.Normal("mu_lam", np.log(0.1), 1.0)
        lam = pm.math.exp(pm.Normal("lam_log", mu_lam, 1.0, shape=n_models))

        mu_c = pm.Normal("mu_c", 1.5, 1.0)
        c = pm.math.invlogit(pm.Normal("c_logit", mu_c, 1.0, shape=n_models))

        p = c[model_idx] * pm.math.exp(-((lam[model_idx] * t) ** k[model_idx]))
        pm.Bernoulli("y", p=p, observed=y)

        trace_weibull = pm.sample(
            DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    results["Weibull (Varying)"] = (m_weibull, trace_weibull)

    # Exponential (constant hazard, k=1)
    with pm.Model() as m_exp:
        mu_lam = pm.Normal("mu_lam", np.log(0.1), 1.0)
        lam = pm.math.exp(pm.Normal("lam_log", mu_lam, 1.0, shape=n_models))

        mu_c = pm.Normal("mu_c", 1.5, 1.0)
        c = pm.math.invlogit(pm.Normal("c_logit", mu_c, 1.0, shape=n_models))

        p = c[model_idx] * pm.math.exp(-(lam[model_idx] * t))
        pm.Bernoulli("y", p=p, observed=y)

        trace_exp = pm.sample(
            DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    results["Exponential (Constant)"] = (m_exp, trace_exp)

    # Logistic (ceiling logistic)
    with pm.Model() as m_log:
        mu_b0 = pm.Normal("mu_b0", 2.0, 1.0)
        b0 = pm.Normal("b0", mu_b0, 1.0, shape=n_models)

        mu_b1 = pm.Normal("mu_b1", -0.5, 0.5)
        b1 = pm.Normal("b1", mu_b1, 0.5, shape=n_models)

        mu_c = pm.Normal("mu_c", 1.5, 1.0)
        c = pm.math.invlogit(pm.Normal("c_logit", mu_c, 1.0, shape=n_models))

        term = -(b0[model_idx] + b1[model_idx] * np.log(t))
        p = c[model_idx] / (1 + pm.math.exp(term))
        pm.Bernoulli("y", p=p, observed=y)

        trace_log = pm.sample(
            DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    results["Logistic (Standard)"] = (m_log, trace_log)

    return results


def compare_models(results):
    cmp_df = az.compare({k: v[1] for k, v in results.items()}, ic=COMPARE_IC, method="stacking")
    return cmp_df


def plot_k_forest(trace, model_names):
    az.plot_forest(trace, var_names=["k"], combined=True)
    plt.axvline(1.0, color="red", linestyle="--", label="Constant Hazard (k = 1)")
    plt.title("Weibull k by Model (AI Only)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "peto_coefficient_forest.png"))
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ai_df, models = load_data(DATA_PATH)
    results = fit_models(ai_df, models)

    cmp_df = compare_models(results)
    cmp_path = os.path.join(OUTPUT_DIR, f"bayesian_model_comparison_{COMPARE_IC}.csv")
    cmp_df.to_csv(cmp_path)

    plot_k_forest(results["Weibull (Varying)"][1], models)

    print("Saved:")
    print(cmp_path)
    print(os.path.join(OUTPUT_DIR, "peto_coefficient_forest.png"))


if __name__ == "__main__":
    main()
