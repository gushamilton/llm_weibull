import os
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "eval-analysis-public" / "data" / "external" / "all_runs.jsonl"
DATA_PATH = Path(os.environ.get("PETO_DATA_FILE", str(DEFAULT_DATA_PATH)))
RESULTS_ROOT = Path(os.environ.get("PETO_RESULTS_ROOT", str(PROJECT_ROOT / "results_mle")))
OUTPUT_DIR = RESULTS_ROOT / "bayes_compare_nc"

DRAWS = int(os.environ.get("PM_DRAWS", "1000"))
TUNE = int(os.environ.get("PM_TUNE", "1000"))
CHAINS = int(os.environ.get("PM_CHAINS", "4"))
TARGET_ACCEPT = float(os.environ.get("PM_TARGET_ACCEPT", "0.97"))
RANDOM_SEED = int(os.environ.get("PM_SEED", "42"))
COMPARE_IC = os.environ.get("PM_COMPARE_IC", "loo")
ONLY_WEIBULL = os.environ.get("PM_ONLY_WEIBULL", "0") == "1"
ONLY_EXPONENTIAL = os.environ.get("PM_ONLY_EXPONENTIAL", "0") == "1"
ONLY_LOGISTIC = os.environ.get("PM_ONLY_LOGISTIC", "0") == "1"


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_json(path, lines=True)
    df = df[df["alias"] != "human"].copy()
    df["t_hours"] = df["human_minutes"] / 60.0
    df = df[df["t_hours"] > 0].copy()

    # Standardize log-time to reduce curvature
    df["log_t"] = np.log(df["t_hours"])
    df["log_t_z"] = (df["log_t"] - df["log_t"].mean()) / df["log_t"].std(ddof=0)

    models = sorted(df["alias"].unique())
    model_map = {name: idx for idx, name in enumerate(models)}
    df["model_idx"] = df["alias"].map(model_map)

    return df, models


def fit_models(df, model_names):
    n_models = len(model_names)
    model_idx = df["model_idx"].values
    t = df["t_hours"].values
    log_tz = df["log_t_z"].values
    y = df["score_binarized"].values

    results = {}

    if not (ONLY_EXPONENTIAL or ONLY_LOGISTIC):
        # Weibull (non-centered)
        with pm.Model() as m_weibull:
            mu_k = pm.Normal("mu_k", 0.0, 0.4)  # log-k
            sigma_k = pm.HalfNormal("sigma_k", 0.4)
            k_raw = pm.Normal("k_raw", 0.0, 1.0, shape=n_models)
            k = pm.Deterministic("k", pm.math.exp(mu_k + sigma_k * k_raw))

            mu_lam = pm.Normal("mu_lam", -2.0, 0.7)  # log-lam
            sigma_lam = pm.HalfNormal("sigma_lam", 0.7)
            lam_raw = pm.Normal("lam_raw", 0.0, 1.0, shape=n_models)
            lam = pm.Deterministic("lam", pm.math.exp(mu_lam + sigma_lam * lam_raw))

            mu_c = pm.Normal("mu_c", 2.5, 0.7)
            sigma_c = pm.HalfNormal("sigma_c", 0.7)
            c_raw = pm.Normal("c_raw", 0.0, 1.0, shape=n_models)
            c = pm.Deterministic("c", pm.math.invlogit(mu_c + sigma_c * c_raw))

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

    if ONLY_WEIBULL:
        return results

    if ONLY_EXPONENTIAL and not ONLY_LOGISTIC:
        # Exponential only
        with pm.Model() as m_exp:
            mu_lam = pm.Normal("mu_lam", -2.0, 0.7)
            sigma_lam = pm.HalfNormal("sigma_lam", 0.7)
            lam_raw = pm.Normal("lam_raw", 0.0, 1.0, shape=n_models)
            lam = pm.Deterministic("lam", pm.math.exp(mu_lam + sigma_lam * lam_raw))

            mu_c = pm.Normal("mu_c", 2.5, 0.7)
            sigma_c = pm.HalfNormal("sigma_c", 0.7)
            c_raw = pm.Normal("c_raw", 0.0, 1.0, shape=n_models)
            c = pm.Deterministic("c", pm.math.invlogit(mu_c + sigma_c * c_raw))

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
        return results

    if ONLY_LOGISTIC and not ONLY_EXPONENTIAL:
        # Logistic only
        with pm.Model() as m_log:
            mu_b0 = pm.Normal("mu_b0", 1.0, 0.8)
            sigma_b0 = pm.HalfNormal("sigma_b0", 0.8)
            b0_raw = pm.Normal("b0_raw", 0.0, 1.0, shape=n_models)
            b0 = pm.Deterministic("b0", mu_b0 + sigma_b0 * b0_raw)

            mu_b1 = pm.Normal("mu_b1", -1.0, 0.6)
            sigma_b1 = pm.HalfNormal("sigma_b1", 0.6)
            b1_raw = pm.Normal("b1_raw", 0.0, 1.0, shape=n_models)
            b1 = pm.Deterministic("b1", mu_b1 + sigma_b1 * b1_raw)

            mu_c = pm.Normal("mu_c", 2.5, 0.7)
            sigma_c = pm.HalfNormal("sigma_c", 0.7)
            c_raw = pm.Normal("c_raw", 0.0, 1.0, shape=n_models)
            c = pm.Deterministic("c", pm.math.invlogit(mu_c + sigma_c * c_raw))

            term = -(b0[model_idx] + b1[model_idx] * log_tz)
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

    # Exponential (non-centered)
    with pm.Model() as m_exp:
        mu_lam = pm.Normal("mu_lam", -2.0, 0.7)
        sigma_lam = pm.HalfNormal("sigma_lam", 0.7)
        lam_raw = pm.Normal("lam_raw", 0.0, 1.0, shape=n_models)
        lam = pm.Deterministic("lam", pm.math.exp(mu_lam + sigma_lam * lam_raw))

        mu_c = pm.Normal("mu_c", 2.5, 0.7)
        sigma_c = pm.HalfNormal("sigma_c", 0.7)
        c_raw = pm.Normal("c_raw", 0.0, 1.0, shape=n_models)
        c = pm.Deterministic("c", pm.math.invlogit(mu_c + sigma_c * c_raw))

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

    # Logistic (non-centered, use standardized log-time)
    with pm.Model() as m_log:
        mu_b0 = pm.Normal("mu_b0", 1.0, 0.8)
        sigma_b0 = pm.HalfNormal("sigma_b0", 0.8)
        b0_raw = pm.Normal("b0_raw", 0.0, 1.0, shape=n_models)
        b0 = pm.Deterministic("b0", mu_b0 + sigma_b0 * b0_raw)

        mu_b1 = pm.Normal("mu_b1", -1.0, 0.6)
        sigma_b1 = pm.HalfNormal("sigma_b1", 0.6)
        b1_raw = pm.Normal("b1_raw", 0.0, 1.0, shape=n_models)
        b1 = pm.Deterministic("b1", mu_b1 + sigma_b1 * b1_raw)

        mu_c = pm.Normal("mu_c", 2.5, 0.7)
        sigma_c = pm.HalfNormal("sigma_c", 0.7)
        c_raw = pm.Normal("c_raw", 0.0, 1.0, shape=n_models)
        c = pm.Deterministic("c", pm.math.invlogit(mu_c + sigma_c * c_raw))

        term = -(b0[model_idx] + b1[model_idx] * log_tz)
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


def plot_k_forest(trace):
    az.plot_forest(trace, var_names=["k"], combined=True)
    plt.axvline(1.0, color="red", linestyle="--", label="Constant Hazard (k = 1)")
    plt.title("Weibull k by Model (AI Only, Non-centered)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "peto_coefficient_forest.png")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, models = load_data(DATA_PATH)
    results = fit_models(df, models)

    if len(results) > 1:
        cmp_df = compare_models(results)
        cmp_path = OUTPUT_DIR / f"bayesian_model_comparison_{COMPARE_IC}.csv"
        cmp_df.to_csv(cmp_path)
        print(cmp_path)

    if "Weibull (Varying)" in results:
        plot_k_forest(results["Weibull (Varying)"][1])
        trace_path = OUTPUT_DIR / "weibull_trace.nc"
        results["Weibull (Varying)"][1].to_netcdf(trace_path)
        print(OUTPUT_DIR / "peto_coefficient_forest.png")
        print(trace_path)
    if "Exponential (Constant)" in results:
        exp_trace = OUTPUT_DIR / "exponential_trace.nc"
        results["Exponential (Constant)"][1].to_netcdf(exp_trace)
        print(exp_trace)
    if "Logistic (Standard)" in results:
        log_trace = OUTPUT_DIR / "logistic_trace.nc"
        results["Logistic (Standard)"][1].to_netcdf(log_trace)
        print(log_trace)


if __name__ == "__main__":
    main()
