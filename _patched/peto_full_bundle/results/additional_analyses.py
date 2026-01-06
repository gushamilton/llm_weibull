import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

from mle_utils import fit_weibull_mle

# -----------------------------
# Paths / config
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "eval-analysis-public" / "data" / "external" / "all_runs.jsonl"
DATA_PATH = Path(os.environ.get("PETO_DATA_FILE", str(DEFAULT_DATA_PATH)))

# Prefer MLE refit outputs if present
DEFAULT_WEIBULL_CSV_MLE = PROJECT_ROOT / "results_mle" / "weibull_params_with_bootstrap_ci.csv"
DEFAULT_WEIBULL_CSV_OLD = PROJECT_ROOT / "results" / "weibull_params_with_bootstrap_ci.csv"
WEIBULL_CSV = Path(os.environ.get(
    "PETO_WEIBULL_CSV",
    str(DEFAULT_WEIBULL_CSV_MLE if DEFAULT_WEIBULL_CSV_MLE.exists() else DEFAULT_WEIBULL_CSV_OLD),
))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results_mle" / "additional"
OUTPUT_DIR = Path(os.environ.get("PETO_ADDITIONAL_DIR", str(DEFAULT_OUTPUT_DIR)))

BURNIN_MINUTES = float(os.environ.get("BURNIN_MINUTES", "30"))
MIN_N = int(os.environ.get("PETO_MIN_N", "50"))
TASK_GROUPS = ["SWAA", "HCAST", "RE-Bench"]

PM_DRAWS = int(os.environ.get("PM_DRAWS", "600"))
PM_TUNE = int(os.environ.get("PM_TUNE", "600"))
PM_CHAINS = int(os.environ.get("PM_CHAINS", "2"))
PM_TARGET_ACCEPT = float(os.environ.get("PM_TARGET_ACCEPT", "0.9"))
PM_SEED = int(os.environ.get("PM_SEED", "42"))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _mask_positive_time(t: np.ndarray, y: np.ndarray):
    """Safely filter to t>0 while preserving alignment."""
    mask = np.isfinite(t) & (t > 0)
    return t[mask], y[mask]


def fit_weibull(t: np.ndarray, y: np.ndarray):
    """Weibull fit using Bernoulli MLE.

    Returns (lambda, k) or raises ValueError if the optimizer fails.
    """
    res = fit_weibull_mle(t, y)
    if not res.success:
        raise ValueError(res.message)
    lam, k = res.params
    return lam, k


def peto_curve(df: pd.DataFrame):
    cap = df.groupby("alias")["score_binarized"].mean().reset_index()
    cap = cap[cap["alias"] != "human"].copy()
    cap.rename(columns={"score_binarized": "capability"}, inplace=True)

    weib = pd.read_csv(WEIBULL_CSV)
    weib = weib[weib["alias"] != "human"].copy()

    merged = cap.merge(weib[["alias", "lambda"]], on="alias", how="inner")
    merged["log_lambda"] = np.log(merged["lambda"])

    # Linear fit for visualization
    x = merged["capability"].values
    y = merged["log_lambda"].values
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0.0, y.mean() if len(y) else 0.0

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="#2ca02c")
    x_line = np.linspace(x.min(), x.max(), 100) if len(x) else np.array([0, 1])
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color="#d62728", linewidth=2)
    plt.xlabel("Capability Proxy (Mean Success Rate)")
    plt.ylabel("log(lambda)")
    plt.title("Peto Curve: log(lambda) vs Capability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "peto_curve_log_lambda_vs_capability.png"
    plt.savefig(plot_path)
    plt.close()

    out_path = OUTPUT_DIR / "peto_curve_data.csv"
    merged.to_csv(out_path, index=False)

    return str(out_path), str(plot_path)


def burnin_test(df: pd.DataFrame):
    results = []
    for alias, gdf in df.groupby("alias"):
        if alias == "human":
            continue
        t = gdf["human_minutes"].to_numpy(dtype=float)
        y = gdf["score_binarized"].to_numpy(dtype=float)

        t, y = _mask_positive_time(t, y)
        if len(t) < MIN_N:
            continue

        try:
            lam_full, k_full = fit_weibull(t, y)
        except Exception:
            continue

        mask = t >= BURNIN_MINUTES
        t_trunc = t[mask]
        y_trunc = y[mask]
        if len(t_trunc) < MIN_N:
            continue

        try:
            lam_trunc, k_trunc = fit_weibull(t_trunc, y_trunc)
        except Exception:
            continue

        results.append(
            {
                "alias": alias,
                "n_full": len(t),
                "n_trunc": len(t_trunc),
                "lambda_full": lam_full,
                "k_full": k_full,
                "lambda_trunc": lam_trunc,
                "k_trunc": k_trunc,
            }
        )

    out = pd.DataFrame(results).sort_values(by="alias")
    out_path = OUTPUT_DIR / "burnin_weibull_comparison.csv"
    out.to_csv(out_path, index=False)

    # Plot k shift
    plt.figure(figsize=(8, 6))
    for _, row in out.iterrows():
        plt.plot([0, 1], [row["k_full"], row["k_trunc"]], marker="o", color="#1f77b4", alpha=0.6)
    plt.axhline(1.0, color="#d62728", linestyle="--", linewidth=2)
    plt.xticks([0, 1], ["Full", f">= {int(BURNIN_MINUTES)} min"])
    plt.ylabel("Weibull k (MLE)")
    plt.title("Burn-in Test: k Before vs After Truncation")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "burnin_k_shift.png"
    plt.savefig(plot_path)
    plt.close()

    return str(out_path), str(plot_path)


def task_group_stratification(df: pd.DataFrame):
    rows = []
    for group in TASK_GROUPS:
        gdf = df[df["task_source"] == group]
        for alias, mdf in gdf.groupby("alias"):
            if alias == "human":
                continue
            t = mdf["human_minutes"].to_numpy(dtype=float)
            y = mdf["score_binarized"].to_numpy(dtype=float)

            t, y = _mask_positive_time(t, y)
            if len(t) < MIN_N:
                continue
            try:
                lam, k = fit_weibull(t, y)
            except Exception:
                continue

            rows.append(
                {
                    "task_source": group,
                    "alias": alias,
                    "n": len(t),
                    "lambda": lam,
                    "k": k,
                }
            )

    out = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "task_group_model_weibull.csv"
    out.to_csv(out_path, index=False)

    # Plot lambda by group
    fig, axes = plt.subplots(1, len(TASK_GROUPS), figsize=(14, 6), sharey=True)
    for ax, group in zip(axes, TASK_GROUPS):
        sdf = out[out["task_source"] == group].copy().sort_values(by="lambda")
        ax.scatter(sdf["lambda"], sdf["alias"], color="#2ca02c")
        ax.set_xscale("log")
        ax.set_title(group)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_xlabel("lambda (log)")
    axes[0].set_ylabel("Model Alias")
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "task_group_lambda_by_model.png"
    plt.savefig(plot_path)
    plt.close()

    return str(out_path), str(plot_path)


def bayesian_covariate(df: pd.DataFrame):
    df = df[df["alias"] != "human"].copy()
    cap = df.groupby("alias")["score_binarized"].mean().reset_index()
    cap.rename(columns={"score_binarized": "capability"}, inplace=True)

    models = sorted(df["alias"].unique().tolist())
    model_map = {name: idx for idx, name in enumerate(models)}
    df["model_idx"] = df["alias"].map(model_map)

    cap = cap.set_index("alias").loc[models].reset_index()
    cap["cap_z"] = (cap["capability"] - cap["capability"].mean()) / cap["capability"].std(ddof=0)

    model_idx = df["model_idx"].values
    t = (df["human_minutes"] / 60.0).values
    y = df["score_binarized"].values

    with pm.Model() as model:
        mu_k = pm.Normal("mu_k", 0.0, 0.5)
        sigma_k = pm.HalfNormal("sigma_k", 0.5)
        k_model = pm.Normal("k_model", mu=mu_k, sigma=sigma_k, shape=len(models))
        k = pm.Deterministic("k", pm.math.exp(k_model))

        alpha = pm.Normal("alpha", np.log(0.1), 1.0)
        beta = pm.Normal("beta", 0.0, 1.0)
        sigma_lam = pm.HalfNormal("sigma_lam", 1.0)
        lam_model = pm.Normal("lam_model", mu=alpha + beta * cap["cap_z"].values, sigma=sigma_lam, shape=len(models))
        lam = pm.Deterministic("lam", pm.math.exp(lam_model))

        mu_c = pm.Normal("mu_c", 2.944, 1.0)
        sigma_c = pm.HalfNormal("sigma_c", 1.0)
        c_model = pm.Normal("c_model", mu=mu_c, sigma=sigma_c, shape=len(models))
        c = pm.Deterministic("c", pm.math.invlogit(c_model))

        hazard = pm.math.exp(-((lam[model_idx] * t) ** k[model_idx]))
        p = c[model_idx] * hazard
        pm.Bernoulli("y", p=p, observed=y)

        trace = pm.sample(
            PM_DRAWS,
            tune=PM_TUNE,
            chains=PM_CHAINS,
            target_accept=PM_TARGET_ACCEPT,
            random_seed=PM_SEED,
            return_inferencedata=True,
        )

    summary = az.summary(trace, var_names=["beta"], hdi_prob=0.95)
    beta_path = OUTPUT_DIR / "bayes_capability_beta_summary.csv"
    summary.to_csv(beta_path)

    beta_draws = trace.posterior["beta"].values.flatten()
    prob_neg = (beta_draws < 0).mean()
    with open(OUTPUT_DIR / "bayes_capability_beta_pneg.txt", "w") as f:
        f.write(f"{prob_neg:.6f}\n")

    plt.figure(figsize=(6, 4))
    plt.hist(beta_draws, bins=40, color="#1f77b4", alpha=0.8)
    plt.axvline(0.0, color="#d62728", linestyle="--", linewidth=2)
    plt.xlabel("beta (capability -> log(lambda))")
    plt.ylabel("count")
    plt.title("Posterior of beta")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "bayes_capability_beta_posterior.png"
    plt.savefig(plot_path)
    plt.close()

    cap_path = OUTPUT_DIR / "capability_proxy_by_model.csv"
    cap.to_csv(cap_path, index=False)

    return str(beta_path), str(plot_path), str(cap_path)


def main():
    ensure_dir(OUTPUT_DIR)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}. Set PETO_DATA_FILE.")
    if not WEIBULL_CSV.exists():
        raise FileNotFoundError(
            f"Weibull CSV not found: {WEIBULL_CSV}.\n"
            f"Run bootstrap_weibull_params_mle.py (or set PETO_WEIBULL_CSV)."
        )

    df = pd.read_json(DATA_PATH, lines=True)

    peto_curve(df)
    burnin_test(df)
    task_group_stratification(df)
    bayesian_covariate(df)


if __name__ == "__main__":
    main()
