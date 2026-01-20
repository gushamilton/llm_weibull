import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mle_utils import bic_from_ll, fit_logistic_mle, fit_weibull_mle

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "eval-analysis-public" / "data" / "external" / "all_runs.jsonl"
DATA_PATH = Path(os.environ.get("PETO_DATA_FILE", str(DEFAULT_DATA_PATH)))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results_mle"
OUTPUT_DIR = Path(os.environ.get("PETO_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))

BOOTSTRAP_N = int(os.environ.get("BOOTSTRAP_N", "300"))
SEED = int(os.environ.get("BOOTSTRAP_SEED", "42"))
MIN_RUNS = int(os.environ.get("PETO_MIN_N", "50"))

EXCLUDE_ALIASES = {"gpt-3.5-turbo-instruct"}


def _mask_positive_time(t: np.ndarray, y: np.ndarray):
    mask = np.isfinite(t) & (t > 0)
    return t[mask], y[mask]


def _bootstrap_ci(vals: np.ndarray, alpha: float = 0.05):
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return lo, hi


def _fit_models(t: np.ndarray, y: np.ndarray):
    log_res = fit_logistic_mle(t, y)
    weib_res = fit_weibull_mle(t, y)
    return log_res, weib_res


def _aic_from_ll(ll: float, n_params: int) -> float:
    return float(2 * n_params - 2.0 * ll)


def _plot_diff_ci(df: pd.DataFrame, base_col: str, out_path: Path, title: str):
    sdf = df.sort_values(by=base_col, ascending=True).reset_index(drop=True)
    y = np.arange(len(sdf))

    plt.figure(figsize=(8, max(6, 0.35 * len(sdf))))
    plt.errorbar(
        sdf[base_col],
        y,
        xerr=[
            sdf[base_col] - sdf[f"{base_col}_ci_low"],
            sdf[f"{base_col}_ci_high"] - sdf[base_col],
        ],
        fmt="o",
        markersize=4,
        capsize=2,
    )
    plt.axvline(0.0, color="red", linestyle="--", linewidth=1)
    plt.yticks(y, sdf["alias"])
    plt.xlabel(base_col)
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}. Set PETO_DATA_FILE.")

    df = pd.read_json(DATA_PATH, lines=True)
    required_cols = ["alias", "human_minutes", "score_binarized"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in dataset.")

    rng = np.random.default_rng(SEED)
    rows = []

    aliases = pd.Series(df["alias"].dropna().unique()).sort_values().tolist()
    for alias in aliases:
        if alias in EXCLUDE_ALIASES:
            continue

        subset = df[df["alias"] == alias][required_cols].dropna().copy()
        t = subset["human_minutes"].to_numpy(dtype=float)
        y = subset["score_binarized"].to_numpy(dtype=float)
        t, y = _mask_positive_time(t, y)

        n_runs = len(t)
        if n_runs < MIN_RUNS:
            continue

        # Point estimate on full data
        log_res, weib_res = _fit_models(t, y)
        if not log_res.success or not weib_res.success:
            continue

        log_bic = bic_from_ll(log_res.ll, n_params=2, n_obs=n_runs)
        weib_bic = bic_from_ll(weib_res.ll, n_params=2, n_obs=n_runs)
        log_aic = _aic_from_ll(log_res.ll, n_params=2)
        weib_aic = _aic_from_ll(weib_res.ll, n_params=2)

        # Bootstrap by run
        b_log_b0 = []
        b_log_b1 = []
        b_weib_lam = []
        b_weib_k = []
        b_bic_diff = []
        b_aic_diff = []
        b_bic_w_log = []
        b_aic_w_log = []
        ok_log = 0
        ok_weib = 0
        ok_both = 0

        for _ in range(BOOTSTRAP_N):
            idx = rng.integers(0, n_runs, size=n_runs)
            tb = t[idx]
            yb = y[idx]

            log_b, weib_b = _fit_models(tb, yb)
            if log_b.success:
                ok_log += 1
                b0, b1 = log_b.params
                b_log_b0.append(b0)
                b_log_b1.append(b1)
            if weib_b.success:
                ok_weib += 1
                lam, k = weib_b.params
                b_weib_lam.append(lam)
                b_weib_k.append(k)
            if log_b.success and weib_b.success:
                ok_both += 1
                bic_log = bic_from_ll(log_b.ll, n_params=2, n_obs=len(tb))
                bic_weib = bic_from_ll(weib_b.ll, n_params=2, n_obs=len(tb))
                aic_log = _aic_from_ll(log_b.ll, n_params=2)
                aic_weib = _aic_from_ll(weib_b.ll, n_params=2)
                diff_bic = bic_log - bic_weib
                diff_aic = aic_log - aic_weib
                b_bic_diff.append(diff_bic)
                b_aic_diff.append(diff_aic)
                b_bic_w_log.append(1.0 / (1.0 + np.exp(0.5 * diff_bic)))
                b_aic_w_log.append(1.0 / (1.0 + np.exp(0.5 * diff_aic)))

        b_log_b0 = np.asarray(b_log_b0, dtype=float)
        b_log_b1 = np.asarray(b_log_b1, dtype=float)
        b_weib_lam = np.asarray(b_weib_lam, dtype=float)
        b_weib_k = np.asarray(b_weib_k, dtype=float)
        b_bic_diff = np.asarray(b_bic_diff, dtype=float)
        b_aic_diff = np.asarray(b_aic_diff, dtype=float)
        b_bic_w_log = np.asarray(b_bic_w_log, dtype=float)
        b_aic_w_log = np.asarray(b_aic_w_log, dtype=float)

        row = {
            "alias": alias,
            "n_runs": n_runs,
            "logistic_b0": float(log_res.params[0]),
            "logistic_b1": float(log_res.params[1]),
            "weibull_lambda": float(weib_res.params[0]),
            "weibull_k": float(weib_res.params[1]),
            "logistic_bic": log_bic,
            "weibull_bic": weib_bic,
            "logistic_aic": log_aic,
            "weibull_aic": weib_aic,
            "bootstrap_n": BOOTSTRAP_N,
            "bootstrap_success_logistic": int(ok_log),
            "bootstrap_success_weibull": int(ok_weib),
            "bootstrap_success_both": int(ok_both),
        }

        if len(b_log_b0) >= 30:
            row["logistic_b0_ci_low"], row["logistic_b0_ci_high"] = _bootstrap_ci(b_log_b0)
            row["logistic_b1_ci_low"], row["logistic_b1_ci_high"] = _bootstrap_ci(b_log_b1)
        else:
            row["logistic_b0_ci_low"] = row["logistic_b0_ci_high"] = np.nan
            row["logistic_b1_ci_low"] = row["logistic_b1_ci_high"] = np.nan

        if len(b_weib_lam) >= 30:
            row["weibull_lambda_ci_low"], row["weibull_lambda_ci_high"] = _bootstrap_ci(b_weib_lam)
            row["weibull_k_ci_low"], row["weibull_k_ci_high"] = _bootstrap_ci(b_weib_k)
        else:
            row["weibull_lambda_ci_low"] = row["weibull_lambda_ci_high"] = np.nan
            row["weibull_k_ci_low"] = row["weibull_k_ci_high"] = np.nan

        if len(b_bic_diff) >= 30:
            row["bic_diff"] = float(np.median(b_bic_diff))
            row["bic_diff_ci_low"], row["bic_diff_ci_high"] = _bootstrap_ci(b_bic_diff)
            row["bic_logistic_win_rate"] = float(np.mean(b_bic_diff < 0))
            row["bic_weight_logistic"] = float(np.mean(b_bic_w_log))
            row["bic_weight_logistic_ci_low"], row["bic_weight_logistic_ci_high"] = _bootstrap_ci(
                b_bic_w_log
            )
        else:
            row["bic_diff"] = row["bic_diff_ci_low"] = row["bic_diff_ci_high"] = np.nan
            row["bic_logistic_win_rate"] = np.nan
            row["bic_weight_logistic"] = np.nan
            row["bic_weight_logistic_ci_low"] = np.nan
            row["bic_weight_logistic_ci_high"] = np.nan

        if len(b_aic_diff) >= 30:
            row["aic_diff"] = float(np.median(b_aic_diff))
            row["aic_diff_ci_low"], row["aic_diff_ci_high"] = _bootstrap_ci(b_aic_diff)
            row["aic_logistic_win_rate"] = float(np.mean(b_aic_diff < 0))
            row["aic_weight_logistic"] = float(np.mean(b_aic_w_log))
            row["aic_weight_logistic_ci_low"], row["aic_weight_logistic_ci_high"] = _bootstrap_ci(
                b_aic_w_log
            )
        else:
            row["aic_diff"] = row["aic_diff_ci_low"] = row["aic_diff_ci_high"] = np.nan
            row["aic_logistic_win_rate"] = np.nan
            row["aic_weight_logistic"] = np.nan
            row["aic_weight_logistic_ci_low"] = np.nan
            row["aic_weight_logistic_ci_high"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(by="alias")
    out_csv = OUTPUT_DIR / "weibull_logistic_run_bootstrap_ci.csv"
    out.to_csv(out_csv, index=False)

    plot_bic = OUTPUT_DIR / "run_bootstrap_bic_diff.png"
    plot_aic = OUTPUT_DIR / "run_bootstrap_aic_diff.png"
    plot_df = out.dropna(subset=["bic_diff"]).copy()
    if not plot_df.empty:
        _plot_diff_ci(
            plot_df,
            "bic_diff",
            plot_bic,
            "Run-level Bootstrap: BIC(Logistic) - BIC(Weibull)",
        )

    plot_df = out.dropna(subset=["aic_diff"]).copy()
    if not plot_df.empty:
        _plot_diff_ci(
            plot_df,
            "aic_diff",
            plot_aic,
            "Run-level Bootstrap: AIC(Logistic) - AIC(Weibull)",
        )

    print(f"Saved: {out_csv}")
    if plot_bic.exists():
        print(f"Saved: {plot_bic}")
    if plot_aic.exists():
        print(f"Saved: {plot_aic}")


if __name__ == "__main__":
    main()
