import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mle_utils import fit_weibull_mle

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "eval-analysis-public" / "data" / "external" / "all_runs.jsonl"
DATA_PATH = Path(os.environ.get("PETO_DATA_FILE", str(DEFAULT_DATA_PATH)))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results_mle"
OUTPUT_DIR = Path(os.environ.get("PETO_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))

BOOTSTRAP_N = int(os.environ.get("BOOTSTRAP_N", "300"))
MIN_N = int(os.environ.get("PETO_MIN_N", "50"))
SEED = int(os.environ.get("BOOTSTRAP_SEED", "42"))

EXCLUDE_ALIASES = {"gpt-3.5-turbo-instruct"}


def _mask_positive_time(t: np.ndarray, y: np.ndarray):
    mask = np.isfinite(t) & (t > 0)
    return t[mask], y[mask]


def _fit_once(t: np.ndarray, y: np.ndarray):
    res = fit_weibull_mle(t, y)
    if not res.success:
        raise ValueError(res.message)
    lam, k = res.params
    return float(lam), float(k)


def _bootstrap_ci(vals: np.ndarray, alpha: float = 0.05):
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return lo, hi


def plot_param_ci(df: pd.DataFrame, param: str, out_path: Path, xscale: str | None = None):
    sdf = df.sort_values(by=param, ascending=True).reset_index(drop=True)
    y = np.arange(len(sdf))

    plt.figure(figsize=(8, max(6, 0.35 * len(sdf))))
    plt.errorbar(
        sdf[param],
        y,
        xerr=[sdf[param] - sdf[f"{param}_ci_low"], sdf[f"{param}_ci_high"] - sdf[param]],
        fmt="o",
        markersize=4,
        capsize=2,
    )
    plt.yticks(y, sdf["alias"])
    if xscale:
        plt.xscale(xscale)
    plt.xlabel(param)
    plt.title(f"Weibull {param} by model (MLE + bootstrap CI)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}. Set PETO_DATA_FILE.")

    df = pd.read_json(DATA_PATH, lines=True)
    if "alias" not in df.columns:
        raise KeyError("Expected column 'alias' not found in dataset.")

    required_cols = ["human_minutes", "score_binarized"]
    aliases = pd.Series(df["alias"].dropna().unique()).sort_values().tolist()

    rng = np.random.default_rng(SEED)

    rows = []
    for alias in aliases:
        if alias in EXCLUDE_ALIASES:
            continue

        subset = df[df["alias"] == alias][required_cols].dropna().copy()
        t = subset["human_minutes"].to_numpy(dtype=float)
        y = subset["score_binarized"].to_numpy(dtype=float)
        t, y = _mask_positive_time(t, y)

        n = len(t)
        if n < MIN_N:
            continue

        # Point estimate
        try:
            lam_hat, k_hat = _fit_once(t, y)
        except Exception:
            lam_hat, k_hat = np.nan, np.nan

        # Bootstrap
        lam_bs = []
        k_bs = []
        if np.isfinite(lam_hat) and np.isfinite(k_hat):
            for _ in range(BOOTSTRAP_N):
                idx = rng.integers(0, n, size=n)
                tb = t[idx]
                yb = y[idx]
                try:
                    lam_b, k_b = _fit_once(tb, yb)
                    lam_bs.append(lam_b)
                    k_bs.append(k_b)
                except Exception:
                    continue

        lam_bs = np.asarray(lam_bs, dtype=float)
        k_bs = np.asarray(k_bs, dtype=float)

        if len(lam_bs) >= 30:
            lam_lo, lam_hi = _bootstrap_ci(lam_bs)
        else:
            lam_lo, lam_hi = np.nan, np.nan

        if len(k_bs) >= 30:
            k_lo, k_hi = _bootstrap_ci(k_bs)
        else:
            k_lo, k_hi = np.nan, np.nan

        rows.append(
            {
                "alias": alias,
                "n": n,
                "lambda": lam_hat,
                "lambda_ci_low": lam_lo,
                "lambda_ci_high": lam_hi,
                "k": k_hat,
                "k_ci_low": k_lo,
                "k_ci_high": k_hi,
                "bootstrap_n": BOOTSTRAP_N,
                "bootstrap_success": int(len(lam_bs)),
            }
        )

    out = pd.DataFrame(rows).sort_values(by="alias")
    out_csv = OUTPUT_DIR / "weibull_params_with_bootstrap_ci.csv"
    out.to_csv(out_csv, index=False)

    plot_k = OUTPUT_DIR / "weibull_k_by_model_ci.png"
    plot_lam = OUTPUT_DIR / "weibull_lambda_by_model_ci.png"
    plot_param_ci(out.dropna(subset=["k"]).copy(), "k", plot_k, xscale=None)
    plot_param_ci(out.dropna(subset=["lambda"]).copy(), "lambda", plot_lam, xscale="log")

    print(f"Saved: {out_csv}")
    print(f"Saved: {plot_k}")
    print(f"Saved: {plot_lam}")


if __name__ == "__main__":
    main()
