import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mle_utils import logistic_prob, weibull_prob, fit_logistic_mle, fit_weibull_mle

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "eval-analysis-public" / "data" / "external" / "all_runs.jsonl"
DATA_PATH = Path(os.environ.get("PETO_DATA_FILE", str(DEFAULT_DATA_PATH)))

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results_mle"
RESULTS_ROOT = Path(os.environ.get("PETO_RESULTS_ROOT", str(DEFAULT_RESULTS_ROOT)))
MODEL_FIT_PATH = RESULTS_ROOT / "model_fit_summary.csv"

OUTPUT_DIR = RESULTS_ROOT / "additional" / "logistic_weibull_diff_curves"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_MULT = float(os.environ.get("PETO_DIFF_MAX_MULT", "10"))
GRID_N = int(os.environ.get("PETO_DIFF_GRID_N", "400"))
START_Q = float(os.environ.get("PETO_DIFF_START_Q", "0"))


def _positive_mask(t: np.ndarray) -> np.ndarray:
    return np.isfinite(t) & (t > 0)


def _time_grid(t: np.ndarray) -> np.ndarray:
    t = t[_positive_mask(t)]
    if t.size == 0:
        return np.array([])
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    t_start = float(np.quantile(t, START_Q)) if START_Q > 0 else t_min
    t_hi = t_max * MAX_MULT
    return np.logspace(np.log10(t_start), np.log10(t_hi), GRID_N)


def _plot_diff(t_grid: np.ndarray, diff: np.ndarray, alias: str, color: str, out_path: Path) -> None:
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(t_grid, diff, color=color, linewidth=2)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Task difficulty (minutes, log scale)")
    plt.ylabel("Weibull - Logistic (probability)")
    plt.title(f"Logistic vs Weibull Difference: {alias}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _save_csv(t_grid: np.ndarray, diff: np.ndarray, out_path: Path) -> None:
    df = pd.DataFrame({"minutes": t_grid, "weibull_minus_logistic": diff})
    df.to_csv(out_path, index=False)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}. Set PETO_DATA_FILE.")
    if not MODEL_FIT_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_FIT_PATH}. Run generate_model_fits.py first.")

    df = pd.read_json(DATA_PATH, lines=True)
    fits = pd.read_csv(MODEL_FIT_PATH)

    # Per-model curves
    for _, row in fits.iterrows():
        alias = row["alias"]
        b0 = row["logistic_b0"]
        b1 = row["logistic_b1"]
        lam = row["weibull_lambda"]
        k = row["weibull_k"]
        if not np.isfinite([b0, b1, lam, k]).all():
            continue

        subset = df[df["alias"] == alias]["human_minutes"].to_numpy(dtype=float)
        t_grid = _time_grid(subset)
        if t_grid.size == 0:
            continue

        p_log = logistic_prob(t_grid, b0, b1)
        p_weib = weibull_prob(t_grid, lam, k)
        diff = p_weib - p_log

        safe_alias = "".join(ch if ch.isalnum() else "_" for ch in alias).lower()
        color = "#d62728" if alias.strip().lower() == "human" else "#1f77b4"
        _plot_diff(t_grid, diff, alias, color, OUTPUT_DIR / f"{safe_alias}_diff.png")
        _save_csv(t_grid, diff, OUTPUT_DIR / f"{safe_alias}_diff.csv")

    # Overall pooled curve (AI only)
    pooled = df[df["alias"].str.lower() != "human"].copy()
    t = pooled["human_minutes"].to_numpy(dtype=float)
    y = pooled["score_binarized"].to_numpy(dtype=float)
    mask = _positive_mask(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    if t.size:
        fit_log = fit_logistic_mle(t, y)
        fit_weib = fit_weibull_mle(t, y)
        if fit_log.success and fit_weib.success:
            t_grid = _time_grid(t)
            p_log = logistic_prob(t_grid, *fit_log.params)
            p_weib = weibull_prob(t_grid, *fit_weib.params)
            diff = p_weib - p_log
            _plot_diff(t_grid, diff, "Overall (AI pooled)", "#1f77b4", OUTPUT_DIR / "overall_ai_pooled_diff.png")
            _save_csv(t_grid, diff, OUTPUT_DIR / "overall_ai_pooled_diff.csv")


if __name__ == "__main__":
    main()
