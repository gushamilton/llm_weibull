import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mle_utils import logistic_prob, weibull_prob

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "eval-analysis-public" / "data" / "external" / "all_runs.jsonl"
DATA_PATH = Path(os.environ.get("PETO_DATA_FILE", str(DEFAULT_DATA_PATH)))

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results_mle"
RESULTS_ROOT = Path(os.environ.get("PETO_RESULTS_ROOT", str(DEFAULT_RESULTS_ROOT)))
MODEL_FIT_PATH = RESULTS_ROOT / "model_fit_summary.csv"

OUTPUT_DIR = RESULTS_ROOT / "additional"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DELTA = float(os.environ.get("PETO_DIVERGENCE_DELTA", "0.1"))
MAX_MULT = float(os.environ.get("PETO_DIVERGENCE_MAX_MULT", "10"))
GRID_N = int(os.environ.get("PETO_DIVERGENCE_GRID_N", "400"))
START_Q = float(os.environ.get("PETO_DIVERGENCE_START_Q", "0"))


def _positive_mask(t: np.ndarray) -> np.ndarray:
    return np.isfinite(t) & (t > 0)


def _divergence_stats(t_grid: np.ndarray, p_log: np.ndarray, p_weib: np.ndarray, delta: float):
    diff = np.abs(p_log - p_weib)
    idx = np.where(diff >= delta)[0]
    if len(idx) == 0:
        t_div = np.nan
    else:
        t_div = float(t_grid[idx[0]])
    max_idx = int(np.argmax(diff))
    return t_div, float(diff[max_idx]), float(t_grid[max_idx])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}. Set PETO_DATA_FILE.")
    if not MODEL_FIT_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_FIT_PATH}. Run generate_model_fits.py first.")

    df = pd.read_json(DATA_PATH, lines=True)
    fits = pd.read_csv(MODEL_FIT_PATH)

    if "alias" not in df.columns or "human_minutes" not in df.columns:
        raise KeyError("Expected columns 'alias' and 'human_minutes' not found.")

    rows = []
    for _, row in fits.iterrows():
        alias = row["alias"]
        b0 = row["logistic_b0"]
        b1 = row["logistic_b1"]
        lam = row["weibull_lambda"]
        k = row["weibull_k"]
        if not np.isfinite([b0, b1, lam, k]).all():
            continue

        subset = df[df["alias"] == alias]["human_minutes"].to_numpy(dtype=float)
        subset = subset[_positive_mask(subset)]
        if len(subset) == 0:
            continue

        t_min = float(np.min(subset))
        t_max = float(np.max(subset))
        if START_Q > 0:
            t_start = float(np.quantile(subset, START_Q))
        else:
            t_start = t_min
        t_hi = t_max * MAX_MULT

        t_grid = np.logspace(np.log10(t_start), np.log10(t_hi), GRID_N)
        p_log = logistic_prob(t_grid, b0, b1)
        p_weib = weibull_prob(t_grid, lam, k)

        t_div, max_diff, t_at_max = _divergence_stats(t_grid, p_log, p_weib, DELTA)
        reached = np.isfinite(t_div)
        t_plot = t_div if reached else t_at_max
        within = t_plot <= t_max

        rows.append(
            {
                "alias": alias,
                "t_min": t_min,
                "t_start": t_start,
                "t_max": t_max,
                "t_divergence": t_div,
                "t_at_max_diff": t_at_max,
                "max_abs_diff": max_diff,
                "divergence_reached": reached,
                "t_plot": t_plot,
                "within_observed": within,
                "beyond_observed": t_plot > t_max,
            }
        )

    out = pd.DataFrame(rows).sort_values(by="t_divergence")
    out_csv = OUTPUT_DIR / "logistic_weibull_divergence_times.csv"
    out.to_csv(out_csv, index=False)

    # Plot (only if we have positive divergence times)
    plot_df = out.dropna(subset=["t_plot"]).copy()
    plot_df = plot_df[plot_df["t_plot"] > 0].sort_values(by="t_plot").reset_index(drop=True)
    if plot_df.empty:
        print(f"Saved: {out_csv}")
        print("No divergence times found within the search range; plot not generated.")
        return

    y = np.arange(len(plot_df))
    plt.figure(figsize=(9, max(6, 0.35 * len(plot_df))))
    within = plot_df["within_observed"].to_numpy(dtype=bool)
    reached = plot_df["divergence_reached"].to_numpy(dtype=bool)
    plt.scatter(
        plot_df.loc[within & reached, "t_plot"],
        y[within & reached],
        s=30,
        color="#1f77b4",
        label="Reached within observed range",
    )
    plt.scatter(
        plot_df.loc[~within & reached, "t_plot"],
        y[~within & reached],
        s=30,
        facecolors="none",
        edgecolors="#ff7f0e",
        label="Reached beyond observed range",
    )
    plt.scatter(
        plot_df.loc[~reached, "t_plot"],
        y[~reached],
        s=30,
        facecolors="none",
        edgecolors="#2ca02c",
        label="Max diff below threshold",
    )
    plt.xscale("log")
    plt.yticks(y, plot_df["alias"])
    plt.xlabel(f"Minutes to reach |P_log - P_weib| ≥ {DELTA:.2f} (log scale)")
    plt.title("Logistic vs Weibull Divergence Time by Model")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_plot = OUTPUT_DIR / "logistic_weibull_divergence_times.png"
    plt.savefig(out_plot)
    plt.close()

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_plot}")


if __name__ == "__main__":
    main()
