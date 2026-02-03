#!/usr/bin/env python3
"""Regenerate BIC diff plot with tighter labels using existing CSV data."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results_mle"

def plot_bic_diff(df: pd.DataFrame, out_path: Path):
    """Plot ΔBIC with bootstrap CIs and region annotations."""
    sdf = df.sort_values(by="bic_diff", ascending=True).reset_index(drop=True)
    y = np.arange(len(sdf))

    fig, ax = plt.subplots(figsize=(7, max(5, 0.38 * len(sdf))))
    
    ax.errorbar(
        sdf["bic_diff"],
        y,
        xerr=[
            sdf["bic_diff"] - sdf["bic_diff_ci_low"],
            sdf["bic_diff_ci_high"] - sdf["bic_diff"],
        ],
        fmt="o",
        markersize=5,
        capsize=3,
        color="#2563eb",
        ecolor="#93c5fd",
        elinewidth=1.5,
    )
    
    ax.axvline(0.0, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(sdf["alias"], fontsize=9)
    ax.set_xlabel("ΔBIC (Logistic − Weibull)", fontsize=10)
    ax.set_title("Model Comparison: BIC Difference", fontsize=11, fontweight="medium")
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")
    
    # Add region annotations at top
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[0] + 0.03 * (xlim[1] - xlim[0]), ylim[1] - 0.02 * (ylim[1] - ylim[0]),
            "← Logistic better", fontsize=8, color="#6b7280", va="top", style="italic")
    ax.text(xlim[1] - 0.03 * (xlim[1] - xlim[0]), ylim[1] - 0.02 * (ylim[1] - ylim[0]),
            "Weibull better →", fontsize=8, color="#6b7280", va="top", ha="right", style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    csv_path = OUTPUT_DIR / "weibull_logistic_run_bootstrap_ci.csv"
    df = pd.read_csv(csv_path)
    plot_df = df.dropna(subset=["bic_diff"]).copy()
    
    if not plot_df.empty:
        plot_bic_diff(plot_df, OUTPUT_DIR / "run_bootstrap_bic_diff.png")


if __name__ == "__main__":
    main()
