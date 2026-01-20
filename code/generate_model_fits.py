import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mle_utils import (
    bic_from_ll,
    fit_exponential_mle,
    fit_logistic_mle,
    fit_weibull_mle,
    logistic_prob,
    exponential_prob,
    weibull_prob,
)

# -----------------------------
# Paths / config
# -----------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# Expected default path in repo (data not included in the zip bundle)
DEFAULT_DATA_FILE = PROJECT_ROOT / "eval-analysis-public" / "data" / "external" / "all_runs.jsonl"
DATA_FILE = Path(os.environ.get("PETO_DATA_FILE", str(DEFAULT_DATA_FILE)))

# Write to a separate directory by default so you can compare old vs new outputs.
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results_mle"
OUTPUT_DIR = Path(os.environ.get("PETO_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
PLOTS_DIR = OUTPUT_DIR / "model_fits"

SKIP_ALIASES = {"gpt-3.5-turbo-instruct"}
MIN_N = int(os.environ.get("PETO_MIN_N", "50"))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Set PETO_DATA_FILE to point at eval-analysis-public/data/external/all_runs.jsonl"
        )


def main() -> None:
    _require_file(DATA_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(DATA_FILE, lines=True)
    if "alias" not in df.columns:
        raise KeyError("Expected column 'alias' not found in dataset.")

    required_cols = ["human_minutes", "score_binarized"]

    aliases = pd.Series(df["alias"].dropna().unique()).sort_values().tolist()

    rows = []

    for alias in aliases:
        if alias in SKIP_ALIASES:
            continue

        subset = df[df["alias"] == alias].copy()
        if any(c not in subset.columns for c in required_cols):
            continue
        subset = subset[required_cols].dropna()
        subset = subset[subset["human_minutes"] > 0]
        if len(subset) < MIN_N:
            continue

        t_data = subset["human_minutes"].to_numpy(dtype=float)
        y_data = subset["score_binarized"].to_numpy(dtype=float)

        # -----------------------------
        # Fit by Bernoulli MLE (not least squares)
        # -----------------------------
        fit_log = fit_logistic_mle(t_data, y_data)
        fit_exp = fit_exponential_mle(t_data, y_data)
        fit_weib = fit_weibull_mle(t_data, y_data)

        n = len(y_data)

        bic_log = bic_from_ll(fit_log.ll, 2, n) if fit_log.success else np.nan
        bic_exp = bic_from_ll(fit_exp.ll, 1, n) if fit_exp.success else np.nan
        bic_weib = bic_from_ll(fit_weib.ll, 2, n) if fit_weib.success else np.nan

        bics = {"logistic": bic_log, "weibull": bic_weib}
        best_bic = min(bics.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else np.inf)[0]

        b0, b1 = fit_log.params if fit_log.success else (np.nan, np.nan)
        (lam_exp,) = fit_exp.params if fit_exp.success else (np.nan,)
        lam_weib, k_weib = fit_weib.params if fit_weib.success else (np.nan, np.nan)
        is_effectively_exponential = bool(np.isfinite(k_weib) and 0.9 < k_weib < 1.1)

        rows.append(
            {
                "alias": alias,
                "n": n,
                "logistic_b0": b0,
                "logistic_b1": b1,
                "exponential_lambda": lam_exp,
                "weibull_lambda": lam_weib,
                "weibull_k": k_weib,
                "is_effectively_exponential": is_effectively_exponential,
                "ll_logistic": fit_log.ll,
                "ll_exponential": fit_exp.ll,
                "ll_weibull": fit_weib.ll,
                "bic_logistic": bic_log,
                "bic_exponential": bic_exp,
                "bic_weibull": bic_weib,
                "best_bic": best_bic,
            }
        )

        # -----------------------------
        # Plot
        # -----------------------------
        bins = subset.groupby("human_minutes")["score_binarized"].agg(["mean", "count"]).reset_index()
        bins = bins[bins["count"] > 2]

        x_min = float(np.min(t_data))
        x_max = float(np.max(t_data))
        x_range = np.logspace(np.log10(x_min), np.log10(x_max), 200)

        plt.figure(figsize=(10, 6))
        if len(bins) > 0:
            plt.scatter(
                bins["human_minutes"],
                bins["mean"],
                color="black",
                alpha=0.6,
                s=bins["count"] * 2,
                label="Empirical (size=count)",
            )

        if fit_log.success:
            plt.plot(x_range, logistic_prob(x_range, b0, b1), "b--", linewidth=2, label="Logistic (MLE)")
        if fit_exp.success:
            plt.plot(x_range, exponential_prob(x_range, lam_exp), "orange", linewidth=2, label="Exponential (MLE)")
        if fit_weib.success:
            plt.plot(
                x_range,
                weibull_prob(x_range, lam_weib, k_weib),
                "r-",
                linewidth=3,
                label=f"Weibull (MLE) k={k_weib:.2f}",
            )

        plt.xscale("log")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Task Difficulty (Human Minutes) [Log Scale]")
        plt.ylabel("Probability of Success")
        plt.title(f"{alias} | Best BIC (Logistic vs Weibull): {best_bic}")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        safe_alias = "".join(ch if ch.isalnum() else "_" for ch in alias).lower()
        plot_path = PLOTS_DIR / f"{safe_alias}.png"
        plt.savefig(plot_path)
        plt.close()

    summary = pd.DataFrame(rows).sort_values(by="alias")
    summary_path = OUTPUT_DIR / "model_fit_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved summary: {summary_path}")
    print(f"Saved plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
