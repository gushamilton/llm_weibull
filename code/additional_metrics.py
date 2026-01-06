import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "eval-analysis-public" / "data" / "external" / "all_runs.jsonl"
DATA_PATH = Path(os.environ.get("PETO_DATA_FILE", str(DEFAULT_DATA_PATH)))

DEFAULT_WEIBULL_CSV_MLE = PROJECT_ROOT / "results_mle" / "weibull_params_with_bootstrap_ci.csv"
DEFAULT_WEIBULL_CSV_OLD = PROJECT_ROOT / "results" / "weibull_params_with_bootstrap_ci.csv"
WEIBULL_CSV = Path(os.environ.get(
    "PETO_WEIBULL_CSV",
    str(DEFAULT_WEIBULL_CSV_MLE if DEFAULT_WEIBULL_CSV_MLE.exists() else DEFAULT_WEIBULL_CSV_OLD),
))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results_mle" / "additional"
OUTPUT_DIR = Path(os.environ.get("PETO_ADDITIONAL_DIR", str(DEFAULT_OUTPUT_DIR)))

EXCLUDE_ALIASES = {"human"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_model_metrics():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}. Set PETO_DATA_FILE.")
    if not WEIBULL_CSV.exists():
        raise FileNotFoundError(
            f"Weibull CSV not found: {WEIBULL_CSV}. "
            f"Run bootstrap_weibull_params_mle.py (or set PETO_WEIBULL_CSV)."
        )

    df = pd.read_json(DATA_PATH, lines=True)
    df = df[~df["alias"].isin(EXCLUDE_ALIASES)].copy()

    agg = df.groupby("alias").agg(
        accuracy=("score_binarized", "mean"),
        mean_minutes=("human_minutes", "mean"),
        median_minutes=("human_minutes", "median"),
        n=("score_binarized", "size"),
    )

    weib = pd.read_csv(WEIBULL_CSV)
    weib = weib[~weib["alias"].isin(EXCLUDE_ALIASES)].copy()

    out = agg.reset_index().merge(weib[["alias", "lambda", "k"]], on="alias", how="inner")
    out["log_lambda"] = np.log(out["lambda"])
    return out


def plot_lambda_vs_accuracy(metrics):
    plt.figure(figsize=(7, 5))
    plt.scatter(metrics["accuracy"], metrics["lambda"], color="#2ca02c")
    plt.yscale("log")
    plt.xlabel("Accuracy (mean success rate)")
    plt.ylabel("Weibull lambda [log scale]")
    plt.title("Lambda vs Accuracy (AI Only)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "lambda_vs_accuracy.png"
    plt.savefig(path)
    plt.close()
    return str(path)


def corr_heatmap(metrics):
    corr_cols = ["accuracy", "lambda", "k", "mean_minutes", "median_minutes", "n", "log_lambda"]
    corr = metrics[corr_cols].corr()

    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)

    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Correlation Heatmap (Model-Level Metrics)")
    plt.tight_layout()

    path = OUTPUT_DIR / "metrics_correlation_heatmap.png"
    plt.savefig(path)
    plt.close()
    return str(path)


def main():
    ensure_dir(OUTPUT_DIR)
    metrics = load_model_metrics()

    metrics_path = OUTPUT_DIR / "model_level_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    lambda_plot = plot_lambda_vs_accuracy(metrics)
    corr_plot = corr_heatmap(metrics)

    print(str(metrics_path))
    print(lambda_plot)
    print(corr_plot)


if __name__ == "__main__":
    main()
