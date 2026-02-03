import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add code directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))
try:
    from mle_utils import logistic_prob, weibull_prob
except ImportError:
    def logistic_prob(t, b0, b1):
        logit = b0 + b1 * np.log(t)
        return 1.0 / (1.0 + np.exp(-logit))

    def weibull_prob(t, lam, k):
        return np.exp(-((lam * t) ** k))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT
SUMMARY_CSV = PROJECT_ROOT / "results_mle" / "model_fit_summary.csv"
DATA_FILE = Path(os.environ.get("PETO_DATA_FILE", str(PROJECT_ROOT / "eval-analysis-public" / "data" / "external" / "all_runs.jsonl")))
WEIBULL_CI_CSV = PROJECT_ROOT / "results_mle" / "weibull_params_with_bootstrap_ci.csv"

# -----------------------------------------------------------------------------
# GLOBAL STYLE SETTINGS
# -----------------------------------------------------------------------------
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def get_params(alias):
    if not SUMMARY_CSV.exists():
        print(f"Summary CSV not found: {SUMMARY_CSV}")
        return None
    df = pd.read_csv(SUMMARY_CSV)
    row = df[df["alias"] == alias]
    if len(row) == 0:
        return None
    row = row.iloc[0]
    return {
        "b0": row["logistic_b0"],
        "b1": row["logistic_b1"],
        "lam": row["weibull_lambda"],
        "k": row["weibull_k"],
        "n": row["n"]
    }

def generate_figure_1():
    print("Generating Figure 1 (Weibull k by model)...")
    if not WEIBULL_CI_CSV.exists():
        print("Weibull CI CSV not found.")
        return
        
    df = pd.read_csv(WEIBULL_CI_CSV)
    df = df.sort_values(by="k", ascending=True)
    
    plt.figure(figsize=(8, max(6, 0.4 * len(df))))
    
    y = np.arange(len(df))
    plt.errorbar(
        df["k"],
        y,
        xerr=[
            df["k"] - df["k_ci_low"],
            df["k_ci_high"] - df["k"]
        ],
        fmt="o",
        markersize=5,
        capsize=3,
        color="#1f77b4"
    )
    
    # Add k=1 dashed line
    plt.axvline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Constant Failure Rate (k=1)")
    
    plt.yticks(y, df["alias"])
    plt.xlabel("Weibull Shape parameter $k$")
    plt.title("Weibull Shape ($k$) by Model (with 95% CI)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure1.png", dpi=300)
    plt.close()

def generate_figure_2():
    print("Generating Figure 2 (Weibull lambda by model)...")
    if not WEIBULL_CI_CSV.exists():
        print("Weibull CI CSV not found.")
        return
        
    df = pd.read_csv(WEIBULL_CI_CSV)
    df = df.sort_values(by="lambda", ascending=True)
    
    plt.figure(figsize=(8, max(6, 0.4 * len(df))))
    
    y = np.arange(len(df))
    plt.errorbar(
        df["lambda"],
        y,
        xerr=[
            df["lambda"] - df["lambda_ci_low"],
            df["lambda_ci_high"] - df["lambda"]
        ],
        fmt="o",
        markersize=5,
        capsize=3,
        color="#2ca02c"
    )
    
    plt.xscale("log")  # FIX: Log scale for lambda
    plt.yticks(y, df["alias"])
    plt.xlabel("Weibull Scale parameter $\lambda$ (Log Scale)")
    plt.title("Weibull Scale ($\lambda$) by Model (with 95% CI)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure2.png", dpi=300)
    plt.close()

def generate_figure_3():
    print("Generating Figure 3 (Side-by-side metrics)...")
    metrics_path = PROJECT_ROOT / "results_mle" / "additional" / "model_level_metrics.csv"
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found.")
        return

    df = pd.read_csv(metrics_path)
    df = df[df["alias"] != "human"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Lambda vs Accuracy
    ax1.scatter(df["accuracy"], df["lambda"], color="#2ca02c", s=100, alpha=0.9, edgecolor='k', linewidth=0.8)
    ax1.set_yscale("log")
    
    # Trend line for Lambda
    if len(df) > 2:
        valid_mask = (df["lambda"] > 0)
        sdf = df[valid_mask]
        z = np.polyfit(sdf["accuracy"], np.log(sdf["lambda"]), 1)
        p = np.poly1d(z)
        x_line = np.linspace(sdf["accuracy"].min(), sdf["accuracy"].max(), 100)
        ax1.plot(x_line, np.exp(p(x_line)), "k--", alpha=0.5, linewidth=1.5)

    ax1.set_xlabel("Accuracy (Mean Success Rate)", fontsize=12, fontweight='medium')
    ax1.set_ylabel("Weibull Lambda (Scale)", fontsize=12, fontweight='medium')
    ax1.set_title("Scale ($\lambda$) vs Capability", fontsize=14, fontweight='medium')
    
    # Plot 2: K vs Accuracy
    ax2.scatter(df["accuracy"], df["k"], color="#1f77b4", s=100, alpha=0.9, edgecolor='k', linewidth=0.8)
    
    # Trend line for K
    if len(df) > 2:
        z = np.polyfit(df["accuracy"], df["k"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df["accuracy"].min(), df["accuracy"].max(), 100)
        ax2.plot(x_line, p(x_line), "k--", alpha=0.5, linewidth=1.5)

    ax2.set_xlabel("Accuracy (Mean Success Rate)", fontsize=12, fontweight='medium')
    ax2.set_ylabel("Weibull k (Shape)", fontsize=12, fontweight='medium')
    ax2.set_title("Shape ($k$) vs Capability", fontsize=14, fontweight='medium')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure3.png", dpi=300)
    plt.close()

def plot_model_fit(alias, output_filename):
    print(f"Generating {output_filename} for {alias}...")
    
    params = get_params(alias)
    if not params:
        print(f"No parameters found for {alias}")
        return

    # Try to load empirical data for points
    bins = None
    if DATA_FILE.exists():
        try:
            df = pd.read_json(str(DATA_FILE), lines=True)
            subset = df[df["alias"] == alias].copy()
            if len(subset) > 0:
                subset = subset[["human_minutes", "score_binarized"]].dropna()
                subset = subset[subset["human_minutes"] > 0]
                bins = subset.groupby("human_minutes")["score_binarized"].agg(["mean", "count"]).reset_index()
                bins = bins[bins["count"] > 2]
                x_min = float(np.min(subset["human_minutes"]))
                x_max = float(np.max(subset["human_minutes"]))
            else:
                x_min, x_max = 1e-1, 1e4
        except Exception:
            x_min, x_max = 1e-1, 1e4
    else:
        print(f"Warning: Data file {DATA_FILE} not found. Plotting curves only.")
        x_min, x_max = 1e-1, 1e4

    plt.figure(figsize=(10, 6))
    
    # Plot Empirical Points if available
    if bins is not None and len(bins) > 0:
        plt.scatter(
            bins["human_minutes"],
            bins["mean"],
            color="black",
            alpha=0.6,
            s=bins["count"] * 3,
            label="Data",  # FIX: Changed label
            zorder=3
        )

    # Plot Curves
    x_range = np.logspace(np.log10(x_min), np.log10(x_max), 500)
    
    # Logistic
    plt.plot(x_range, logistic_prob(x_range, params['b0'], params['b1']), 
             "b--", linewidth=2.5, label="Logistic (MLE)", zorder=2)

    # Weibull
    plt.plot(x_range, weibull_prob(x_range, params['lam'], params['k']), 
             "r-", linewidth=3, label=f"Weibull (MLE) k={params['k']:.2f}", zorder=2)

    plt.xscale("log")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Task Difficulty (Human Minutes) [Log Scale]", fontsize=12)
    plt.ylabel("Probability of Success", fontsize=12)
    plt.title(f"{alias}", fontsize=14)  # FIX: Simplified title
    plt.legend(fontsize=10, loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_filename, dpi=300)
    plt.close()

def generate_figure_6():
    print("Generating Figure 6...")
    csv_path = PROJECT_ROOT / "results_mle" / "additional" / "avg_log_diff_logistic_vs_weibull_by_p_gpt_4o.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    mask = (df["p"] >= 0.2) & (df["p"] <= 0.99999)
    df = df[mask].copy()

    p_vals = df["p"]
    ratios = np.exp(df["mean_log_ratio"])

    plt.figure(figsize=(9, 6))
    plt.plot(p_vals, ratios, color="#1f77b4", linewidth=2.5)
    
    plt.xscale("logit")
    plt.yscale("log")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1.2)
    
    plt.xlabel("Success Probability ($p$)", fontsize=12)
    plt.ylabel("Horizon Ratio (Logistic / Weibull)", fontsize=12)
    plt.title("GPT-4o: Horizon Ratio by Success Level", fontsize=14)
    
    ticks = [0.2, 0.5, 0.8, 0.9, 0.99, 0.999, 0.99999]
    tick_labels = ["20%", "50%", "80%", "90%", "99%", "99.9%", "99.999%"]
    plt.xticks(ticks, tick_labels)
    plt.yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], 
               ["0.1x", "0.2x", "0.5x", "1x", "2x", "5x", "10x", "20x", "50x", "100x"])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure6.png", dpi=300)
    plt.close()

def main():
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    plot_model_fit("GPT-4o", "Figure4.png")
    plot_model_fit("human", "Figure5.png")
    generate_figure_6()

if __name__ == "__main__":
    main()
