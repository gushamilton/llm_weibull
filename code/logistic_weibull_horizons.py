import math
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

SUMMARY_CSV = PROJECT_ROOT / "results_mle" / "model_fit_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "results_mle" / "additional"

P_VALUES = [0.5, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.01, 0.0001]
P_LABELS = {
    0.5: "p50",
    0.8: "p80",
    0.9: "p90",
    0.99: "p99",
    0.999: "p999",
    0.9999: "p9999",
    0.01: "p1",
    0.0001: "p0001",
}


def logistic_time(p: float, b0: float, b1: float) -> float:
    if not math.isfinite(b0) or not math.isfinite(b1) or b1 == 0:
        return float("nan")
    logit = math.log(p / (1.0 - p))
    return math.exp((logit - b0) / b1)


def weibull_time(p: float, lam: float, k: float) -> float:
    if not math.isfinite(lam) or not math.isfinite(k) or lam <= 0 or k <= 0:
        return float("nan")
    return (-math.log(p)) ** (1.0 / k) / lam


def load_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scatter(p: float, rows):
    labels = []
    logi = []
    weib = []
    colors = []
    sizes = []
    for row in rows:
        if row["alias"].strip().lower() == "human":
            continue
        if math.isfinite(row["logistic"]) and math.isfinite(row["weibull"]):
            labels.append(row["alias"])
            logi.append(row["logistic"])
            weib.append(row["weibull"])
            if row["alias"].strip() == "GPT-4o":
                colors.append("#d62728")
                sizes.append(140)
            else:
                colors.append("#1f77b4")
                sizes.append(60)

    if not logi:
        return

    logi = np.array(logi)
    weib = np.array(weib)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(logi, weib, c=colors, s=sizes, alpha=0.85)
    min_val = min(np.min(logi), np.min(weib))
    max_val = max(np.max(logi), np.max(weib))
    ax.plot([min_val, max_val], [min_val, max_val], color="gray", linestyle="--", linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Logistic time horizon (minutes)")
    ax.set_ylabel("Weibull time horizon (minutes)")
    ax.set_title(f"p{p:g} Logistic vs Weibull")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    suffix = P_LABELS[p]
    plot_path = OUTPUT_DIR / f"{suffix}_logistic_vs_weibull.png"
    fig.savefig(plot_path)
    plt.close(fig)


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary: {SUMMARY_CSV}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_rows(SUMMARY_CSV)

    for p in P_VALUES:
        out_rows = []
        for row in summary:
            b0 = float(row["logistic_b0"])
            b1 = float(row["logistic_b1"])
            lam = float(row["weibull_lambda"])
            k = float(row["weibull_k"])
            out_rows.append(
                {
                    "alias": row["alias"],
                    "logistic": logistic_time(p, b0, b1),
                    "weibull": weibull_time(p, lam, k),
                }
            )

        suffix = P_LABELS[p]
        csv_path = OUTPUT_DIR / f"{suffix}_logistic_weibull.csv"
        write_csv(csv_path, out_rows, ["alias", "logistic", "weibull"])
        plot_scatter(p, out_rows)

    # Combined p50/p80 CSV for convenience
    combined = []
    for row in summary:
        b0 = float(row["logistic_b0"])
        b1 = float(row["logistic_b1"])
        lam = float(row["weibull_lambda"])
        k = float(row["weibull_k"])
        combined.append(
            {
                "alias": row["alias"],
                "p50_logistic": logistic_time(0.5, b0, b1),
                "p50_weibull": weibull_time(0.5, lam, k),
                "p80_logistic": logistic_time(0.8, b0, b1),
                "p80_weibull": weibull_time(0.8, lam, k),
            }
        )
    write_csv(OUTPUT_DIR / "p50_p80_logistic_weibull.csv", combined, list(combined[0].keys()))


if __name__ == "__main__":
    main()
