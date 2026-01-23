import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

WEIBULL_CSV = PROJECT_ROOT / "results_mle" / "weibull_params_with_bootstrap_ci.csv"
OUTPUT_DIR = PROJECT_ROOT / "results_mle" / "additional"
OUTPUT_PATH = OUTPUT_DIR / "human_ai_k_forest.png"


def load_weibull_rows(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    if not WEIBULL_CSV.exists():
        raise FileNotFoundError(f"Missing Weibull CSV: {WEIBULL_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_weibull_rows(WEIBULL_CSV)
    entries = []
    for row in rows:
        alias = row["alias"]
        entries.append(
            {
                "alias": alias,
                "k": float(row["k"]),
                "k_low": float(row["k_ci_low"]),
                "k_high": float(row["k_ci_high"]),
                "is_human": alias.strip().lower() == "human",
            }
        )

    entries.sort(key=lambda r: r["k"], reverse=True)

    labels = [r["alias"] for r in entries]
    ks = np.array([r["k"] for r in entries])
    k_err_low = ks - np.array([r["k_low"] for r in entries])
    k_err_high = np.array([r["k_high"] for r in entries]) - ks
    colors = ["#d62728" if r["is_human"] else "#1f77b4" for r in entries]

    y_pos = np.arange(len(entries))

    plt.figure(figsize=(8, max(6, 0.35 * len(entries) + 2)))
    for idx in range(len(entries)):
        plt.errorbar(
            ks[idx],
            y_pos[idx],
            xerr=[[k_err_low[idx]], [k_err_high[idx]]],
            fmt="o",
            ecolor=colors[idx],
            color=colors[idx],
            elinewidth=1.5,
            capsize=3,
            linestyle="none",
            zorder=3,
        )

    plt.yticks(y_pos, labels)
    plt.xlabel("Weibull Shape Parameter (K)")
    plt.ylabel("Model Alias")
    plt.title("Weibull K by Model (Human vs AI)")
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="Constant Hazard (K=1)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH)
    plt.close()

    print(f"Saved plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
