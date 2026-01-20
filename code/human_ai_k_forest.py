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
    ai_rows = []
    human_row = None

    for row in rows:
        alias = row["alias"]
        k = float(row["k"])
        k_low = float(row["k_ci_low"])
        k_high = float(row["k_ci_high"])
        entry = {"alias": alias, "k": k, "k_low": k_low, "k_high": k_high}
        if alias.strip().lower() == "human":
            human_row = entry
        else:
            ai_rows.append(entry)

    ai_rows.sort(key=lambda r: r["k"])

    labels = [r["alias"] for r in ai_rows]
    ks = np.array([r["k"] for r in ai_rows])
    k_err_low = ks - np.array([r["k_low"] for r in ai_rows])
    k_err_high = np.array([r["k_high"] for r in ai_rows]) - ks

    y_pos = np.arange(len(ai_rows))

    plt.figure(figsize=(8, max(6, 0.35 * len(ai_rows) + 2)))
    plt.errorbar(
        ks,
        y_pos,
        xerr=[k_err_low, k_err_high],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.5,
        capsize=3,
        label="AI models",
    )

    if human_row is not None:
        human_y = len(ai_rows) + 0.8
        plt.errorbar(
            [human_row["k"]],
            [human_y],
            xerr=[[human_row["k"] - human_row["k_low"]], [human_row["k_high"] - human_row["k"]]],
            fmt="o",
            color="#d62728",
            ecolor="#d62728",
            elinewidth=2,
            capsize=4,
            label="Human",
        )
        plt.annotate(
            "Humans: Rapidly Decreasing Hazard (K ≈ 0.37)",
            xy=(human_row["k"], human_y),
            xytext=(human_row["k"] + 0.2, human_y + 0.4),
            arrowprops=dict(arrowstyle="->", color="#d62728"),
            color="#d62728",
            fontsize=10,
        )

    plt.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="K = 1 (random failure)")
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1.2, label="K = 0.5 (strongly decreasing)")

    plt.yticks(y_pos, labels)
    plt.xlabel("Weibull Shape Parameter (K)")
    plt.ylabel("Model Alias")
    plt.title("Weibull K by Model (Human vs AI)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH)
    plt.close()

    print(f"Saved plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
