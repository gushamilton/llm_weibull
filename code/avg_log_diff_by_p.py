import math
import os
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

SUMMARY_CSV = PROJECT_ROOT / "results_mle" / "model_fit_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "results_mle" / "additional"

P_MIN = float(1e-4)
P_MAX = float(0.99999)
P_STEPS = int(240)

MODEL_ALIAS = os.environ.get("PETO_MODEL_ALIAS")
P_POINTS = [0.5, 0.8, 0.9, 0.99, 0.999, 0.99999]


def logistic_time(p: float, b0: float, b1: float) -> float:
    if not math.isfinite(b0) or not math.isfinite(b1) or b1 == 0:
        return float("nan")
    logit = math.log(p / (1.0 - p))
    return math.exp((logit - b0) / b1)


def weibull_time(p: float, lam: float, k: float) -> float:
    if not math.isfinite(lam) or not math.isfinite(k) or lam <= 0 or k <= 0:
        return float("nan")
    return (-math.log(p)) ** (1.0 / k) / lam


def load_summary(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _plot_ratio_curve(p_vals, ratios, out_path, title):
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(p_vals, ratios, color="#1f77b4", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Success probability p")
    plt.ylabel("Horizon ratio (Logistic / Weibull, x-times)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_ratio_curve_smoothed(p_vals, ratios, out_path, title):
    mask = (p_vals >= 0.2) & (p_vals <= 0.99999)
    p_vals = p_vals[mask]
    ratios = ratios[mask]

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(p_vals, ratios, color="#1f77b4", linewidth=2)
    plt.xscale("logit")
    plt.yscale("log")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Success probability")
    plt.ylabel("Horizon ratio (Logistic / Weibull, x-times)")
    plt.title(title)
    plt.xticks(
        [0.2, 0.5, 0.8, 0.9, 0.99, 0.999, 0.99999],
        ["20%", "50%", "80%", "90%", "99%", "99.9%", "99.999%"],
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_ratio_curve_with_uncertainty(p_vals, ratios, low, high, out_path, title):
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(p_vals, ratios, color="#1f77b4", linewidth=2, label="Mean")
    plt.fill_between(p_vals, low, high, color="#1f77b4", alpha=0.2, label="P16–P84")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Parity (x=1)")
    plt.xscale("log")
    plt.xlabel("Success probability p")
    plt.ylabel("Horizon ratio (Logistic / Weibull, x-times)")
    plt.title(title)
    plt.xticks(
        [0.5, 0.8, 0.9, 0.99, 0.999, 0.99999],
        ["50%", "80%", "90%", "99%", "99.9%", "99.999%"],
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_ratio_points(p_vals, ratios, out_path, title):
    labels = [
        "50%",
        "80%",
        "90%",
        "99%",
        "99.9%",
        "99.999%",
    ]
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(range(len(p_vals)), ratios, color="#1f77b4", linewidth=2, marker="o")
    plt.yscale("log")
    plt.xticks(range(len(p_vals)), labels)
    plt.xlabel("Success probability")
    plt.ylabel("Horizon ratio (Logistic / Weibull, x-times)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary: {SUMMARY_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_summary(SUMMARY_CSV)

    params = []
    for row in rows:
        try:
            params.append(
                {
                    "alias": row["alias"],
                    "b0": float(row["logistic_b0"]),
                    "b1": float(row["logistic_b1"]),
                    "lam": float(row["weibull_lambda"]),
                    "k": float(row["weibull_k"]),
                }
            )
        except Exception:
            continue

    ps = np.logspace(np.log10(P_MIN), np.log10(P_MAX), P_STEPS)

    mean_rows = []
    ci_rows = []

    for p in ps:
        diffs = []
        for entry in params:
            if MODEL_ALIAS and entry["alias"] != MODEL_ALIAS:
                continue
            t_log = logistic_time(p, entry["b0"], entry["b1"])
            t_weib = weibull_time(p, entry["lam"], entry["k"])
            if not math.isfinite(t_log) or not math.isfinite(t_weib) or t_log <= 0 or t_weib <= 0:
                continue
            diffs.append(math.log(t_log / t_weib))

        if not diffs:
            continue

        mean_val = float(np.mean(diffs))
        mean_rows.append({"p": float(p), "mean_log_ratio": mean_val})

        p16, p84 = np.percentile(diffs, [16, 84])
        ci_rows.append(
            {
                "p": float(p),
                "mean_log_ratio": mean_val,
                "p16_log_ratio": float(p16),
                "p84_log_ratio": float(p84),
            }
        )

    # Save CSVs
    suffix = "" if not MODEL_ALIAS else f"_{MODEL_ALIAS.lower().replace(' ', '_').replace('.', '').replace('-', '_')}"
    mean_path = OUTPUT_DIR / f"avg_log_diff_logistic_vs_weibull_by_p{suffix}.csv"
    ci_path = OUTPUT_DIR / f"avg_log_diff_logistic_vs_weibull_by_p_with_uncertainty{suffix}.csv"

    with mean_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["p", "mean_log_ratio"])
        writer.writeheader()
        writer.writerows(mean_rows)

    with ci_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["p", "mean_log_ratio", "p16_log_ratio", "p84_log_ratio"])
        writer.writeheader()
        writer.writerows(ci_rows)

    # Plot mean curve
    p_vals = np.array([r["p"] for r in mean_rows])
    mean_vals = np.array([r["mean_log_ratio"] for r in mean_rows])

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(p_vals, mean_vals, color="#1f77b4", linewidth=2)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Success probability p")
    plt.ylabel("Mean ln(Logistic / Weibull)")
    plt.title("Average log difference in horizon vs p")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"avg_log_diff_logistic_vs_weibull_by_p{suffix}.png")
    plt.close()

    # Plot with uncertainty band
    p_vals = np.array([r["p"] for r in ci_rows])
    mean_vals = np.array([r["mean_log_ratio"] for r in ci_rows])
    ci_low = np.array([r["p16_log_ratio"] for r in ci_rows])
    ci_high = np.array([r["p84_log_ratio"] for r in ci_rows])

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(p_vals, mean_vals, color="#1f77b4", linewidth=2, label="Mean")
    plt.fill_between(p_vals, ci_low, ci_high, color="#1f77b4", alpha=0.2, label="P16–P84")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlim(1e-1, 1.0)
    plt.xlabel("Success probability p")
    plt.ylabel("Mean ln(Logistic / Weibull)")
    plt.title("Average log difference in horizon vs p (with uncertainty)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"avg_log_diff_logistic_vs_weibull_by_p_with_uncertainty{suffix}.png")
    plt.close()

    if MODEL_ALIAS:
        ratios = np.exp(mean_vals)
        ratio_path = OUTPUT_DIR / f"logistic_weibull_horizon_ratio_by_p{suffix}.png"
        _plot_ratio_curve(
            p_vals,
            ratios,
            ratio_path,
            f"{MODEL_ALIAS}: Logistic/Weibull Horizon Ratio vs p",
        )

        smooth_path = OUTPUT_DIR / f"logistic_weibull_horizon_ratio_by_p_smoothed{suffix}.png"
        _plot_ratio_curve_smoothed(
            p_vals,
            ratios,
            smooth_path,
            f"{MODEL_ALIAS}: Horizon Ratio by Success Level (Smoothed)",
        )

        point_rows = []
        point_ratios = []
        for p in P_POINTS:
            diffs = []
            for entry in params:
                if entry["alias"] != MODEL_ALIAS:
                    continue
                t_log = logistic_time(p, entry["b0"], entry["b1"])
                t_weib = weibull_time(p, entry["lam"], entry["k"])
                if not math.isfinite(t_log) or not math.isfinite(t_weib) or t_log <= 0 or t_weib <= 0:
                    continue
                diffs.append(math.log(t_log / t_weib))
            if diffs:
                mean_ln = float(np.mean(diffs))
                ratio = float(np.exp(mean_ln))
                point_rows.append({"p": p, "horizon_ratio": ratio})
                point_ratios.append(ratio)

        if point_rows:
            point_csv = OUTPUT_DIR / f"logistic_weibull_horizon_ratio_points{suffix}.csv"
            with point_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["p", "horizon_ratio"])
                writer.writeheader()
                writer.writerows(point_rows)

            point_plot = OUTPUT_DIR / f"logistic_weibull_horizon_ratio_points{suffix}.png"
            _plot_ratio_points(
                P_POINTS,
                point_ratios,
                point_plot,
                f"{MODEL_ALIAS}: Horizon Ratio by Success Level",
            )
    else:
        ratios = np.exp(mean_vals)
        low = np.exp(ci_low)
        high = np.exp(ci_high)
        ratio_path = OUTPUT_DIR / "logistic_weibull_horizon_ratio_by_p_mean.png"
        _plot_ratio_curve_with_uncertainty(
            p_vals,
            ratios,
            low,
            high,
            ratio_path,
            "Mean Horizon Ratio by Success Level",
        )


if __name__ == "__main__":
    main()
