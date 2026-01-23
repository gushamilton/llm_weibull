import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOGISTIC_FITS = PROJECT_ROOT / "eval-analysis-public" / "data" / "wrangled" / "logistic_fits" / "headline.csv"
WEIBULL_CSV = PROJECT_ROOT / "results_mle" / "weibull_params_with_bootstrap_ci.csv"
RELEASE_DATES = PROJECT_ROOT / "eval-analysis-public" / "data" / "external" / "release_dates.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results_mle" / "additional"

P_VALUES = {
    "p50": 0.5,
    "p80": 0.8,
    "p999": 0.999,
    "p001": 0.01,
}


def _weibull_time(p: float, lam: float, k: float) -> float:
    return (-math.log(p)) ** (1.0 / k) / lam


def _logistic_time(p: float, coef: float, intercept: float) -> float:
    logit = math.log(p / (1.0 - p))
    x = (logit - intercept) / coef  # x is log2(time)
    return 2**x


def _fit_line_with_ci(x, y, x_grid):
    n = len(x)
    if n < 3:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    sxx = np.sum((x - x_mean) ** 2)
    if sxx == 0:
        return None

    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    y_hat = intercept + slope * x
    resid = y - y_hat
    s = math.sqrt(np.sum(resid**2) / (n - 2))

    se = s * np.sqrt(1 / n + (x_grid - x_mean) ** 2 / sxx)
    z = 1.96

    y_grid = intercept + slope * x_grid
    lower = y_grid - z * se
    upper = y_grid + z * se
    return y_grid, lower, upper


def _plot_panel(df_log, df_weib, p_label, out_path):
    plt.figure(figsize=(9, 5))

    # Plot points
    plt.scatter(
        df_log["release_date"],
        df_log[p_label],
        color="#1f77b4",
        label="Logistic",
        alpha=0.9,
    )
    plt.scatter(
        df_weib["release_date"],
        df_weib[p_label],
        color="#d62728",
        label="Weibull",
        alpha=0.9,
    )

    # Fit lines in log10(time) space
    x_log = df_log["release_ord"].to_numpy()
    y_log = np.log10(df_log[p_label].to_numpy())
    x_weib = df_weib["release_ord"].to_numpy()
    y_weib = np.log10(df_weib[p_label].to_numpy())

    x_grid = np.linspace(
        min(df_log["release_ord"].min(), df_weib["release_ord"].min()),
        max(df_log["release_ord"].max(), df_weib["release_ord"].max()),
        200,
    )
    x_dates = [pd.Timestamp.fromordinal(int(x)) for x in x_grid]

    fit_log = _fit_line_with_ci(x_log, y_log, x_grid)
    if fit_log:
        y_grid, low, high = fit_log
        plt.plot(x_dates, 10**y_grid, color="#1f77b4", linewidth=2)
        plt.fill_between(x_dates, 10**low, 10**high, color="#1f77b4", alpha=0.15)

    fit_weib = _fit_line_with_ci(x_weib, y_weib, x_grid)
    if fit_weib:
        y_grid, low, high = fit_weib
        plt.plot(x_dates, 10**y_grid, color="#d62728", linewidth=2)
        plt.fill_between(x_dates, 10**low, 10**high, color="#d62728", alpha=0.15)

    plt.yscale("log")
    plt.xlabel("Model release date")
    plt.ylabel("Task time (minutes)")
    plt.title(f"Calendar Time vs {p_label.upper()} (Logistic vs Weibull)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LOGISTIC_FITS.exists():
        raise FileNotFoundError(f"Missing logistic fits: {LOGISTIC_FITS}")
    if not WEIBULL_CSV.exists():
        raise FileNotFoundError(f"Missing Weibull params: {WEIBULL_CSV}")
    if not RELEASE_DATES.exists():
        raise FileNotFoundError(f"Missing release dates: {RELEASE_DATES}")

    logistic = pd.read_csv(LOGISTIC_FITS)
    logistic = logistic[logistic["release_date"].notna()].copy()
    logistic["release_date"] = pd.to_datetime(logistic["release_date"], errors="coerce")
    logistic = logistic.dropna(subset=["release_date", "coefficient", "intercept"])
    logistic = logistic[logistic["agent"].str.lower() != "human"].copy()

    weib = pd.read_csv(WEIBULL_CSV)
    weib = weib[weib["alias"].str.lower() != "human"].copy()

    dates = yaml.safe_load(RELEASE_DATES.read_text())
    date_map = dates.get("date", {})
    weib["release_date"] = weib["alias"].map(date_map)
    weib = weib[weib["release_date"].notna()].copy()
    weib["release_date"] = pd.to_datetime(weib["release_date"], errors="coerce")

    for label, p in P_VALUES.items():
        logistic[label] = logistic.apply(
            lambda r: _logistic_time(p, r["coefficient"], r["intercept"]), axis=1
        )
        weib[label] = weib.apply(
            lambda r: _weibull_time(p, r["lambda"], r["k"]), axis=1
        )

        df_log = logistic.dropna(subset=[label, "release_date"]).copy()
        df_weib = weib.dropna(subset=[label, "release_date"]).copy()

        df_log = df_log[df_log[label] > 0]
        df_weib = df_weib[df_weib[label] > 0]

        df_log["release_ord"] = df_log["release_date"].map(pd.Timestamp.toordinal)
        df_weib["release_ord"] = df_weib["release_date"].map(pd.Timestamp.toordinal)

        out_path = OUTPUT_DIR / f"calendar_{label}_logistic_weibull.png"
        _plot_panel(df_log, df_weib, label, out_path)


if __name__ == "__main__":
    main()
