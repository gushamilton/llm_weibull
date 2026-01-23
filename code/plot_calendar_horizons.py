import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOGISTIC_FITS = PROJECT_ROOT / "eval-analysis-public" / "data" / "wrangled" / "logistic_fits" / "headline.csv"
WEIBULL_CSV = PROJECT_ROOT / "results_mle" / "weibull_params_with_bootstrap_ci.csv"
RELEASE_DATES = PROJECT_ROOT / "eval-analysis-public" / "data" / "external" / "release_dates.yaml"
OUTPUT_DIR = PROJECT_ROOT / "results_mle" / "additional"


def _weibull_time(p: float, lam: float, k: float) -> float:
    return (-math.log(p)) ** (1.0 / k) / lam


def _plot_calendar(df: pd.DataFrame, title: str, out_path: Path) -> None:
    df = df.sort_values("release_date")
    plt.figure(figsize=(9, 5))
    plt.plot(df["release_date"], df["p50"], marker="o", label="p50", color="#1f77b4")
    plt.plot(df["release_date"], df["p80"], marker="o", label="p80", color="#ff7f0e")
    plt.yscale("log")
    plt.xlabel("Model release date")
    plt.ylabel("Task time (minutes)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Logistic p50/p80 from logistic fits
    if not LOGISTIC_FITS.exists():
        raise FileNotFoundError(f"Missing logistic fits: {LOGISTIC_FITS}")
    logistic = pd.read_csv(LOGISTIC_FITS)
    logistic = logistic[logistic["release_date"].notna()].copy()
    logistic["release_date"] = pd.to_datetime(logistic["release_date"], errors="coerce")
    logistic = logistic.dropna(subset=["release_date", "p50", "p80"])
    logistic = logistic[logistic["agent"].str.lower() != "human"].copy()
    _plot_calendar(
        logistic[["release_date", "p50", "p80"]],
        "Logistic: Calendar Time vs p50/p80",
        OUTPUT_DIR / "calendar_p50_p80_logistic.png",
    )

    # Weibull p50/p80 from params + release dates
    if not WEIBULL_CSV.exists():
        raise FileNotFoundError(f"Missing Weibull params: {WEIBULL_CSV}")
    weib = pd.read_csv(WEIBULL_CSV)
    weib = weib[weib["alias"].str.lower() != "human"].copy()

    dates = yaml.safe_load(RELEASE_DATES.read_text())
    date_map = dates.get("date", {})
    weib["release_date"] = weib["alias"].map(date_map)
    weib = weib[weib["release_date"].notna()].copy()
    weib["release_date"] = pd.to_datetime(weib["release_date"], errors="coerce")

    weib["p50"] = weib.apply(lambda r: _weibull_time(0.5, r["lambda"], r["k"]), axis=1)
    weib["p80"] = weib.apply(lambda r: _weibull_time(0.8, r["lambda"], r["k"]), axis=1)

    weib = weib.replace([float("inf"), -float("inf")], pd.NA).dropna(subset=["release_date", "p50", "p80"])

    _plot_calendar(
        weib[["release_date", "p50", "p80"]],
        "Weibull: Calendar Time vs p50/p80",
        OUTPUT_DIR / "calendar_p50_p80_weibull.png",
    )


if __name__ == "__main__":
    main()
