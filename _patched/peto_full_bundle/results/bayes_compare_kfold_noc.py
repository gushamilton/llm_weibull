import os
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import logsumexp

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / 'eval-analysis-public' / 'data' / 'external' / 'all_runs.jsonl'
DATA_PATH = Path(os.environ.get('PETO_DATA_FILE', str(DEFAULT_DATA_PATH)))
RESULTS_ROOT = Path(os.environ.get('PETO_RESULTS_ROOT', str(PROJECT_ROOT / 'results_mle')))
OUTPUT_DIR = RESULTS_ROOT / 'bayes_compare_kfold_noc'

K_FOLDS = int(os.environ.get("K_FOLDS", "5"))
DRAWS = int(os.environ.get("PM_DRAWS", "500"))
TUNE = int(os.environ.get("PM_TUNE", "500"))
CHAINS = int(os.environ.get("PM_CHAINS", "4"))
TARGET_ACCEPT = float(os.environ.get("PM_TARGET_ACCEPT", "0.9"))
SEED = int(os.environ.get("PM_SEED", "42"))


def fit_weibull(train):
    t, y, model_idx, n_models = train
    with pm.Model() as model:
        mu_k = pm.Normal("mu_k", 1.0, 0.5)
        k = pm.Deterministic("k", pm.math.exp(pm.Normal("k_log", mu_k, 0.5, shape=n_models)))

        mu_lam = pm.Normal("mu_lam", np.log(0.1), 1.0)
        lam = pm.Deterministic("lam", pm.math.exp(pm.Normal("lam_log", mu_lam, 1.0, shape=n_models)))

        p = pm.math.exp(-((lam[model_idx] * t) ** k[model_idx]))
        pm.Bernoulli("y", p=p, observed=y)

        idata = pm.sample(
            DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=SEED,
            return_inferencedata=True,
            progressbar=False,
        )
    return idata


def fit_exponential(train):
    t, y, model_idx, n_models = train
    with pm.Model() as model:
        mu_lam = pm.Normal("mu_lam", np.log(0.1), 1.0)
        lam = pm.Deterministic("lam", pm.math.exp(pm.Normal("lam_log", mu_lam, 1.0, shape=n_models)))

        p = pm.math.exp(-(lam[model_idx] * t))
        pm.Bernoulli("y", p=p, observed=y)

        idata = pm.sample(
            DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=SEED,
            return_inferencedata=True,
            progressbar=False,
        )
    return idata


def fit_logistic(train):
    t, y, model_idx, n_models = train
    with pm.Model() as model:
        mu_b0 = pm.Normal("mu_b0", 2.0, 1.0)
        b0 = pm.Normal("b0", mu_b0, 1.0, shape=n_models)

        mu_b1 = pm.Normal("mu_b1", -0.5, 0.5)
        b1 = pm.Normal("b1", mu_b1, 0.5, shape=n_models)

        term = -(b0[model_idx] + b1[model_idx] * np.log(t))
        p = 1 / (1 + pm.math.exp(term))
        pm.Bernoulli("y", p=p, observed=y)

        idata = pm.sample(
            DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=SEED,
            return_inferencedata=True,
            progressbar=False,
        )
    return idata


def elpd_weibull(idata, test):
    t, y, model_idx = test
    k = idata.posterior["k"].values.reshape(-1, idata.posterior["k"].shape[-1])
    lam = idata.posterior["lam"].values.reshape(-1, idata.posterior["lam"].shape[-1])

    p = np.exp(-((lam[:, model_idx] * t) ** k[:, model_idx]))
    p = np.clip(p, 1e-12, 1 - 1e-12)

    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    lpd = logsumexp(ll, axis=0) - np.log(ll.shape[0])
    return lpd


def elpd_exponential(idata, test):
    t, y, model_idx = test
    lam = idata.posterior["lam"].values.reshape(-1, idata.posterior["lam"].shape[-1])

    p = np.exp(-(lam[:, model_idx] * t))
    p = np.clip(p, 1e-12, 1 - 1e-12)

    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    lpd = logsumexp(ll, axis=0) - np.log(ll.shape[0])
    return lpd


def elpd_logistic(idata, test):
    t, y, model_idx = test
    b0 = idata.posterior["b0"].values.reshape(-1, idata.posterior["b0"].shape[-1])
    b1 = idata.posterior["b1"].values.reshape(-1, idata.posterior["b1"].shape[-1])

    term = -(b0[:, model_idx] + b1[:, model_idx] * np.log(t))
    p = 1 / (1 + np.exp(term))
    p = np.clip(p, 1e-12, 1 - 1e-12)

    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    lpd = logsumexp(ll, axis=0) - np.log(ll.shape[0])
    return lpd


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_json(DATA_PATH, lines=True)
    df = df[df["alias"] != "human"].copy()
    df["t"] = df["human_minutes"] / 60.0
    df = df[df["t"] > 0].copy()

    models = sorted(df["alias"].unique().tolist())
    model_map = {name: idx for idx, name in enumerate(models)}
    df["model_idx"] = df["alias"].map(model_map)

    n = len(df)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    folds = np.array_split(idx, K_FOLDS)

    fold_rows = []
    alias_rows = []

    for i, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train = (
            train_df["t"].values,
            train_df["score_binarized"].values,
            train_df["model_idx"].values,
            len(models),
        )
        test = (
            test_df["t"].values,
            test_df["score_binarized"].values,
            test_df["model_idx"].values,
        )

        idata_weib = fit_weibull(train)
        lpd_weib = elpd_weibull(idata_weib, test)

        idata_exp = fit_exponential(train)
        lpd_exp = elpd_exponential(idata_exp, test)

        idata_log = fit_logistic(train)
        lpd_log = elpd_logistic(idata_log, test)

        test_alias = test_df["alias"].values
        for alias in np.unique(test_alias):
            mask = test_alias == alias
            alias_rows.append(
                {"fold": i, "alias": alias, "model": "weibull", "elpd": float(lpd_weib[mask].sum())}
            )
            alias_rows.append(
                {"fold": i, "alias": alias, "model": "exponential", "elpd": float(lpd_exp[mask].sum())}
            )
            alias_rows.append(
                {"fold": i, "alias": alias, "model": "logistic", "elpd": float(lpd_log[mask].sum())}
            )

        fold_rows.append(
            {
                "fold": i,
                "elpd_weibull": float(lpd_weib.sum()),
                "elpd_exponential": float(lpd_exp.sum()),
                "elpd_logistic": float(lpd_log.sum()),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_path = os.path.join(OUTPUT_DIR, "kfold_elpd_by_fold.csv")
    fold_df.to_csv(fold_path, index=False)

    summary = fold_df[["elpd_weibull", "elpd_exponential", "elpd_logistic"]].agg(["mean", "std"]).T
    summary["se"] = summary["std"] / np.sqrt(K_FOLDS)
    summary_path = os.path.join(OUTPUT_DIR, "kfold_elpd_summary.csv")
    summary.to_csv(summary_path)

    alias_df = pd.DataFrame(alias_rows)
    alias_path = os.path.join(OUTPUT_DIR, "kfold_elpd_by_alias.csv")
    alias_df.to_csv(alias_path, index=False)

    alias_summary = (
        alias_df.groupby(["alias", "model"])["elpd"]
        .agg(["mean", "std"])
        .reset_index()
    )
    alias_summary["se"] = alias_summary["std"] / np.sqrt(K_FOLDS)
    alias_summary_path = os.path.join(OUTPUT_DIR, "kfold_elpd_by_alias_summary.csv")
    alias_summary.to_csv(alias_summary_path, index=False)

    print(fold_path)
    print(summary_path)
    print(alias_path)
    print(alias_summary_path)


if __name__ == "__main__":
    main()
