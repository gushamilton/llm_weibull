"""Utility functions for fitting pass-probability curves via Bernoulli MLE.

This repo models P(pass | task time) using parametric curves (logistic-in-log(t),
exponential, Weibull). Earlier scripts used scipy.optimize.curve_fit, which
minimizes squared error against y in {0,1}. That's not Bernoulli MLE.

These helpers fit by maximizing the Bernoulli likelihood so that:
- log-likelihood values correspond to the fitted parameters
- information criteria like BIC (and LR tests, etc.) are computed consistently

Notes
-----
- t is assumed positive (we use log(t) for the logistic model).
- We use parameter transforms to enforce constraints (lambda>0, k>0).
- We try multiple starting points to reduce optimization failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


_EPS = 1e-12


def bernoulli_loglik(p: np.ndarray, y: np.ndarray) -> float:
    """Bernoulli log-likelihood with clipping for numerical stability."""
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logistic_prob(t: np.ndarray, b0: float, b1: float) -> np.ndarray:
    return expit(b0 + b1 * np.log(t))


def exponential_prob(t: np.ndarray, lam: float) -> np.ndarray:
    return np.exp(-lam * t)


def weibull_prob(t: np.ndarray, lam: float, k: float) -> np.ndarray:
    return np.exp(-np.power(lam * t, k))


@dataclass(frozen=True)
class FitResult:
    params: Tuple[float, ...]
    ll: float
    success: bool
    message: str


def _fit_generic(
    nll: Callable[[np.ndarray], float],
    x0s: Iterable[np.ndarray],
    method: str = "L-BFGS-B",
) -> Tuple[np.ndarray, bool, str]:
    best = None
    best_fun = np.inf
    best_msg = ""
    for x0 in x0s:
        res = minimize(nll, x0=np.asarray(x0, dtype=float), method=method)
        if np.isfinite(res.fun) and res.fun < best_fun:
            best_fun = float(res.fun)
            best = np.asarray(res.x, dtype=float)
            best_msg = str(res.message)
    if best is None:
        return np.full_like(np.asarray(next(iter(x0s))), np.nan, dtype=float), False, "Optimization failed"
    return best, True, best_msg


def fit_logistic_mle(t: np.ndarray, y: np.ndarray) -> FitResult:
    """Fit p(t)=expit(b0 + b1 log t) by Bernoulli MLE."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    def nll(theta: np.ndarray) -> float:
        b0, b1 = theta
        p = logistic_prob(t, b0, b1)
        return -bernoulli_loglik(p, y)

    # crude but robust start points
    x0s = [
        np.array([2.0, -0.5]),
        np.array([0.0, -0.5]),
        np.array([2.0, -1.0]),
        np.array([1.0, -0.2]),
    ]

    theta, ok, msg = _fit_generic(nll, x0s)
    if not ok or not np.all(np.isfinite(theta)):
        return FitResult(params=(np.nan, np.nan), ll=np.nan, success=False, message=msg)

    ll = bernoulli_loglik(logistic_prob(t, float(theta[0]), float(theta[1])), y)
    return FitResult(params=(float(theta[0]), float(theta[1])), ll=ll, success=True, message=msg)


def fit_exponential_mle(t: np.ndarray, y: np.ndarray) -> FitResult:
    """Fit p(t)=exp(-lambda t) by Bernoulli MLE."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    def nll(theta: np.ndarray) -> float:
        (log_lam,) = theta
        lam = float(np.exp(log_lam))
        p = exponential_prob(t, lam)
        return -bernoulli_loglik(p, y)

    # starting lambda: rough scale from mean success
    ybar = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    tbar = float(np.mean(t))
    lam0 = max(1e-6, -np.log(ybar) / max(tbar, 1e-6))

    x0s = [
        np.array([np.log(lam0)]),
        np.array([np.log(max(lam0 * 0.3, 1e-6))]),
        np.array([np.log(max(lam0 * 3.0, 1e-6))]),
        np.array([np.log(0.01)]),
    ]

    theta, ok, msg = _fit_generic(nll, x0s)
    if not ok or not np.all(np.isfinite(theta)):
        return FitResult(params=(np.nan,), ll=np.nan, success=False, message=msg)

    lam = float(np.exp(theta[0]))
    ll = bernoulli_loglik(exponential_prob(t, lam), y)
    return FitResult(params=(lam,), ll=ll, success=True, message=msg)


def fit_weibull_mle(t: np.ndarray, y: np.ndarray) -> FitResult:
    """Fit p(t)=exp(-(lambda t)^k) by Bernoulli MLE."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    def nll(theta: np.ndarray) -> float:
        log_lam, log_k = theta
        lam = float(np.exp(log_lam))
        k = float(np.exp(log_k))
        p = weibull_prob(t, lam, k)
        return -bernoulli_loglik(p, y)

    ybar = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    tbar = float(np.mean(t))
    lam0 = max(1e-6, -np.log(ybar) / max(tbar, 1e-6))

    # try a handful of plausible k values
    k_starts = [0.5, 0.8, 1.0, 1.2, 1.5]
    x0s = [np.array([np.log(lam0), np.log(k0)]) for k0 in k_starts]
    x0s += [np.array([np.log(0.01), np.log(1.0)])]

    theta, ok, msg = _fit_generic(nll, x0s)
    if not ok or not np.all(np.isfinite(theta)):
        return FitResult(params=(np.nan, np.nan), ll=np.nan, success=False, message=msg)

    lam = float(np.exp(theta[0]))
    k = float(np.exp(theta[1]))
    ll = bernoulli_loglik(weibull_prob(t, lam, k), y)
    return FitResult(params=(lam, k), ll=ll, success=True, message=msg)


def bic_from_ll(ll: float, n_params: int, n_obs: int) -> float:
    return float(n_params * np.log(n_obs) - 2.0 * ll)
