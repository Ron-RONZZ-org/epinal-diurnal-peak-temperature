"""Temperature-weighted von Mises regression (supplementary analysis).

Days with larger diurnal temperature ranges have more clearly defined
peaks and contribute more weight to the log-likelihood.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import i0

_logger = logging.getLogger(__name__)

_HOURS_TO_RAD = 2.0 * np.pi / 24.0
_RAD_TO_HOURS = 24.0 / (2.0 * np.pi)


def _hours_to_radians(hours: np.ndarray) -> np.ndarray:
    """Convert hour-of-day values to radians."""
    return hours * _HOURS_TO_RAD


def _radians_to_hours(radians: np.ndarray | float) -> np.ndarray | float:
    """Convert radians to hour-of-day values, normalised to [0, 24)."""
    return (radians * _RAD_TO_HOURS) % 24.0


def _weighted_neg_log_likelihood(
    params: np.ndarray,
    theta: np.ndarray,
    year_centered: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted negative log-likelihood for von Mises regression.

    Each observation contributes ``w_i`` times the log-likelihood.
    The effective sample size is ``sum(w_i)``.
    """
    beta_0, beta_1, log_kappa = params
    kappa = np.exp(log_kappa)
    mu = beta_0 + beta_1 * year_centered
    n_eff = weights.sum()
    ll = np.sum(weights * kappa * np.cos(theta - mu)) - n_eff * np.log(
        2.0 * np.pi * i0(kappa)
    )
    return float(-ll)


def temperature_weighted_regression(
    df: pd.DataFrame,
    n_iter: int = 1000,
    ci_level: float = 0.95,
    weight_col: str = "diurnal_amplitude",
) -> dict[str, Any]:
    """Von Mises regression weighted by diurnal amplitude (supplementary).

    Days with larger diurnal ranges have more clearly defined peaks and
    receive more weight in the log-likelihood.

    Args:
        df: DataFrame with ``peak_hour``, ``year``, and *weight_col*.
        n_iter: Bootstrap iterations.
        ci_level: Bootstrap CI level.
        weight_col: Column name for observation weights.

    Returns:
        Dict with regression coefficients, bootstrap CI, and weight summary.
    """
    hours = df["peak_hour"].values.astype(float)
    year = df["year"].values.astype(float)
    weights = df[weight_col].values.astype(float)

    valid = ~(np.isnan(hours) | np.isnan(year) | np.isnan(weights) | (weights <= 0))
    if not valid.any():
        return {"status": "no_valid_observations"}

    hours = hours[valid]
    year = year[valid]
    weights = weights[valid]

    theta = _hours_to_radians(hours)
    year_center = float(np.mean(year))
    year_centered = year - year_center

    sin_sum = np.sin(theta).sum()
    cos_sum = np.cos(theta).sum()
    beta_0_init = np.arctan2(sin_sum, cos_sum)
    n = float(len(theta))
    r = np.sqrt(sin_sum**2 + cos_sum**2) / n
    if r < 0.99:
        kappa_init = max(r * (2.0 - r**2) / (1.0 - r**2), 0.01)
    else:
        kappa_init = 100.0

    opt_result = minimize(
        _weighted_neg_log_likelihood,
        x0=np.array([beta_0_init, 0.0, np.log(kappa_init)]),
        args=(theta, year_centered, weights),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )

    beta_0, beta_1, log_kappa = opt_result.x
    kappa = np.exp(log_kappa)
    beta_1_hpd = beta_1 * _radians_to_hours(np.array([1.0]))[0] * 10.0

    weighted_df = df.iloc[valid].copy()
    weighted_df["_weight"] = weights
    bootstrap_result = _bootstrap_weighted_engine(
        weighted_df, n_iter=n_iter, ci_level=ci_level
    )

    return {
        "status": "ok" if opt_result.success else "mle_did_not_converge",
        "beta_0": float(beta_0),
        "beta_1": float(beta_1),
        "beta_1_hours_per_decade": float(beta_1_hpd),
        "kappa": float(kappa),
        "year_center": year_center,
        "n_obs": int(len(theta)),
        "n_years": int(year_centered.size),
        "converged": bool(opt_result.success),
        "log_likelihood": float(-opt_result.fun),
        "bootstrap": bootstrap_result,
        "weight_var": weight_col,
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
    }


def _bootstrap_weighted_engine(
    df: pd.DataFrame,
    n_iter: int = 1000,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap resampling for the temperature-weighted regression.

    Resamples rows and extracts the weighted regression's slope.
    """
    alpha = 1.0 - ci_level
    lower_pct = 100.0 * alpha / 2.0
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    estimates: list[float] = []
    n = len(df)

    for i in range(n_iter):
        if (i + 1) % 100 == 0:
            _logger.info("Weighted bootstrap iteration %d / %d", i + 1, n_iter)

        resample = df.sample(n=n, replace=True, random_state=i)
        try:
            result = _run_weighted(resample)
        except (ValueError, RuntimeError):
            continue

        if result.get("status") != "ok":
            continue
        estimates.append(result["beta_1_hours_per_decade"])

    if len(estimates) < 2:
        return {
            "n_iter": n_iter,
            "ci_level": ci_level,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "median": np.nan,
            "std_error": np.nan,
            "status": "too_few_valid_resamples",
        }

    arr = np.array(estimates)
    return {
        "n_iter": len(estimates),
        "ci_level": ci_level,
        "ci_lower": float(np.nanpercentile(arr, lower_pct)),
        "ci_upper": float(np.nanpercentile(arr, upper_pct)),
        "median": float(np.nanmedian(arr)),
        "std_error": float(np.nanstd(arr, ddof=1)),
        "status": "ok",
    }


def _run_weighted(df: pd.DataFrame) -> dict[str, Any]:
    """Fit weighted von Mises regression on a DataFrame with ``_weight`` column."""
    hours = df["peak_hour"].values.astype(float)
    year = df["year"].values.astype(float)
    weights = df["_weight"].values.astype(float)

    valid = ~(np.isnan(hours) | np.isnan(year) | np.isnan(weights) | (weights <= 0))
    if not valid.any():
        return {"status": "no_valid_observations"}

    hours = hours[valid]
    year = year[valid]
    weights = weights[valid]

    theta = _hours_to_radians(hours)
    year_center = float(np.mean(year))
    year_centered = year - year_center

    sin_sum = np.sin(theta).sum()
    cos_sum = np.cos(theta).sum()
    beta_0_init = np.arctan2(sin_sum, cos_sum)
    n = float(len(theta))
    r = np.sqrt(sin_sum**2 + cos_sum**2) / n
    if r < 0.99:
        kappa_init = max(r * (2.0 - r**2) / (1.0 - r**2), 0.01)
    else:
        kappa_init = 100.0

    opt_result = minimize(
        _weighted_neg_log_likelihood,
        x0=np.array([beta_0_init, 0.0, np.log(kappa_init)]),
        args=(theta, year_centered, weights),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )

    beta_0, beta_1, log_kappa = opt_result.x
    kappa = np.exp(log_kappa)
    beta_1_hpd = beta_1 * _radians_to_hours(np.array([1.0]))[0] * 10.0

    return {
        "status": "ok" if opt_result.success else "mle_did_not_converge",
        "beta_0": float(beta_0),
        "beta_1": float(beta_1),
        "beta_1_hours_per_decade": float(beta_1_hpd),
        "kappa": float(kappa),
        "year_center": year_center,
        "n_obs": int(len(theta)),
        "converged": bool(opt_result.success),
        "log_likelihood": float(-opt_result.fun),
    }
