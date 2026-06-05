"""Core regression math for von Mises circular statistics and GLM.

This module contains the low-level math for the von Mises negative
log-likelihood, the regression fitter, and a thin bootstrap wrapper.
Conversion helpers, circular mean/std, and shared constants live in
:mod:`_circular_utils`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import i0

from epinal_peak._circular_utils import (
    _hours_to_radians,
    _radians_to_hours,
    _HOURS_TO_RAD,
    _RAD_TO_HOURS,
    circular_mean,
    circular_std,
)

_logger = logging.getLogger(__name__)


def _neg_log_likelihood(
    params: np.ndarray, theta: np.ndarray, year_centered: np.ndarray
) -> float:
    """Negative log-likelihood of the von Mises regression model.

    Args:
        params: [beta_0, beta_1, log_kappa] where kappa = exp(log_kappa).
        theta: Circular response in radians.
        year_centered: Centered year predictor.

    Returns:
        Negative log-likelihood value.
    """
    beta_0, beta_1, log_kappa = params
    kappa = np.exp(log_kappa)
    mu = beta_0 + beta_1 * year_centered
    n = len(theta)

    ll = np.sum(kappa * np.cos(theta - mu)) - n * np.log(2.0 * np.pi * i0(kappa))
    return float(-ll)


def von_mises_regression(df: pd.DataFrame) -> dict[str, Any]:
    """Fit a von Mises GLM with year as a linear predictor.

    Maximises the von Mises log-likelihood directly using
    ``scipy.optimize.minimize`` (Fisher & Lee 1992).

    The model is:

    .. math::

        \\theta_i = 2\\pi \\times \\text{peak\\_hour}_i / 24
        \\mu_i = \\beta_0 + \\beta_1 \\times \\text{year\\_centered}_i
        \\log L = \\sum_i \\kappa \\cos(\\theta_i - \\mu_i)
                  - n \\log(2\\pi I_0(\\kappa))

    where :math:`I_0` is the modified Bessel function
    (``scipy.special.i0``).  :math:`\\kappa = \\exp(\\text{log\\_kappa})`
    ensures :math:`\\kappa > 0`.

    Args:
        df: DataFrame with columns ``year`` and ``peak_hour``.

    Returns:
        Dictionary with keys:
            - ``beta_0``: Intercept (radians).
            - ``beta_1``: Slope (radians/year).
            - ``beta_1_hours_per_decade``: Slope in hours/decade.
            - ``kappa``: Estimated concentration parameter.
            - ``year_center``: Year centering value.
            - ``n_obs``: Number of observations used.
            - ``converged``: Whether the MLE optimisation converged.
            - ``log_likelihood``: Maximised log-likelihood.
            - ``status``: ``"ok"`` or ``"error"``.

    Raises:
        ValueError: If the DataFrame is empty or missing required columns.
    """
    if df.empty:
        raise ValueError("DataFrame is empty — nothing to fit.")
    for col in ("year", "peak_hour"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    peak_hour = df["peak_hour"].values.astype(float)
    year = df["year"].values.astype(float)

    valid = ~(np.isnan(peak_hour) | np.isnan(year))
    if not valid.any():
        raise ValueError("No valid observations after NaN removal.")

    peak_hour = peak_hour[valid]
    year = year[valid]

    if np.ptp(peak_hour) == 0.0:
        return {
            "beta_0": _hours_to_radians(peak_hour[0]),
            "beta_1": 0.0,
            "beta_1_hours_per_decade": 0.0,
            "kappa": np.nan,
            "year_center": float(np.mean(year)),
            "n_obs": len(peak_hour),
            "converged": True,
            "log_likelihood": np.nan,
            "status": "all_peak_hours_identical",
        }

    if len(np.unique(year)) < 2:
        return {
            "beta_0": np.nan,
            "beta_1": np.nan,
            "beta_1_hours_per_decade": np.nan,
            "kappa": np.nan,
            "year_center": float(year.mean()),
            "n_obs": len(peak_hour),
            "converged": False,
            "log_likelihood": np.nan,
            "status": "insufficient_unique_years",
        }

    theta = _hours_to_radians(peak_hour)
    year_center = float(np.mean(year))
    year_centered = year - year_center

    sin_sum = np.sin(theta).sum()
    cos_sum = np.cos(theta).sum()
    beta_0_init = np.arctan2(sin_sum, cos_sum)
    beta_1_init = 0.0

    n = float(len(theta))
    r = np.sqrt(sin_sum**2 + cos_sum**2) / n
    if r < 0.99:
        kappa_init = max(r * (2.0 - r**2) / (1.0 - r**2), 0.01)
    else:
        kappa_init = 100.0
    log_kappa_init = np.log(kappa_init)

    result = minimize(
        _neg_log_likelihood,
        x0=np.array([beta_0_init, beta_1_init, log_kappa_init]),
        args=(theta, year_centered),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )

    beta_0, beta_1, log_kappa = result.x
    kappa = np.exp(log_kappa)
    beta_1_hours_per_decade = beta_1 * _RAD_TO_HOURS * 10.0

    return {
        "beta_0": float(beta_0),
        "beta_1": float(beta_1),
        "beta_1_hours_per_decade": float(beta_1_hours_per_decade),
        "kappa": float(kappa),
        "year_center": year_center,
        "n_obs": int(len(theta)),
        "converged": bool(result.success),
        "log_likelihood": float(-result.fun),
        "status": "ok" if result.success else "mle_did_not_converge",
    }


def bootstrap_trend(
    df: pd.DataFrame, n_iter: int = 1000, ci_level: float = 0.95
) -> tuple[float, float, float]:
    """Bootstrap the trend coefficient and return percentile CIs.

    Resamples daily observations with replacement, refits the von Mises
    regression each iteration, and extracts percentile confidence
    intervals for ``beta_1_hours_per_decade``.

    Args:
        df: DataFrame with columns ``year`` and ``peak_hour``.
        n_iter: Number of bootstrap resamples.
        ci_level: Confidence level for percentile intervals.

    Returns:
        Tuple ``(lower_ci, median_coefficient, upper_ci)`` in hours/decade.
    """
    from epinal_peak._analysis_bootstrap import _bootstrap_engine

    result = _bootstrap_engine(df, n_iter=n_iter, ci_level=ci_level)
    return (result["ci_lower"], result["median"], result["ci_upper"])
