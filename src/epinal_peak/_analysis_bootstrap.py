"""Bootstrap resampling engine for von Mises regression confidence intervals."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# NOTE: von_mises_regression is imported lazily inside _bootstrap_engine
# to avoid a circular import (analysis.py imports from this module).


def _bootstrap_engine(
    df: pd.DataFrame,
    n_iter: int = 1000,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Resample with replacement, refit, return percentile CIs.

    Args:
        df: DataFrame with ``year`` and ``peak_hour`` columns.
        n_iter: Number of bootstrap resamples.
        ci_level: Confidence level (e.g., 0.95 for 95 %).

    Returns:
        Dict with keys:

        - ``n_iter``: Number of successful resamples.
        - ``ci_level``: Requested confidence level.
        - ``ci_lower``: Lower percentile bound (hours/decade).
        - ``ci_upper``: Upper percentile bound (hours/decade).
        - ``median``: Median of bootstrap distribution (hours/decade).
        - ``std_error``: Bootstrap standard error (hours/decade).
        - ``status``: ``"ok"`` or ``"too_few_valid_resamples"``.
        - ``dist_sample``: Full bootstrap distribution (list, for diagnostics).
    """
    alpha = 1.0 - ci_level
    lower_pct = 100.0 * alpha / 2.0
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    estimates: list[float] = []
    n = len(df)

    # Lazy import to avoid circular dependency
    from epinal_peak.analysis import von_mises_regression as _vm_reg

    for i in range(n_iter):
        if (i + 1) % 100 == 0:
            _logger.info("Bootstrap iteration %d / %d", i + 1, n_iter)

        resample = df.sample(n=n, replace=True, random_state=i)
        try:
            result = _vm_reg(resample)
        except (ValueError, RuntimeError):
            continue

        if result.get("status") not in ("ok", "all_peak_hours_identical"):
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
            "dist_sample": estimates,
            "status": "too_few_valid_resamples",
        }

    arr = np.array(estimates)
    ci_lower = float(np.nanpercentile(arr, lower_pct))
    ci_upper = float(np.nanpercentile(arr, upper_pct))
    median = float(np.nanmedian(arr))
    std_error = float(np.nanstd(arr, ddof=1))

    return {
        "n_iter": len(estimates),
        "ci_level": ci_level,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "median": median,
        "std_error": std_error,
        "dist_sample": [float(v) for v in estimates],
        "status": "ok",
    }


def bootstrap_trend(
    df: pd.DataFrame, n_iter: int = 1000, ci_level: float = 0.95
) -> tuple[float, float, float]:
    """Bootstrap CI for trend coefficient (backward-compatible public API).

    Args:
        df: DataFrame with ``year`` and ``peak_hour`` columns.
        n_iter: Number of bootstrap resamples.
        ci_level: Confidence level.

    Returns:
        Tuple ``(lower_ci, median, upper_ci)`` in hours/decade.
        Returns ``(NaN, NaN, NaN)`` if too few valid resamples.
    """
    result = _bootstrap_engine(df, n_iter=n_iter, ci_level=ci_level)
    return (result["ci_lower"], result["median"], result["ci_upper"])
