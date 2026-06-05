"""Shared circular statistics utilities for the Epinal peak project.

This module is the **single source of truth** for hour↔radian conversions,
circular mean / std, and shared constants like season names.  All
``_analysis_*.py`` modules and ``_preprocess_qc.py`` import from here
rather than redefining their own copies.
"""

from __future__ import annotations

import numpy as np

# ── Conversion constants ──────────────────────────────────────────────

_HOURS_IN_DAY: int = 24
"""Number of hours per day."""

_HOURS_TO_RAD: float = 2.0 * np.pi / _HOURS_IN_DAY
"""Conversion factor: hours → radians (2π / 24)."""

_RAD_TO_HOURS: float = _HOURS_IN_DAY / (2.0 * np.pi)
"""Conversion factor: radians → hours (24 / 2π)."""


# ── Season names (meteorological, Northern Hemisphere) ────────────────

_SEASON_ORDER: list[str] = ["spring", "summer", "autumn", "winter"]
"""Canonical season order — must match values used in the ``season`` column."""


# ── Conversion helpers ────────────────────────────────────────────────


def _hours_to_radians(hours: np.ndarray) -> np.ndarray:
    """Convert hour-of-day values to radians.

    Args:
        hours: Hour-of-day values (0--23).

    Returns:
        Radians on ``[0, 2π)``.
    """
    return hours * _HOURS_TO_RAD


def _radians_to_hours(radians: np.ndarray | float) -> np.ndarray | float:
    """Convert radians to hour-of-day values, normalised to ``[0, 24)``.

    Args:
        radians: Angular values in radians.

    Returns:
        Hour-of-day values (0--24, with 24 mapping to 0).
    """
    return (radians * _RAD_TO_HOURS) % _HOURS_IN_DAY


def circular_mean(hours: np.ndarray) -> float:
    """Compute the circular mean of hourly data (0--23 range).

    Converts hours to radians, computes the mean direction from the
    summed unit vectors, and converts back to hours.

    Args:
        hours: Array of hour-of-day values (0--23).  NaNs are ignored.

    Returns:
        Circular mean in hours (0--23).  Returns NaN if all values are
        NaN or if the resultant vector length is zero (uniform circular
        distribution).

    Examples:
        >>> circular_mean(np.array([0.0, 0.0]))
        0.0
        >>> circular_mean(np.array([23.0, 1.0]))
        0.0
    """
    hours = np.asarray(hours, dtype=float)
    valid = ~np.isnan(hours)
    if not valid.any():
        return np.nan

    radians = _hours_to_radians(hours[valid])
    sin_sum = np.sin(radians).sum()
    cos_sum = np.cos(radians).sum()
    mean_rad = np.arctan2(sin_sum, cos_sum)
    if mean_rad < 0.0:
        mean_rad += 2.0 * np.pi
    hours_out = mean_rad * _RAD_TO_HOURS
    return float(hours_out % _HOURS_IN_DAY)


def circular_std(hours: np.ndarray) -> float:
    """Compute the circular standard deviation of hourly data (0--23 range).

    Uses the mean resultant length *R*:

    .. math::

        \\sigma = \\sqrt{-2 \\ln(R)} \\times \\frac{24}{2\\pi}

    where *R* = :math:`\\sqrt{(\\sum \\cos \\theta)^2 + (\\sum \\sin \\theta)^2} / n`.

    Args:
        hours: Array of hour-of-day values (0--23).  NaNs are ignored.

    Returns:
        Circular standard deviation in hours (0--23).  Returns NaN if all
        values are NaN or *R* = 0 (uniform circular distribution).

    Examples:
        >>> np.round(circular_std(np.array([0.0, 0.0])), 6)
        0.0
    """
    hours = np.asarray(hours, dtype=float)
    valid = ~np.isnan(hours)
    if not valid.any():
        return np.nan

    radians = _hours_to_radians(hours[valid])
    n = float(len(radians))
    sin_sum = np.sin(radians).sum()
    cos_sum = np.cos(radians).sum()
    r = np.sqrt(sin_sum**2 + cos_sum**2) / n

    if r <= 0.0:
        return np.nan

    if r > 1.0:
        r = 1.0
    elif 1.0 - r < 1e-12:
        r = 1.0
    circular_std_rad = np.sqrt(-2.0 * np.log(r))
    return float(circular_std_rad * _RAD_TO_HOURS)


def circular_autocorr(series: np.ndarray, lag: int = 1) -> float:
    """Compute circular autocorrelation at a given lag.

    Uses the standard circular autocorrelation based on sine deviations
    (Jammalamadaka & SenGupta 2001, eq.~7.2.8):

    .. math::

        \\rho(k) = \\frac{\\sum_{i=1}^{n-k} \\sin(\\theta_i - \\bar{\\theta})
                             \\sin(\\theta_{i+k} - \\bar{\\theta})}
                            {\\sum_{i=1}^{n} \\sin^2(\\theta_i - \\bar{\\theta})}

    where :math:`\\bar{\\theta}` is the circular mean.  This is the circular
    analogue of the linear Pearson ACF and yields values in ``[-1, 1]``,
    with zero indicating no serial dependence.

    Args:
        series: Circular data in radians, shape ``(n,)``.
        lag: Time lag (positive integer).

    Returns:
        Circular autocorrelation coefficient.  Returns NaN if the series
        is too short or all-NaN.
    """
    series = np.asarray(series, dtype=float)
    valid = ~np.isnan(series)
    if not valid.any() or len(series) <= lag:
        return np.nan

    theta = series[valid]
    n = len(theta)
    if n <= lag:
        return np.nan

    # Circular mean
    sin_sum = np.sin(theta).sum()
    cos_sum = np.cos(theta).sum()
    mean_theta = np.arctan2(sin_sum, cos_sum)

    # Sine deviations from the circular mean
    dev = np.sin(theta - mean_theta)

    # Numerator: sum of sin-deviation products at lag
    num = np.sum(dev[: n - lag] * dev[lag:])

    # Denominator: sum of squared sin-deviations (full series)
    denom = np.sum(dev**2)

    if denom == 0.0:
        return np.nan

    return float(num / denom)
