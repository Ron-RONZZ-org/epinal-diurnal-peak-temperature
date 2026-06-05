"""Quality-control functions for the preprocessing pipeline.

This module is private — all public symbols are re-exported via
``preprocess.py``.  Provides physical-limits filtering, flatline
(stuck-sensor) detection, and circular outlier flagging.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from epinal_peak._circular_utils import (
    _hours_to_radians,
    _radians_to_hours,
    _HOURS_IN_DAY,
)
from epinal_peak.config import EpinalPeakConfig

_logger = logging.getLogger(__name__)


def _circular_distance_rad(a: np.ndarray, b: float) -> np.ndarray:
    """Shortest angular distance (radians) between *a* and scalar *b*."""
    diff = a - b
    return np.arctan2(np.sin(diff), np.cos(diff))


# ── Temperature conversion + physical limits ────────────────────────────


def filter_physical_limits(df: pd.DataFrame, config: EpinalPeakConfig) -> pd.DataFrame:
    """Convert temperature to °C and drop physically implausible readings.

    The raw *T* column (already in °C) is copied to ``temperature_c``.
    Rows where ``temperature_c`` falls outside [*physical_temp_min*,
    *physical_temp_max*] are dropped.

    .. note::
        The Météo-France HOR files historically store temperature in
        0.1 °C units, but the ``acquire`` stage saves the filtered
        output with T already in °C.

    Args:
        df: DataFrame with a ``T`` column (°C).
        config: Pipeline configuration with *physical_temp_min* and
            *physical_temp_max*.

    Returns:
        DataFrame with ``temperature_c`` added and out-of-range rows
        removed.
    """
    df = df.copy()
    df["temperature_c"] = df["T"].astype(float)
    before = len(df)

    below = df["temperature_c"] < config.physical_temp_min
    above = df["temperature_c"] > config.physical_temp_max
    mask = ~(below | above)
    df = df.loc[mask].reset_index(drop=True)

    n_dropped = before - len(df)
    if n_dropped:
        _logger.warning(
            "Physical limits filter: %d rows outside [%g, %g]°C dropped",
            n_dropped,
            config.physical_temp_min,
            config.physical_temp_max,
        )
    else:
        _logger.info(
            "Physical limits filter: all %d rows within [%g, %g]°C",
            len(df),
            config.physical_temp_min,
            config.physical_temp_max,
        )
    return df


# ── Flatline (stuck sensor) detection ───────────────────────────────────


def _detect_flatline_in_group(
    temps: np.ndarray, n_req: int,
) -> bool:
    """Return ``True`` if *temps* contains a run of ≥*n_req* identical values.

    Two values are considered identical if their absolute difference
    is ≤ 0.01 °C.  NaN values break any run.
    """
    if len(temps) < n_req:
        return False

    max_run = 0
    current_run = 0
    prev = np.nan
    for t in temps:
        if np.isnan(t) or np.abs(t - prev) > 0.01:
            current_run = 1
        else:
            current_run += 1
        max_run = max(max_run, current_run)
        prev = t
    return max_run >= n_req


def flag_flatline(df: pd.DataFrame, config: EpinalPeakConfig) -> pd.DataFrame:
    """Flag days with ≥*flatline_consecutive_hours* identical temperatures.

    A *flatline* occurs when a sensor is stuck and reports the same
    temperature for many consecutive hours.  This function flags such
    days but does **not** filter them — downstream consumers may choose
    to exclude flagged records.

    Args:
        df: DataFrame with ``local_date``, ``local_time``, and
            ``temperature_c`` columns.
        config: Pipeline configuration with *flatline_consecutive_hours*.

    Returns:
        DataFrame with a ``flatline_flag`` column (``bool``).
    """
    df = df.copy()
    df = df.sort_values("local_time")
    n_req = config.flatline_consecutive_hours

    flagged_dates: set = set()
    for date_key, group in df.groupby("local_date", sort=False):
        temps = group["temperature_c"].values
        if _detect_flatline_in_group(temps, n_req):
            flagged_dates.add(date_key)
            _logger.debug("Flatline detected on %s", date_key)

    df["flatline_flag"] = df["local_date"].isin(flagged_dates)
    _logger.info(
        "Flatline check: %d / %d days flagged",
        len(flagged_dates),
        df["local_date"].nunique(),
    )
    return df


# ── Circular outlier detection (MAD-based) ──────────────────────────────


def flag_circular_outliers(
    peaks: pd.DataFrame,
    config: EpinalPeakConfig,
) -> pd.DataFrame:
    """Flag daily peak hours that are circular outliers.

    Uses the circular median absolute deviation (MAD) method on
    ``peak_hour_primary``.  A day is flagged if its circular distance
    from the circular median exceeds
    ``config.outlier_mad_threshold × circular_MAD``.

    Only non-excluded, non-NaN peak hours are used for the reference
    distribution.  Excluded (``qc_excluded``) days are never flagged.

    Args:
        peaks: Daily peak-hour DataFrame with ``peak_hour_primary``
            and ``qc_excluded`` columns.
        config: Pipeline configuration with *outlier_mad_threshold*.

    Returns:
        DataFrame with updated ``circular_outlier_flag`` column.
    """
    peaks = peaks.copy()

    # Reference: non-excluded, non-NaN peak hours.
    ref_mask = ~peaks["qc_excluded"] & peaks["peak_hour_primary"].notna()
    ref_hours = peaks.loc[ref_mask, "peak_hour_primary"].values

    if len(ref_hours) == 0:
        _logger.warning("No valid reference hours for circular outlier detection.")
        peaks["circular_outlier_flag"] = False
        return peaks

    ref_rad = _hours_to_radians(ref_hours)

    # Circular median: brute-force search over [0, 24) at 0.01 h resolution.
    candidates = np.linspace(0, _HOURS_IN_DAY, 2401, endpoint=False)
    cand_rad = _hours_to_radians(candidates)
    dists = np.abs(_circular_distance_rad(
        ref_rad[:, np.newaxis], cand_rad[np.newaxis, :],
    ))
    total_dist = np.nansum(dists, axis=0)
    median_hour = candidates[np.argmin(total_dist)]
    median_rad = _hours_to_radians(np.array([median_hour]))[0]

    # Circular MAD.
    abs_dev = np.abs(_circular_distance_rad(ref_rad, median_rad))
    mad = float(np.median(abs_dev))

    # When MAD is zero (e.g. all hours identical), flag nothing.
    if mad < 1e-6:
        _logger.info("Circular MAD ≈ 0 — no outliers flagged.")
        peaks["circular_outlier_flag"] = False
        return peaks

    threshold_rad = config.outlier_mad_threshold * mad

    # Compute distances for all non-excluded days.
    all_valid = peaks["peak_hour_primary"].notna()
    all_hours = peaks.loc[all_valid, "peak_hour_primary"].values
    all_rad = _hours_to_radians(all_hours)
    all_dists = np.abs(_circular_distance_rad(all_rad, median_rad))

    outlier_mask = np.zeros(len(peaks), dtype=bool)
    outlier_mask[all_valid] = all_dists > threshold_rad
    # Never flag excluded days.
    outlier_mask[peaks["qc_excluded"]] = False

    peaks["circular_outlier_flag"] = outlier_mask

    n_outliers = int(outlier_mask.sum())
    if n_outliers:
        _logger.info(
            "Circular outliers: %d / %d valid days flagged "
            "(MAD=%.3f h, threshold=%.3f h)",
            n_outliers,
            len(peaks),
            _radians_to_hours(np.array([mad]))[0],
            _radians_to_hours(np.array([threshold_rad]))[0],
        )

    return peaks
