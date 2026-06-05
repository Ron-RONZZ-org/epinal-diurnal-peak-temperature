"""Private helper utilities for the visualization module.

Provides style configuration, file export, circular-augmentation helpers,
trend-line computation, and a reusable seasonal filter.
All functions in this module are private (prefixed ``_``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
from seaborn import color_palette

from epinal_peak._analysis_regression import circular_mean
from epinal_peak.config import EpinalPeakConfig

_logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

from epinal_peak._circular_utils import _HOURS_TO_RAD, _RAD_TO_HOURS

# Seasonal definitions shared across all multi-panel functions.
# Each entry: (dict_key, axis_label, month_range_or_"djf")
_SEASONS: list[tuple[str, str, tuple[int, int] | str]] = [
    ("spring", "MAM", (3, 5)),
    ("summer", "JJA", (6, 8)),
    ("autumn", "SON", (9, 11)),
    ("winter", "DJF", "djf"),
]


# ── Style and export ───────────────────────────────────────────────────


def _apply_style() -> None:
    """Apply publication-quality matplotlib defaults.

    Font sizes, DPI, tick direction, and spine visibility are set
    according to AGENTS-visualize.md specifications: axis labels >= 10 pt,
    tick labels >= 8 pt, titles >= 12 pt, sans-serif, 300 DPI savefig.
    """
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica"],
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
    })


def _dual_export(fig: plt.Figure, path: Path) -> None:
    """Save a figure as PDF (vector) and PNG (raster) at 300 DPI.

    Args:
        fig: Matplotlib figure to save.
        path: Output path *without extension*.  Parent directory is
            created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    _logger.debug("Exported %s.pdf and %s.png", path, path)


# ── Data helpers ───────────────────────────────────────────────────────


def _augment_circular(
    x: np.ndarray,
    y: np.ndarray,
    lo: float = 0.0,
    hi: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate points near the circular boundary for visual wrapping.

    Points within 2 hours of the upper bound are also plotted shifted
    down by ``hi``, and vice versa.  This makes the y-axis visually
    continuous across the 23harr0 wrap.

    Args:
        x: X-values (e.g. year).
        y: Y-values (e.g. peak hour, 0--23).
        lo: Lower bound of circular range.
        hi: Upper bound of circular range.

    Returns:
        ``(x_aug, y_aug)`` with duplicated points appended.
    """
    near_lo = y < lo + 2.0
    near_hi = y > hi - 2.0
    pieces_x: list[np.ndarray] = [x]
    pieces_y: list[np.ndarray] = [y]
    if near_lo.any():
        pieces_x.append(x[near_lo])
        pieces_y.append(y[near_lo] + hi)
    if near_hi.any():
        pieces_x.append(x[near_hi])
        pieces_y.append(y[near_hi] - hi)
    return np.concatenate(pieces_x), np.concatenate(pieces_y)


def _load_valid_data(df: pd.DataFrame, config: EpinalPeakConfig) -> pd.DataFrame:
    """Filter to QC-passing observations with sufficient diurnal amplitude.

    Args:
        df: Raw daily peak DataFrame (must contain ``qc_excluded`` and
            ``diurnal_amplitude``).
        config: Pipeline configuration.

    Returns:
        Filtered DataFrame with reset index.
    """
    valid = df[df["qc_excluded"] == False].copy()
    valid = valid[valid["diurnal_amplitude"] >= config.min_diurnal_amplitude]
    return valid.reset_index(drop=True)


def _filter_season(
    df: pd.DataFrame,
    months: tuple[int, int] | str,
) -> pd.DataFrame:
    """Filter a DataFrame to a meteorological season.

    Args:
        df: DataFrame with an integer ``month`` column.
        months: Either ``(start, end)`` for consecutive months or
            ``"djf"`` for December--January--February (year-crossing).

    Returns:
        Filtered DataFrame (may be empty).
    """
    if months == "djf":
        mask = df["month"].isin([12, 1, 2])
    else:
        start, end = months  # type: ignore[misc]
        mask = df["month"].between(start, end)
    return df[mask].copy()


# ── Trend computation ──────────────────────────────────────────────────


def _von_mises_trend(
    year_grid: np.ndarray,
    beta_0: float,
    beta_1: float,
    year_center: float,
) -> np.ndarray:
    """Evaluate the von Mises regression trend line over a year grid.

    Returns peak hour in the range [0, 24) by wrapping the linear
    prediction ``beta_0 + beta_1 * (year - year_center)`` through the
    circular mapping.

    Args:
        year_grid: Array of years to evaluate.
        beta_0: Model intercept (radians).
        beta_1: Model slope (radians/year).
        year_center: Centering value for the year predictor.

    Returns:
        Predicted peak-hour values in [0, 24).
    """
    mu_rad = beta_0 + beta_1 * (year_grid - year_center)
    return (mu_rad * _RAD_TO_HOURS) % 24.0


def _bootstrap_ci_band(
    year_grid: np.ndarray,
    beta_0: float,
    year_center: float,
    dist_sample: list[float],
    ci_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a bootstrap percentile CI band for the von Mises trend.

    Varies only the slope (drawn from ``dist_sample`` of
    ``beta_1_hours_per_decade``) while keeping the MLE intercept fixed.
    This produces a fan that widens away from ``year_center``.

    Args:
        year_grid: Array of years to evaluate.
        beta_0: MLE intercept (radians).
        year_center: Centering value.
        dist_sample: Bootstrap distribution of slopes (hours/decade).
        ci_level: Confidence level (e.g. 0.95).

    Returns:
        ``(ci_lower, ci_upper)`` in hours, same shape as ``year_grid``.
        Both are NaN-filled if ``dist_sample`` is empty.
    """
    if not dist_sample:
        n = len(year_grid)
        return np.full(n, np.nan), np.full(n, np.nan)

    alpha = 1.0 - ci_level
    lower_pct = 100.0 * alpha / 2.0
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    slopes = np.array(dist_sample, dtype=float)
    # hours/decade → radians/year
    slopes_rad = slopes / (_RAD_TO_HOURS * 10.0)

    ny = len(year_grid)
    preds = np.empty((len(slopes_rad), ny))
    for i, b1 in enumerate(slopes_rad):
        mu_rad = beta_0 + b1 * (year_grid - year_center)
        preds[i, :] = (mu_rad * _RAD_TO_HOURS) % 24.0

    ci_low = np.nanpercentile(preds, lower_pct, axis=0)
    ci_upp = np.nanpercentile(preds, upper_pct, axis=0)
    return ci_low, ci_upp


# ── Axis formatters ────────────────────────────────────────────────────


def _hour_formatter() -> FuncFormatter:
    """Return a tick formatter that displays hours round-the-clock.

    Both ``0`` and ``24`` are labelled ``"0h"`` to reinforce the
    circular axis.
    """
    return FuncFormatter(
        lambda v, _: "0h" if abs(v) < 1e-6 or abs(v - 24.0) < 1e-6 else f"{int(v)}h"
    )
