"""Regression-focused visualisations for the peak-temperature analysis.

Provides ``plot_wrapped_scatter`` (pooled hexbin + trend) and
``plot_seasonal_trend`` (4-panel per-season hexbin + trend).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
from seaborn import color_palette

from epinal_peak.config import EpinalPeakConfig
from epinal_peak._visualize_utils import (
    _augment_circular,
    _bootstrap_ci_band,
    _dual_export,
    _filter_season,
    _hour_formatter,
    _load_valid_data,
    _SEASONS,
    _von_mises_trend,
)

_logger = logging.getLogger(__name__)


def plot_wrapped_scatter(
    df: pd.DataFrame,
    config: EpinalPeakConfig,
    trend_params: dict[str, Any],
) -> Path:
    """Scatter plot of peak hour vs. year with a von Mises trend overlay.

    Args:
        df: Daily peak-hour DataFrame.  Must contain ``year``,
            ``peak_hour``, ``qc_excluded``, ``diurnal_amplitude``.
        config: Pipeline configuration.
        trend_params: Dict with keys ``beta_0``, ``beta_1``,
            ``year_center``, and ``dist_sample`` (from analysis results).

    Returns:
        Path to the saved figure file (without extension).
    """
    valid = _load_valid_data(df, config)
    year = valid["year"].values
    peak = valid["peak_hour"].values

    # Build trend line and CI band
    ymin = max(config.year_start, int(year.min()))
    ymax = min(config.year_end, int(year.max()))
    grid = np.linspace(ymin, ymax, 200)

    trend = _von_mises_trend(
        grid,
        beta_0=trend_params["beta_0"],
        beta_1=trend_params["beta_1"],
        year_center=trend_params["year_center"],
    )
    ci_low, ci_upp = _bootstrap_ci_band(
        grid,
        beta_0=trend_params["beta_0"],
        year_center=trend_params["year_center"],
        dist_sample=trend_params["dist_sample"],
    )

    pal = color_palette("colorblind", 4)

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    # 2D hexagonal density binning
    hb = ax.hexbin(
        year, peak,
        gridsize=20,
        mincnt=1,
        cmap="Blues",
        alpha=0.7,
        edgecolors="none",
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Count", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # CI band
    ax.fill_between(
        grid, ci_low, ci_upp,
        alpha=0.35, color=pal[2], linewidth=0, label="95 % CI",
    )
    # Trend line
    ax.plot(
        grid, trend,
        color=pal[3], linewidth=1.8, label="von Mises trend",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Peak hour (local time)")

    ax.set_ylim(8, 22)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{int(v)}h")
    )
    ax.text(
        0.98, 0.02,
        "Circular y-axis: 0h \u2194 24h\n(shown 8h\u201322h)",
        transform=ax.transAxes, fontsize=6, color="gray",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="none", alpha=0.7),
    )
    ax.legend(fontsize=7, loc="lower left", framealpha=0.8)

    fig.tight_layout()
    out = config.figures_dir / "wrapped_scatter"
    _dual_export(fig, out)
    plt.close(fig)
    return out


def plot_seasonal_trend(
    df: pd.DataFrame,
    config: EpinalPeakConfig,
    seasonal_results: dict[str, Any],
) -> Path:
    """4-panel figure of per-season peak-hour trends.

    Panels are arranged 2x2 in order: spring (MAM), summer (JJA),
    autumn (SON), winter (DJF).  Each panel shows the von Mises
    regression trend for that season, overlaid on the daily scatter.

    Args:
        df: Daily peak-hour DataFrame.  Must contain ``month``.
        config: Pipeline configuration.
        seasonal_results: Dict with keys ``spring``, ``summer``,
            ``autumn``, ``winter``, each containing regression results
            with an optional ``bootstrap`` sub-dict.

    Returns:
        Path to the saved figure file (without extension).
    """
    valid = _load_valid_data(df, config)

    pal = color_palette("colorblind", 4)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharex=True, sharey=True)
    axs = axes.flatten()

    for ax, (key, label, months) in zip(axs, _SEASONS):
        sdf = _filter_season(valid, months)

        if sdf.empty or key not in seasonal_results:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            continue

        res = seasonal_results[key]
        b1 = res.get("beta_1")
        if b1 is None or (isinstance(b1, float) and np.isnan(b1)):
            ax.text(0.5, 0.5, "No fit", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            continue

        year = sdf["year"].values
        peak = sdf["peak_hour"].values

        ymin = max(config.year_start, int(year.min()))
        ymax = min(config.year_end, int(year.max()))
        grid = np.linspace(ymin, ymax, 150)

        trend = _von_mises_trend(
            grid,
            beta_0=res["beta_0"],
            beta_1=res["beta_1"],
            year_center=res["year_center"],
        )

        boot = res.get("bootstrap", {}) or {}
        ds: list[float] = boot.get("dist_sample", [])
        if ds:
            cl, cu = _bootstrap_ci_band(
                grid,
                beta_0=res["beta_0"],
                year_center=res["year_center"],
                dist_sample=ds,
            )
            ax.fill_between(grid, cl, cu, alpha=0.2, color=pal[2], linewidth=0)

        ax.hexbin(
            year, peak,
            gridsize=15,
            mincnt=1,
            cmap="Blues",
            alpha=0.6,
            edgecolors="none",
        )
        ax.plot(grid, trend, color=pal[3], linewidth=1.2)

        nyrs = res.get("n_years", sdf["year"].nunique())
        ax.set_title(f"{label}  (n={len(sdf)})", fontsize=9)
        ax.set_ylim(8, 22)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{int(v)}h")
        )

    for ax in axes[1, :]:
        ax.set_xlabel("Year")
    for ax in axes[:, 0]:
        ax.set_ylabel("Peak hour")

    fig.tight_layout()
    out = config.figures_dir / "seasonal_trend"
    _dual_export(fig, out)
    plt.close(fig)
    return out
