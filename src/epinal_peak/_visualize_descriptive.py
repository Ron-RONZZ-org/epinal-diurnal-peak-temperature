"""Descriptive visualisations for the peak-temperature analysis.

Provides rose diagrams, moving circular means, annual context
time series, and their per-season multi-panel counterparts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from seaborn import color_palette

from epinal_peak._analysis_regression import circular_mean
from epinal_peak.config import EpinalPeakConfig
from epinal_peak._visualize_utils import (
    _dual_export,
    _filter_season,
    _hour_formatter,
    _load_valid_data,
    _HOURS_TO_RAD,
    _SEASONS,
)

_logger = logging.getLogger(__name__)


def plot_rose_diagram(
    df: pd.DataFrame,
    config: EpinalPeakConfig,
    split_year: int | None = None,
) -> Path:
    """Circular histogram (rose diagram) of peak hours.

    Splits the record at *split_year* (default: median year of valid
    data) into early and late periods and compares their circular
    distributions.

    Args:
        df: Daily peak-hour DataFrame.
        config: Pipeline configuration.
        split_year: Year boundary for early/late split.  If ``None``,
            uses the median year of valid observations.

    Returns:
        Path to the saved figure file (without extension).
    """
    valid = _load_valid_data(df, config)
    if split_year is None:
        split_year = int(valid["year"].median())

    early = valid[valid["year"] < split_year]
    late = valid[valid["year"] >= split_year]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(5.5, 3.0),
        subplot_kw={"projection": "polar"},
    )

    n_bins = 24
    bin_edges = np.linspace(0, 2.0 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    pal = color_palette("colorblind", 4)

    panels = [
        (ax_l, early, f"Early ({int(early['year'].min())}\u2013{split_year - 1})"),
        (ax_r, late, f"Late ({split_year}\u2013{int(late['year'].max())})"),
    ]

    for ax, pdf, lbl in panels:
        if pdf.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        radians = pdf["peak_hour"].values * _HOURS_TO_RAD
        counts, _ = np.histogram(radians, bins=bin_edges)
        freq = counts / counts.sum()

        ax.bar(
            bin_centers, freq,
            width=2.0 * np.pi / n_bins,
            color=pal[0],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.8,
        )
        ax.set_title(lbl, fontsize=9)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # clockwise
        ax.set_xticks(np.linspace(0, 2.0 * np.pi, 4, endpoint=False))
        ax.set_xticklabels(["0h", "6h", "12h", "18h"], fontsize=7)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0%}")
        )

    fig.tight_layout()
    out = config.figures_dir / "rose_diagram"
    _dual_export(fig, out)
    plt.close(fig)
    return out


def plot_moving_average(
    df: pd.DataFrame,
    config: EpinalPeakConfig,
    window: int = 5,
) -> Path:
    """Moving circular mean of peak hours over a multi-year window.

    For each year, computes the circular mean of peak hours in the
    window ``[year - window//2, year + window//2]`` (inclusive).

    Args:
        df: Daily peak-hour DataFrame.
        config: Pipeline configuration.
        window: Window length in years (default 5).

    Returns:
        Path to the saved figure file (without extension).
    """
    valid = _load_valid_data(df, config)
    years = np.arange(config.year_start, config.year_end + 1)
    means = np.full(len(years), np.nan)
    half = window // 2

    for i, y in enumerate(years):
        mask = valid["year"].between(y - half, y + half)
        vals = valid.loc[mask, "peak_hour"].values
        if len(vals) >= 30:
            means[i] = circular_mean(vals)

    ok = ~np.isnan(means)
    pal = color_palette("colorblind", 4)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(
        years[ok], means[ok],
        color=pal[0], linewidth=1.5, marker="o", markersize=3,
        label=f"{window}-yr circular mean",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean peak hour")
    ax.set_ylim(0, 24)
    ax.yaxis.set_major_locator(MultipleLocator(3))
    ax.yaxis.set_major_formatter(_hour_formatter())
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = config.figures_dir / "moving_average"
    _dual_export(fig, out)
    plt.close(fig)
    return out


def plot_context_timeseries(
    df: pd.DataFrame,
    config: EpinalPeakConfig,
) -> Path:
    """Annual mean and spread of daily maximum temperature.

    Args:
        df: Daily peak-hour DataFrame (must contain ``peak_temperature``).
        config: Pipeline configuration.

    Returns:
        Path to the saved figure file (without extension).
    """
    valid = _load_valid_data(df, config)

    annual = (
        valid.groupby("year")
        .agg(
            tmean=("peak_temperature", "mean"),
            tsd=("peak_temperature", "std"),
            tmin=("peak_temperature", "min"),
            tmax=("peak_temperature", "max"),
        )
        .reset_index()
    )

    pal = color_palette("colorblind", 4)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.fill_between(
        annual["year"], annual["tmin"], annual["tmax"],
        alpha=0.12, color=pal[0], label="Annual range",
    )
    ax.fill_between(
        annual["year"],
        annual["tmean"] - annual["tsd"],
        annual["tmean"] + annual["tsd"],
        alpha=0.25, color=pal[0], label="\u00b11 SD",
    )
    ax.plot(
        annual["year"], annual["tmean"],
        color=pal[2], linewidth=1.2, label="Annual mean T$_{max}$",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Daily max temperature (\u00b0C)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

    fig.tight_layout()
    out = config.figures_dir / "context_timeseries"
    _dual_export(fig, out)
    plt.close(fig)
    return out


# ── Per-season multi-panel figures ─────────────────────────────────────


def plot_seasonal_rose(
    df: pd.DataFrame,
    config: EpinalPeakConfig,
) -> Path:
    """4-panel rose diagram, one panel per meteorological season.

    Each panel shows the circular histogram (polar bar chart) of daily
    peak hours for that season, giving a direct visual of the marginal
    circular distribution.  Panel layout mirrors ``plot_seasonal_trend``.

    Args:
        df: Daily peak-hour DataFrame.
        config: Pipeline configuration.

    Returns:
        Path to the saved figure file (without extension).
    """
    valid = _load_valid_data(df, config)

    n_bins = 24
    bin_edges = np.linspace(0, 2.0 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    pal = color_palette("colorblind", 4)

    fig, axes = plt.subplots(
        2, 2, figsize=(5.5, 5.0),
        subplot_kw={"projection": "polar"},
    )
    axs = axes.flatten()

    for ax, (key, label, months) in zip(axs, _SEASONS):
        sdf = _filter_season(valid, months)

        if sdf.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        radians = sdf["peak_hour"].values * _HOURS_TO_RAD
        counts, _ = np.histogram(radians, bins=bin_edges)
        freq = counts / counts.sum()

        ax.bar(
            bin_centers, freq,
            width=2.0 * np.pi / n_bins,
            color=pal[0],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.8,
        )
        ax.set_title(f"{label}  (n={len(sdf)})", fontsize=9)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # clockwise
        ax.set_xticks(np.linspace(0, 2.0 * np.pi, 4, endpoint=False))
        ax.set_xticklabels(["0h", "6h", "12h", "18h"], fontsize=7)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0%}")
        )

    fig.tight_layout()
    out = config.figures_dir / "seasonal_rose"
    _dual_export(fig, out)
    plt.close(fig)
    return out


def plot_seasonal_moving_average(
    df: pd.DataFrame,
    config: EpinalPeakConfig,
    window: int = 5,
) -> Path:
    """4-panel moving circular mean, one panel per season.

    Each panel shows the ``window``-year rolling circular mean for one
    season, providing a non-parametric view of the trend that relaxes
    the linearity assumption of the von Mises GLM.

    Args:
        df: Daily peak-hour DataFrame.
        config: Pipeline configuration.
        window: Window length in years (default 5).

    Returns:
        Path to the saved figure file (without extension).
    """
    valid = _load_valid_data(df, config)
    years = np.arange(config.year_start, config.year_end + 1)
    half = window // 2
    pal = color_palette("colorblind", 4)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0), sharex=True, sharey=True)
    axs = axes.flatten()

    for ax, (key, label, months) in zip(axs, _SEASONS):
        sdf = _filter_season(valid, months)

        means = np.full(len(years), np.nan)
        for i, y in enumerate(years):
            mask = sdf["year"].between(y - half, y + half)
            vals = sdf.loc[mask, "peak_hour"].values
            if len(vals) >= 15:
                means[i] = circular_mean(vals)

        ok = ~np.isnan(means)
        ax.plot(
            years[ok], means[ok],
            color=pal[0], linewidth=1.5, marker="o", markersize=2.5,
            label=f"{window}-yr mean",
        )
        ax.set_title(f"{label}  (n={len(sdf)})", fontsize=9)
        ax.set_ylim(8, 22)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_formatter(_hour_formatter())
        if ok.any():
            ax.legend(fontsize=7)

    for ax in axes[1, :]:
        ax.set_xlabel("Year")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean peak hour")

    fig.tight_layout()
    out = config.figures_dir / "seasonal_moving_average"
    _dual_export(fig, out)
    plt.close(fig)
    return out
