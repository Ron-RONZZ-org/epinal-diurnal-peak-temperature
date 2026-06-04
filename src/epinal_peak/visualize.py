"""Publication-quality figures for the peak-temperature analysis.

Generates wrapped scatter plots (main), seasonal trend panels,
rose diagrams, circular moving average, and annual context time series.
All figures are exported as PDF (vector) and PNG (raster) at 300 DPI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
from seaborn import color_palette

import epinal_peak
from epinal_peak.analysis import circular_mean
from epinal_peak.config import EpinalPeakConfig

_logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
_HOURS_TO_RAD = 2.0 * np.pi / 24.0
_RAD_TO_HOURS = 24.0 / (2.0 * np.pi)

# ── Private helpers ────────────────────────────────────────────────────


def _apply_style() -> None:
    """Apply publication-quality matplotlib defaults.

    Font sizes, DPI, tick direction, and spine visibility are set
    according to AGENTS-visualize.md specifications: axis labels >= 10 pt,
    tick labels >= 8 pt, titles >= 12 pt, sans-serif, 300 DPI savefig.
    """
    # ── NOTE ── seaborn is imported for its colorblind palette only.
    # All actual plotting is done with matplotlib for full control
    # over circular-axis wrapping and polar projections.
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


def _augment_circular(
    x: np.ndarray,
    y: np.ndarray,
    lo: float = 0.0,
    hi: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate points near the circular boundary for visual wrapping.

    Points within 2 hours of the upper bound are also plotted shifted
    down by ``hi``, and vice versa.  This makes the y-axis visually
    continuous across the 23↔0 wrap.

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


def _load_valid_data(df: pd.DataFrame, config: EpinalPeakConfig) -> pd.DataFrame:
    """Filter to QC-passing observations with sufficient diurnal amplitude.

    Applies the ``qc_excluded`` flag and the ``min_diurnal_amplitude``
    threshold from config.

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


def _hour_formatter() -> FuncFormatter:
    """Return a tick formatter that displays hours round-the-clock.

    Both ``0`` and ``24`` are labelled ``"0h"`` to reinforce the
    circular axis.
    """
    return FuncFormatter(
        lambda v, _: "0h" if abs(v) < 1e-6 or abs(v - 24.0) < 1e-6 else f"{int(v)}h"
    )


# ── Public plotting functions ──────────────────────────────────────────


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

    # Augment for circular wrapping
    xs, ys = _augment_circular(year, peak)
    xl, yl = _augment_circular(grid, trend)
    _, yl_lo = _augment_circular(grid, ci_low)
    _, yl_hi = _augment_circular(grid, ci_upp)

    pal = color_palette("colorblind", 4)

    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    # CI band
    ax.fill_between(
        xl, yl_lo, yl_hi,
        alpha=0.2, color=pal[0], linewidth=0, label="95 % CI",
    )
    # Scatter
    ax.scatter(
        xs, ys,
        s=4, c=[pal[0]], alpha=0.35, edgecolors="none",
        label=f"Daily obs. (n = {len(valid)})",
    )
    # Trend line
    ax.plot(
        xl, yl,
        color=pal[2], linewidth=1.5, label="von Mises trend",
    )

    # Axis styling
    ax.set_xlabel("Year")
    ax.set_ylabel("Peak hour (local time)")
    ax.set_ylim(0, 24)
    ax.yaxis.set_major_locator(MultipleLocator(3))
    ax.yaxis.set_major_formatter(_hour_formatter())
    ax.axhline(24, color="gray", linewidth=0.4, linestyle=":", alpha=0.4)
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

    Panels are arranged 2×2 in order: spring (MAM), summer (JJA),
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

    # Each entry: (key, label, month_range)
    _seasons: list[tuple[str, str, tuple[int, int] | str]] = [
        ("spring", "MAM", (3, 5)),
        ("summer", "JJA", (6, 8)),
        ("autumn", "SON", (9, 11)),
        ("winter", "DJF", "djf"),
    ]

    pal = color_palette("colorblind", 4)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharex=True, sharey=True)
    axs = axes.flatten()

    for ax, (key, label, months) in zip(axs, _seasons):
        # Filter data for this season
        if months == "djf":
            # DJF: December (year N-1), January & February (year N)
            mask = valid["month"].isin([12, 1, 2])
        else:
            start, end = months  # type: ignore[misc]
            mask = valid["month"].between(start, end)
        sdf = valid[mask].copy()

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

        # Bootstrap CI (optional — some variants may lack it)
        boot = res.get("bootstrap", {}) or {}
        ds: list[float] = boot.get("dist_sample", [])
        if ds:
            cl, cu = _bootstrap_ci_band(
                grid,
                beta_0=res["beta_0"],
                year_center=res["year_center"],
                dist_sample=ds,
            )
            _, yl = _augment_circular(grid, cl)
            _, yu = _augment_circular(grid, cu)
            ax.fill_between(grid, yl, yu, alpha=0.12, color=pal[0], linewidth=0)

        # Augment data
        xs, ys = _augment_circular(year, peak)
        xl, yl2 = _augment_circular(grid, trend)

        ax.scatter(xs, ys, s=3, c=[pal[0]], alpha=0.25, edgecolors="none")
        ax.plot(xl, yl2, color=pal[2], linewidth=1.2)

        nyrs = res.get("n_years", sdf["year"].nunique())
        ax.set_title(f"{label}  (n={len(sdf)}, {nyrs} yr)", fontsize=9)
        ax.set_ylim(0, 24)
        ax.yaxis.set_major_locator(MultipleLocator(6))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: "0" if abs(v) < 1e-6 or abs(v - 24) < 1e-6 else f"{int(v)}")
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
        (ax_l, early, f"Early ({int(early['year'].min())}–{split_year - 1})"),
        (ax_r, late, f"Late ({split_year}–{int(late['year'].max())})"),
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

    Shows the annual mean, ±1 standard deviation band, and full
    (min--max) range of ``peak_temperature`` for each year.

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

    # Full range
    ax.fill_between(
        annual["year"], annual["tmin"], annual["tmax"],
        alpha=0.12, color=pal[0], label="Annual range",
    )
    # ±1 SD
    ax.fill_between(
        annual["year"],
        annual["tmean"] - annual["tsd"],
        annual["tmean"] + annual["tsd"],
        alpha=0.25, color=pal[0], label="±1 SD",
    )
    # Mean
    ax.plot(
        annual["year"], annual["tmean"],
        color=pal[2], linewidth=1.2, label="Annual mean T$_{max}$",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Daily max temperature (°C)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

    fig.tight_layout()
    out = config.figures_dir / "context_timeseries"
    _dual_export(fig, out)
    plt.close(fig)
    return out


# ── CLI entry point ────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for the visualization stage.

    Loads the processed daily peak CSV and analysis results JSON,
    then generates all five figure types.  Each figure is wrapped in a
    try/except so that one failure does not prevent the others from
    being produced.
    """
    config = EpinalPeakConfig()
    epinal_peak.setup_logging(config)
    _apply_style()

    # ── Load daily data ──────────────────────────────────────────────
    csv_path = Path(config.processed_dir) / config.peak_data_filename
    if not csv_path.exists():
        _logger.error("Processed data not found: %s", csv_path)
        return
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    _logger.info("Loaded %d daily records from %s", len(df), csv_path)

    # ── Load analysis results ────────────────────────────────────────
    json_path = Path(config.results_dir) / config.analysis_results_filename
    if not json_path.exists():
        _logger.error("Analysis results not found: %s", json_path)
        return
    with open(json_path) as f:
        results = json.load(f)
    _logger.info("Loaded analysis results from %s", json_path)

    primary = results.get("primary", {}) or {}
    boot = primary.get("bootstrap", {}) or {}
    trend_params: dict[str, Any] = {
        "beta_0": primary.get("beta_0"),
        "beta_1": primary.get("beta_1"),
        "year_center": primary.get("year_center"),
        "dist_sample": boot.get("dist_sample", []),
    }
    seasonal_results = results.get("seasonal", {}) or {}

    # ── Generate all figures ─────────────────────────────────────────
    figure_tasks: list[tuple[str, Any, list[Any]]] = [
        ("wrapped_scatter", plot_wrapped_scatter, [df, config, trend_params]),
        ("seasonal_trend", plot_seasonal_trend, [df, config, seasonal_results]),
        ("rose_diagram", plot_rose_diagram, [df, config]),
        ("moving_average", plot_moving_average, [df, config]),
        ("context_timeseries", plot_context_timeseries, [df, config]),
    ]

    n_ok = 0
    n_fail = 0
    for name, func, args in figure_tasks:
        try:
            out = func(*args)
            _logger.info("✓ %s → %s", name, out)
            n_ok += 1
        except Exception:
            _logger.error("✗ %s failed", name, exc_info=True)
            n_fail += 1

    _logger.info(
        "Visualization complete: %d succeeded, %d failed. "
        "Figures in %s", n_ok, n_fail, config.figures_dir
    )


if __name__ == "__main__":
    main()
