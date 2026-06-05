"""Publication-quality figures for the peak-temperature analysis.

This module is the public API entry point for figure generation.
It re-exports all plotting functions from submodules and provides a
``main()`` CLI entry point that generates every figure in sequence.

Individual plot implementations live in:
- ``_visualize_regression.py`` (wrapped scatter, seasonal trend)
- ``_visualize_descriptive.py`` (rose diagrams, moving averages, context)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

import epinal_peak
from epinal_peak._visualize_descriptive import (
    plot_context_timeseries,
    plot_moving_average,
    plot_rose_diagram,
    plot_seasonal_moving_average,
    plot_seasonal_rose,
)
from epinal_peak._visualize_regression import (
    plot_seasonal_trend,
    plot_wrapped_scatter,
)
from epinal_peak._visualize_utils import _apply_style
from epinal_peak.config import EpinalPeakConfig

_logger = logging.getLogger(__name__)

# Re-export public plotting functions for convenience.
__all__ = [
    "main",
    "plot_context_timeseries",
    "plot_moving_average",
    "plot_rose_diagram",
    "plot_seasonal_moving_average",
    "plot_seasonal_rose",
    "plot_seasonal_trend",
    "plot_wrapped_scatter",
]


# ── CLI entry point ────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for the visualization stage.

    Loads the processed daily peak CSV and analysis results JSON,
    then generates all figure types.  Each figure is wrapped in a
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
        ("seasonal_rose", plot_seasonal_rose, [df, config]),
        ("seasonal_moving_average", plot_seasonal_moving_average, [df, config]),
        ("rose_diagram", plot_rose_diagram, [df, config]),
        ("moving_average", plot_moving_average, [df, config]),
        ("context_timeseries", plot_context_timeseries, [df, config]),
    ]

    n_ok = 0
    n_fail = 0
    for name, func, args in figure_tasks:
        try:
            out = func(*args)
            _logger.info("\u2713 %s \u2192 %s", name, out)
            n_ok += 1
        except Exception:
            _logger.error("\u2717 %s failed", name, exc_info=True)
            n_fail += 1

    _logger.info(
        "Visualization complete: %d succeeded, %d failed. "
        "Figures in %s", n_ok, n_fail, config.figures_dir
    )


if __name__ == "__main__":
    main()
