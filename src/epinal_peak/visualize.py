"""Publication-quality figures for the peak-temperature analysis.

Generates wrapped scatter plots, rose diagrams, and seasonal trend plots.
All figures are exported as PDF (vector) and PNG (raster) at 300+ DPI.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import epinal_peak
from epinal_peak.config import EpinalPeakConfig

_logger = logging.getLogger(__name__)


def plot_wrapped_scatter(df: pd.DataFrame, config: EpinalPeakConfig) -> Path:
    """Scatter plot of peak hour vs. year with a smoothed trend overlay.

    Args:
        df: DataFrame with columns ``year``, ``peak_hour``, ``date``.
        config: Pipeline configuration with path and style settings.

    Returns:
        Path to the saved figure file.

    .. note::
        Placeholder for M6 implementation.
    """
    _logger.info("plot_wrapped_scatter called — not yet implemented")
    return config.figures_dir / "wrapped_scatter.pdf"


def plot_rose_diagram(df: pd.DataFrame, config: EpinalPeakConfig) -> Path:
    """Circular histogram (rose diagram) of peak hours.

    Args:
        df: DataFrame with a ``peak_hour`` column.
        config: Pipeline configuration with path and style settings.

    Returns:
        Path to the saved figure file.

    .. note::
        Placeholder for M6 implementation.
    """
    _logger.info("plot_rose_diagram called — not yet implemented")
    return config.figures_dir / "rose_diagram.pdf"


def plot_seasonal_trend(df: pd.DataFrame, config: EpinalPeakConfig) -> Path:
    """Separate trend plots for summer and winter months.

    Args:
        df: DataFrame with columns ``year``, ``peak_hour``, ``month``.
        config: Pipeline configuration with path and style settings.

    Returns:
        Path to the saved figure file.

    .. note::
        Placeholder for M6 implementation.
    """
    _logger.info("plot_seasonal_trend called — not yet implemented")
    return config.figures_dir / "seasonal_trend.pdf"


def main() -> None:
    """CLI entry point for the visualization stage."""
    config = EpinalPeakConfig()
    epinal_peak.setup_logging(config)

    _logger.info("Starting visualization — placeholder stage")
    _logger.warning("Not yet implemented — no figures generated.")

    df = pd.DataFrame({"year": [2000], "peak_hour": [14], "month": [7], "date": ["2000-01-01"]})
    _ = plot_wrapped_scatter(df, config)
    _ = plot_rose_diagram(df, config)
    _ = plot_seasonal_trend(df, config)

    _logger.info("Completed visualization stage")


if __name__ == "__main__":
    main()
