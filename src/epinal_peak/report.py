"""Summary statistics and tabular reporting.

Computes key descriptive statistics (N, circular mean, circular standard
deviation, trend coefficient) and formats them as publication-ready tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import epinal_peak
from epinal_peak.config import EpinalPeakConfig

_logger = logging.getLogger(__name__)


def summarize_statistics(df: pd.DataFrame) -> dict:
    """Compute key descriptive statistics from peak-hour data.

    Args:
        df: DataFrame with a ``peak_hour`` column.

    Returns:
        Dictionary with keys ``n_observations``, ``circular_mean_hour``,
        ``circular_std_hours``, ``trend_coefficient``, etc.

    .. note::
        Placeholder — returns zero-filled dict.
    """
    _logger.info("summarize_statistics called — not yet implemented")
    return {
        "n_observations": 0,
        "circular_mean_hour": 0.0,
        "circular_std_hours": 0.0,
        "trend_coefficient": 0.0,
    }


def format_statistics_table(stats: dict) -> str:
    """Format summary statistics as a Markdown table.

    Args:
        stats: Dictionary returned by :func:`summarize_statistics`.

    Returns:
        A Markdown-formatted string.

    .. note::
        Placeholder for M7 implementation.
    """
    _logger.info("format_statistics_table called — not yet implemented")
    return "| Statistic | Value |\n|---|---|\n| N | 0 |"


def main() -> None:
    """CLI entry point for the reporting stage."""
    config = EpinalPeakConfig()
    epinal_peak.setup_logging(config)

    _logger.info("Starting report — placeholder stage")
    _logger.warning("Not yet implemented — no report generated.")

    df = pd.DataFrame({"peak_hour": [14, 15, 13]})
    stats = summarize_statistics(df)
    table = format_statistics_table(stats)
    print(table)

    _logger.info("Completed report stage")


if __name__ == "__main__":
    main()
