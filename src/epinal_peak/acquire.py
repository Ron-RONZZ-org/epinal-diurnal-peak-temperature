"""Data acquisition via pooch with local fallback.

Downloads sub-daily temperature records from Météo-France using ``pooch``.
If the remote source is unavailable, falls back to a local data directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import epinal_peak
from epinal_peak.config import EpinalPeakConfig

_logger = logging.getLogger(__name__)


def download_raw_data(config: EpinalPeakConfig) -> Path:
    """Download raw temperature data from Météo-France via pooch.

    Args:
        config: Pipeline configuration with station ID, registry file,
            and output paths.

    Returns:
        Path to the downloaded CSV file.

    .. note::
        Placeholder for M2 implementation.
    """
    _logger.info("download_raw_data called — not yet implemented")
    return config.raw_dir / config.raw_data_filename


def load_local_fallback(config: EpinalPeakConfig) -> Path:
    """Load temperature data from a local backup directory.

    Args:
        config: Pipeline configuration with fallback directory info.

    Returns:
        Path to the local fallback data file.

    .. note::
        Placeholder for M2 implementation.
    """
    _logger.info("load_local_fallback called — not yet implemented")
    return config.external_dir / config.local_fallback_dir / config.raw_data_filename


def validate_raw_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw data schema using pandera.

    Args:
        df: Raw temperature data.

    Returns:
        The validated DataFrame (or raises on schema violation).

    .. note::
        Placeholder for M2 implementation.
    """
    _logger.info("validate_raw_schema called — not yet implemented")
    return df


def main() -> None:
    """CLI entry point for the data acquisition stage."""
    config = EpinalPeakConfig()
    epinal_peak.setup_logging(config)

    _logger.info("Starting data acquisition — placeholder stage")
    _logger.warning("Not yet implemented — no data downloaded.")

    path = download_raw_data(config)
    _logger.info("Output path: %s", path)

    _logger.info("Completed data acquisition stage")


if __name__ == "__main__":
    main()
