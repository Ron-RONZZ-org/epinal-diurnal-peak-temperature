"""Epinal Diurnal Peak Temperature.

Investigate long-term trends in the diurnal timing of the daily
maximum temperature for Epinal, France.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from epinal_peak.config import EpinalPeakConfig

__version__ = "0.1.0"
__all__: list[str] = [
    "EpinalPeakConfig",
    "setup_logging",
]

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_ROOT_LOGGER_NAME = "epinal_peak"


def setup_logging(config: EpinalPeakConfig) -> None:
    """Configure the project-wide logger.

    Sets up dual-output logging: DEBUG and above to a rotating log file in
    *config.logs_dir*, INFO and above to stdout.  Idempotent — safe to call
    multiple times.

    Args:
        config: Pipeline configuration containing *logs_dir* and other
            parameters (only *logs_dir* is used here).

    Raises:
        OSError: If the log directory cannot be created.
    """
    root_logger = logging.getLogger(_ROOT_LOGGER_NAME)

    # Avoid duplicate handler registration.
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.DEBUG)

    logs_path = Path(config.logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(logs_path / "pipeline.log", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
