"""Pytest configuration and shared fixtures for the Epinal peak project."""

from __future__ import annotations

import logging
from typing import Generator

import pandas as pd
import pytest

# ── Register custom markers ────────────────────────────────────────────


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``slow`` marker for long-running tests."""
    config.addinivalue_line("markers", "slow: marks tests that take >2 seconds to run")


# ── Autouse fixtures ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _disable_logging(caplog: pytest.LogCaptureFixture) -> Generator[None, None, None]:
    """Suppress log output during tests to avoid noise.

    The ``epinal_peak`` root logger is set to CRITICAL for the duration of
    every test.  Individual tests can override by calling
    ``caplog.set_level(...)``.
    """
    caplog.set_level(logging.CRITICAL, logger="epinal_peak")
    yield


# ── Shared data fixtures ───────────────────────────────────────────────


@pytest.fixture
def sample_temperature_data() -> pd.DataFrame:
    """Synthetic hourly temperature data for two days (CET-to-CEST transition).

    Returns:
        DataFrame with 48 hourly records spanning a DST transition (March
        26--27, 2023).  The short day (23 hours on March 26) tests DST
        edge cases in preprocessing.
    """
    # 2023-03-26 is the spring-forward DST day (CET -> CEST).
    # UTC: hours 00..23 for March 26, 00..23 for March 27 = 48 records.
    utc_hours = list(range(24)) + list(range(24))
    dates = ["2023-03-26"] * 24 + ["2023-03-27"] * 24
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2023-03-26", periods=48, freq="h", tz="UTC"
            ),
            "date": dates,
            "hour_utc": utc_hours,
            "temperature": [
                5.0 + (i % 12) * 0.5 for i in range(48)
            ],  # sinusoidal-ish
        }
    )


@pytest.fixture
def sample_peak_hours() -> pd.DataFrame:
    """Synthetic daily peak-hour data with a known weak trend.

    Simulates 30 years (1995--2024) of daily peak hours with a slight
    positive trend (~0.01 h/year).  Useful for testing regression fixtures.

    Returns:
        DataFrame with columns ``date``, ``year``, ``peak_hour``,
        ``peak_temperature``, ``season``.
    """
    rows: list[dict] = []
    rng = pd.date_range("1995-01-01", "2024-12-31", freq="D")
    for date in rng:
        year = date.year
        # Weak trend: peak hour shifts 0.01 h/year
        base_hour = 14.0 + 0.01 * (year - 1995)
        # Add seasonal variation: summer peaks ~2 h later on average
        month = date.month
        seasonal_offset = 1.5 if 6 <= month <= 8 else 0.0
        peak_hour = (base_hour + seasonal_offset) % 24
        rows.append(
            {
                "date": date,
                "year": year,
                "peak_hour": round(peak_hour, 1),
                "peak_temperature": 25.0 + 5.0 * (month in {6, 7, 8}),
                "season": "summer" if 6 <= month <= 8 else "winter",
            }
        )
    return pd.DataFrame(rows)
