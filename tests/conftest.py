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


def _season_label(month: int) -> str:
    """Map calendar month to meteorological season label."""
    if 3 <= month <= 5:
        return "spring"
    if 6 <= month <= 8:
        return "summer"
    if 9 <= month <= 11:
        return "autumn"
    return "winter"  # month 12, 1, 2


def _seasonal_offset(month: int) -> float:
    """Seasonal offset in hours for peak temperature timing.

    Summer peaks later (~1.5 h), winter peaks earliest, spring and
    autumn are intermediate.
    """
    if 3 <= month <= 5:
        return 0.5  # spring
    if 6 <= month <= 8:
        return 1.5  # summer
    if 9 <= month <= 11:
        return 0.5  # autumn
    return 0.0  # winter


def _peak_temperature_base(month: int) -> float:
    """Baseline daily max temperature by season (°C)."""
    if 6 <= month <= 8:
        return 30.0  # summer
    if 3 <= month <= 5:
        return 20.0  # spring
    if 9 <= month <= 11:
        return 18.0  # autumn
    return 8.0  # winter


@pytest.fixture
def sample_peak_hours() -> pd.DataFrame:
    """Synthetic daily peak-hour data with a known weak trend.

    Simulates 40 years (1986--2025) of daily peak hours with a slight
    positive trend (~0.01 h/year).  Uses 4-season meteorological
    labels (spring/summer/autumn/winter).  Useful for testing
    regression and sensitivity fixtures.

    Returns:
        DataFrame with columns ``date``, ``year``, ``peak_hour``,
        ``peak_temperature``, ``season``.
    """
    rows: list[dict] = []
    rng = pd.date_range("1986-01-01", "2025-12-31", freq="D")
    for date in rng:
        year = date.year
        month = date.month
        # Weak trend: peak hour shifts 0.01 h/year
        base_hour = 14.0 + 0.01 * (year - 1986)
        peak_hour = (base_hour + _seasonal_offset(month)) % 24
        rows.append(
            {
                "date": date,
                "year": year,
                "peak_hour": round(peak_hour, 1),
                "peak_temperature": _peak_temperature_base(month),
                "season": _season_label(month),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_peak_hours_amplitude() -> pd.DataFrame:
    """Synthetic daily peak hours with a ``diurnal_amplitude`` column.

    Same 40-year range as :func:`sample_peak_hours`, but includes a
    ``diurnal_amplitude`` column for testing the amplitude exclusion
    filter (``config.min_diurnal_amplitude``).  A small fraction of
    days (1 %) are given amplitudes below 2.0 °C to provide
    filterable records.

    Returns:
        DataFrame with columns ``date``, ``year``, ``peak_hour``,
        ``peak_temperature``, ``season``, ``diurnal_amplitude``.
    """
    import numpy as np

    rng = np.random.default_rng(seed=42)
    df = sample_peak_hours()  # reuse base fixture logic
    n = len(df)
    # Realistic amplitudes: most days 5--15 °C, a few below threshold
    base_amplitude = rng.uniform(5.0, 15.0, size=n).tolist()
    # Inject ~1 % low-amplitude days
    low_count = max(1, n // 100)
    low_indices = rng.choice(n, size=low_count, replace=False)
    for idx in low_indices:
        base_amplitude[idx] = rng.uniform(0.5, 1.9)
    df = df.copy()
    df["diurnal_amplitude"] = base_amplitude
    return df
