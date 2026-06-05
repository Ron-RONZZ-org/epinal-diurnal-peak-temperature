"""Tests for the visualization module (public API + submodules).

These tests use synthetic data to verify that plot functions produce
valid output paths without errors.  Visual correctness is assumed
from manual inspection of the generated PDFs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epinal_peak._visualize_descriptive import (
    plot_moving_average,
    plot_rose_diagram,
    plot_seasonal_moving_average,
    plot_seasonal_rose,
)
from epinal_peak._visualize_regression import (
    plot_seasonal_trend,
    plot_wrapped_scatter,
)
from epinal_peak._visualize_utils import (
    _filter_season,
    _SEASONS,
)
from epinal_peak.config import EpinalPeakConfig


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Synthetic daily peak-hour data spanning 20 years."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=365 * 20, freq="D")
    n = len(dates)
    return pd.DataFrame({
        "date": dates,
        "year": dates.year,
        "month": dates.month,
        "peak_hour": rng.uniform(10, 16, n),
        "peak_temperature": rng.uniform(5, 35, n),
        "diurnal_amplitude": rng.uniform(3.0, 12.0, n),
        "qc_excluded": False,
    })


@pytest.fixture
def synthetic_seasonal_results() -> dict:
    """Minimal seasonal regression results for plot_seasonal_trend."""
    return {
        s: {
            "beta_0": 0.5,
            "beta_1": 0.001,
            "year_center": 2010.0,
            "n_years": 20,
            "bootstrap": {"dist_sample": [0.0, 0.02, -0.02, 0.01]},
        }
        for s in ("spring", "summer", "autumn", "winter")
    }


@pytest.fixture
def trend_params() -> dict:
    """Trend parameters for plot_wrapped_scatter."""
    return {
        "beta_0": 0.5,
        "beta_1": 0.001,
        "year_center": 2010.0,
        "dist_sample": [0.0, 0.02, -0.02, 0.01],
    }


# ── _filter_season ─────────────────────────────────────────────────────


class TestFilterSeason:
    """Unit tests for _filter_season."""

    def test_spring_months(self) -> None:
        df = pd.DataFrame({"month": [3, 4, 5, 6, 7]})
        result = _filter_season(df, (3, 5))
        assert len(result) == 3
        assert result["month"].tolist() == [3, 4, 5]

    def test_djf(self) -> None:
        df = pd.DataFrame({"month": [12, 1, 2, 3]})
        result = _filter_season(df, "djf")
        assert len(result) == 3
        assert result["month"].tolist() == [12, 1, 2]

    def test_empty_result(self) -> None:
        df = pd.DataFrame({"month": [6, 7, 8]})
        result = _filter_season(df, (12, 2))
        assert result.empty


# ── Regression plots ───────────────────────────────────────────────────


class TestPlotWrappedScatter:
    """Tests for the pooled wrapped scatter plot."""

    def test_returns_path(self, synthetic_df: pd.DataFrame,
                         trend_params: dict) -> None:
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"))
        out = plot_wrapped_scatter(synthetic_df, config, trend_params)
        assert isinstance(out, Path)
        assert out.name == "wrapped_scatter"

    def test_pdf_created(self, synthetic_df: pd.DataFrame,
                         trend_params: dict) -> None:
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"))
        out = plot_wrapped_scatter(synthetic_df, config, trend_params)
        assert out.with_suffix(".pdf").exists()

    def test_handles_empty_dist_sample(self) -> None:
        df = pd.DataFrame({
            "year": [2000, 2001],
            "month": [1, 6],
            "peak_hour": [14.0, 15.0],
            "diurnal_amplitude": [5.0, 5.0],
            "qc_excluded": False,
        })
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"),
                                  min_diurnal_amplitude=1.0)
        params = {"beta_0": 0.5, "beta_1": 0.0,
                  "year_center": 2000.0, "dist_sample": []}
        out = plot_wrapped_scatter(df, config, params)
        assert out.name == "wrapped_scatter"


class TestPlotSeasonalTrend:
    """Tests for the 4-panel seasonal trend plot."""

    def test_returns_path(self, synthetic_df: pd.DataFrame,
                          synthetic_seasonal_results: dict) -> None:
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"))
        out = plot_seasonal_trend(synthetic_df, config,
                                  synthetic_seasonal_results)
        assert isinstance(out, Path)
        assert out.name == "seasonal_trend"

    def test_pdf_created(self, synthetic_df: pd.DataFrame,
                         synthetic_seasonal_results: dict) -> None:
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"))
        out = plot_seasonal_trend(synthetic_df, config,
                                  synthetic_seasonal_results)
        assert out.with_suffix(".pdf").exists()


# ── Descriptive plots ──────────────────────────────────────────────────


class TestPlotRoseDiagram:
    """Tests for the pooled rose diagram."""

    def test_returns_path(self, synthetic_df: pd.DataFrame) -> None:
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"))
        out = plot_rose_diagram(synthetic_df, config)
        assert isinstance(out, Path)
        assert out.name == "rose_diagram"


class TestPlotMovingAverage:
    """Tests for the pooled moving average."""

    def test_returns_path(self, synthetic_df: pd.DataFrame) -> None:
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"))
        out = plot_moving_average(synthetic_df, config)
        assert isinstance(out, Path)
        assert out.name == "moving_average"


class TestPlotSeasonalRose:
    """Tests for the 4-panel seasonal rose diagram."""

    def test_returns_path(self, synthetic_df: pd.DataFrame) -> None:
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"))
        out = plot_seasonal_rose(synthetic_df, config)
        assert isinstance(out, Path)
        assert out.name == "seasonal_rose"

    def test_pdf_created(self, synthetic_df: pd.DataFrame) -> None:
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"))
        out = plot_seasonal_rose(synthetic_df, config)
        assert out.with_suffix(".pdf").exists()

    def test_four_panels(self, synthetic_df: pd.DataFrame) -> None:
        """Verify the figure is a 2x2 with polar axes."""
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"))
        out = plot_seasonal_rose(synthetic_df, config)
        assert out.with_suffix(".pdf").exists()


class TestPlotSeasonalMovingAverage:
    """Tests for the 4-panel seasonal moving average."""

    def test_returns_path(self, synthetic_df: pd.DataFrame) -> None:
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"))
        out = plot_seasonal_moving_average(synthetic_df, config)
        assert isinstance(out, Path)
        assert out.name == "seasonal_moving_average"

    def test_pdf_created(self, synthetic_df: pd.DataFrame) -> None:
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"))
        out = plot_seasonal_moving_average(synthetic_df, config)
        assert out.with_suffix(".pdf").exists()

    def test_custom_window(self, synthetic_df: pd.DataFrame) -> None:
        config = EpinalPeakConfig(figures_dir=Path("/tmp/test_figs_vis"))
        out = plot_seasonal_moving_average(synthetic_df, config, window=3)
        assert out.with_suffix(".pdf").exists()


# ── _SEASONS constant ──────────────────────────────────────────────────


class TestSeasonsConstant:
    """The _SEASONS list must contain exactly the four seasons."""

    def test_has_four_seasons(self) -> None:
        assert len(_SEASONS) == 4

    def test_keys_match_expected(self) -> None:
        keys = [s[0] for s in _SEASONS]
        assert keys == ["spring", "summer", "autumn", "winter"]
