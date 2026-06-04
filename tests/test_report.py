"""Tests for the report module — summary statistics and table formatting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from epinal_peak.config import EpinalPeakConfig
from epinal_peak.report import (
    _extract_ci,
    _make_decade_label,
    _safe_circ,
    export_stats,
    format_descriptive_table,
    format_primary_table,
    format_seasonal_table,
    format_sensitivity_table,
    load_analysis_results,
    load_peak_data,
    summarize_statistics,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small synthetic peak-hour DataFrame."""
    rng = np.random.default_rng(42)
    n = 100
    years = np.random.randint(1986, 2026, size=n)
    peak_hours = rng.integers(10, 18, size=n).astype(float)
    # Add some NaNs
    peak_hours[::10] = np.nan
    return pd.DataFrame({
        "date": pd.to_datetime([f"{y}-06-15" for y in years]),
        "year": years.astype(float),
        "month": 6,
        "peak_hour": peak_hours,
        "peak_temperature": rng.uniform(15, 35, size=n),
        "diurnal_amplitude": rng.uniform(2.0, 12.0, size=n),
        "qc_excluded": False,
    })


@pytest.fixture
def sample_primary() -> dict:
    """A synthetic primary analysis result dict."""
    return {
        "beta_0": -2.10,
        "beta_1": -0.000735,
        "beta_1_hours_per_decade": -0.0281,
        "kappa": 2.45,
        "year_center": 2005.5,
        "n_obs": 1000,
        "converged": True,
        "log_likelihood": -1200.0,
        "status": "ok",
        "bootstrap": {
            "n_iter": 1000,
            "ci_level": 0.95,
            "ci_lower": -0.070,
            "ci_upper": 0.012,
            "median": -0.029,
            "std_error": 0.021,
            "status": "ok",
        },
    }


@pytest.fixture
def sample_sensitivity() -> dict:
    """A synthetic sensitivity analysis result dict."""
    return {
        "tie_rule_latest": {
            "beta_1_hours_per_decade": -0.025,
            "n_obs": 1000,
            "status": "ok",
            "bootstrap": {"ci_lower": -0.065, "ci_upper": 0.015},
        },
        "min_years_20": {
            "beta_1_hours_per_decade": -0.030,
            "n_obs": 1000,
            "status": "ok",
            "bootstrap": {"ci_lower": -0.072, "ci_upper": 0.010},
        },
        "subsampling_3hr": {
            "beta_1_hours_per_decade": -0.020,
            "n_obs": 1000,
            "status": "ok",
            "bootstrap": {"ci_lower": -0.060, "ci_upper": 0.018},
        },
    }


@pytest.fixture
def sample_seasonal() -> dict:
    """A synthetic seasonal stratification result dict."""
    return {
        "spring": {
            "beta_1_hours_per_decade": -0.015,
            "n_obs": 250,
            "status": "ok",
            "label": "MAM",
            "bootstrap": {"ci_lower": -0.080, "ci_upper": 0.050},
        },
        "summer": {
            "beta_1_hours_per_decade": -0.040,
            "n_obs": 250,
            "status": "ok",
            "label": "JJA",
            "bootstrap": {"ci_lower": -0.095, "ci_upper": 0.015},
        },
        "autumn": {
            "beta_1_hours_per_decade": None,
            "n_obs": 0,
            "status": "no_data",
            "label": "SON",
        },
        "winter": {
            "beta_1_hours_per_decade": -0.010,
            "n_obs": 250,
            "status": "ok",
            "label": "DJF",
            "bootstrap": {"ci_lower": -0.070, "ci_upper": 0.048},
        },
    }


# ── Tests ───────────────────────────────────────────────────────────────


class TestMakeDecadeLabel:
    def test_typical_year(self) -> None:
        assert _make_decade_label(1992) == "1986-1995"

    def test_boundary_start(self) -> None:
        assert _make_decade_label(1986) == "1986-1995"

    def test_boundary_end(self) -> None:
        assert _make_decade_label(1995) == "1986-1995"

    def test_next_decade(self) -> None:
        assert _make_decade_label(1996) == "1996-2005"

    def test_last_decade(self) -> None:
        assert _make_decade_label(2025) == "2016-2025"


class TestSafeCirc:
    def test_empty_array(self) -> None:
        assert np.isnan(_safe_circ(np.mean, np.array([])))

    def test_valid_values(self) -> None:
        from epinal_peak.analysis import circular_mean
        result = _safe_circ(circular_mean, np.array([14.0, 15.0, 16.0]))
        assert not np.isnan(result)

    def test_nan_values(self) -> None:
        from epinal_peak.analysis import circular_mean
        result = _safe_circ(circular_mean, np.array([np.nan, np.nan]))
        assert np.isnan(result)


class TestExtractCI:
    def test_valid_bootstrap(self) -> None:
        lo, hi = _extract_ci({"ci_lower": -0.07, "ci_upper": 0.012})
        assert lo == pytest.approx(-0.07)
        assert hi == pytest.approx(0.012)

    def test_none_bootstrap(self) -> None:
        lo, hi = _extract_ci(None)
        assert np.isnan(lo)
        assert np.isnan(hi)

    def test_empty_dict(self) -> None:
        lo, hi = _extract_ci({})
        assert np.isnan(lo)
        assert np.isnan(hi)


class TestSummarizeStatistics:
    def test_overall_keys(self, sample_df: pd.DataFrame) -> None:
        stats = summarize_statistics(sample_df)
        assert "overall" in stats
        assert "by_decade" in stats

    def test_overall_n(self, sample_df: pd.DataFrame) -> None:
        stats = summarize_statistics(sample_df)
        # 100 rows, 10 with NaN peak_hour (every 10th), so ~90 valid
        assert stats["overall"]["n_observations"] > 80
        assert stats["overall"]["n_observations"] <= 100

    def test_by_decade_present(self, sample_df: pd.DataFrame) -> None:
        stats = summarize_statistics(sample_df)
        assert len(stats["by_decade"]) >= 4  # 1986-2025 spans 4 decades

    def test_decade_has_keys(self, sample_df: pd.DataFrame) -> None:
        stats = summarize_statistics(sample_df)
        for period in stats["by_decade"]:
            assert "decade" in period
            assert "n_observations" in period
            assert "circular_mean_hour" in period
            assert "circular_std_hours" in period

    def test_circular_mean_ranges(self, sample_df: pd.DataFrame) -> None:
        stats = summarize_statistics(sample_df)
        mean = stats["overall"]["circular_mean_hour"]
        # Data generated in [10, 18), so mean should be in that range
        assert 10.0 <= mean <= 18.0

    def test_circular_std_nonnegative(self, sample_df: pd.DataFrame) -> None:
        stats = summarize_statistics(sample_df)
        std = stats["overall"]["circular_std_hours"]
        assert std >= 0.0


class TestFormatPrimaryTable:
    def test_returns_string(self, sample_primary: dict) -> None:
        table = format_primary_table(sample_primary)
        assert isinstance(table, str)
        assert len(table) > 50

    def test_contains_beta(self, sample_primary: dict) -> None:
        table = format_primary_table(sample_primary)
        assert "-0.0281" in table

    def test_contains_ci(self, sample_primary: dict) -> None:
        table = format_primary_table(sample_primary)
        assert "-0.0700" in table or "-0.070" in table

    def test_contains_kappa(self, sample_primary: dict) -> None:
        table = format_primary_table(sample_primary)
        assert "2.45" in table

    def test_no_bootstrap(self) -> None:
        primary = {
            "beta_1_hours_per_decade": 0.01,
            "kappa": 2.0,
            "n_obs": 500,
            "status": "ok",
        }
        table = format_primary_table(primary)
        assert "not available" in table

    def test_nan_beta(self) -> None:
        primary = {
            "beta_1_hours_per_decade": float("nan"),
            "kappa": 2.0,
            "n_obs": 0,
            "status": "error",
        }
        table = format_primary_table(primary)
        assert "—" in table


class TestFormatSensitivityTable:
    def test_returns_string(self, sample_sensitivity: dict) -> None:
        table = format_sensitivity_table(sample_sensitivity)
        assert isinstance(table, str)
        assert len(table) > 50

    def test_contains_variants(self, sample_sensitivity: dict) -> None:
        table = format_sensitivity_table(sample_sensitivity)
        assert "Tie Rule Latest" in table or "tie_rule" in table

    def test_none_input(self) -> None:
        table = format_sensitivity_table(None)
        assert "No sensitivity" in table

    def test_empty_dict(self) -> None:
        table = format_sensitivity_table({})
        assert "No sensitivity" in table


class TestFormatSeasonalTable:
    def test_returns_string(self, sample_seasonal: dict) -> None:
        table = format_seasonal_table(sample_seasonal)
        assert isinstance(table, str)
        assert len(table) > 50

    def test_contains_seasons(self, sample_seasonal: dict) -> None:
        table = format_seasonal_table(sample_seasonal)
        assert "MAM" in table
        assert "JJA" in table
        assert "SON" in table
        assert "DJF" in table

    def test_none_input(self) -> None:
        table = format_seasonal_table(None)
        assert "No seasonal" in table

    def test_empty_dict(self) -> None:
        table = format_seasonal_table({})
        assert "No seasonal" in table


class TestFormatDescriptiveTable:
    def test_returns_string(self, sample_df: pd.DataFrame) -> None:
        stats = summarize_statistics(sample_df)
        table = format_descriptive_table(stats)
        assert isinstance(table, str)
        assert len(table) > 50

    def test_contains_overall(self, sample_df: pd.DataFrame) -> None:
        stats = summarize_statistics(sample_df)
        table = format_descriptive_table(stats)
        assert "Overall" in table


class TestExportStats:
    def test_creates_files(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        stats = summarize_statistics(sample_df)
        # Override results_dir for testing
        config = EpinalPeakConfig(results_dir=tmp_path)
        export_stats(stats, config)

        csv_path = tmp_path / "summary_stats.csv"
        json_path = tmp_path / "summary_stats.json"
        assert csv_path.exists()
        assert json_path.exists()

    def test_csv_content(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        stats = summarize_statistics(sample_df)
        config = EpinalPeakConfig(results_dir=tmp_path)
        export_stats(stats, config)

        df = pd.read_csv(tmp_path / "summary_stats.csv")
        assert "decade" in df.columns
        assert "circular_mean_hour" in df.columns
        assert len(df) >= 4  # at least 4 decades

    def test_json_content(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        stats = summarize_statistics(sample_df)
        config = EpinalPeakConfig(results_dir=tmp_path)
        export_stats(stats, config)

        with open(tmp_path / "summary_stats.json") as f:
            loaded = json.load(f)
        assert "overall" in loaded
        assert "by_decade" in loaded


class TestLoadPeakData:
    def test_file_not_found(self) -> None:
        config = EpinalPeakConfig(
            processed_dir=Path("/nonexistent"),
        )
        with pytest.raises(FileNotFoundError):
            load_peak_data(config)


class TestLoadAnalysisResults:
    def test_file_not_found(self) -> None:
        config = EpinalPeakConfig(
            results_dir=Path("/nonexistent"),
        )
        with pytest.raises(FileNotFoundError):
            load_analysis_results(config)
