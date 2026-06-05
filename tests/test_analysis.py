"""Tests for the analysis module — circular statistics and von Mises regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epinal_peak import EpinalPeakConfig
from epinal_peak import analysis


# ── Helpers ─────────────────────────────────────────────────────────────


def _synthetic_peak_hours(
    *,
    beta_1_hours_per_year: float = 0.01,
    kappa: float = 10.0,
    n_years: int = 40,
    days_per_year: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily peak-hour data with a known trend.

    Args:
        beta_1_hours_per_year: True slope in hours/year.
        kappa: Von Mises concentration parameter.
        n_years: Number of years.
        days_per_year: Days per year.
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with columns ``year``, ``peak_hour``, ``date``.
    """
    rng = np.random.default_rng(seed)
    total_days = n_years * days_per_year
    dates = pd.date_range("2000-01-01", periods=total_days, freq="D")

    years = dates.year.values.astype(float)
    year_center = float(years.mean())

    # True mu in radians: beta_0 + beta_1 * year_centered
    beta_0_rad = analysis._hours_to_radians(14.0)  # intercept at 14:00
    beta_1_rad = beta_1_hours_per_year * analysis._HOURS_TO_RAD
    mu_rad = beta_0_rad + beta_1_rad * (years - year_center)

    # Generate von Mises random deviations
    # scipy.stats.vonmises(mu=0, kappa=kappa) is centred at 0
    from scipy.stats import vonmises

    noise = vonmises.rvs(kappa, size=total_days, random_state=rng)
    theta = (mu_rad + noise) % (2.0 * np.pi)

    peak_hours = analysis._radians_to_hours(theta)

    return pd.DataFrame({"date": dates, "year": years, "peak_hour": peak_hours})


# ── Tests ───────────────────────────────────────────────────────────────


class TestCircularMean:
    """Tests for ``circular_mean()``."""

    def test_all_same_value(self) -> None:
        """All values identical returns that value."""
        result = analysis.circular_mean(np.array([14.0, 14.0, 14.0]))
        assert result == pytest.approx(14.0)

    def test_wrap_around(self) -> None:
        """Values near 0 and 23 wrap correctly (mean should be 0)."""
        result = analysis.circular_mean(np.array([23.0, 1.0]))
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self) -> None:
        """Symmetric values around a midpoint."""
        result = analysis.circular_mean(np.array([13.0, 15.0]))
        assert result == pytest.approx(14.0, abs=1e-2)

    def test_single_value(self) -> None:
        """Single value returns that value."""
        result = analysis.circular_mean(np.array([8.0]))
        assert result == pytest.approx(8.0)

    def test_all_nan(self) -> None:
        """All NaN returns NaN."""
        result = analysis.circular_mean(np.array([np.nan, np.nan]))
        assert np.isnan(result)

    def test_some_nan(self) -> None:
        """Partial NaN ignores NaN entries."""
        result = analysis.circular_mean(np.array([14.0, np.nan, 14.0]))
        assert result == pytest.approx(14.0)

    def test_linear_mean_differs(self) -> None:
        """Circular mean differs from linear mean for wraparound data."""
        linear = np.mean([23.0, 1.0])
        circular = analysis.circular_mean(np.array([23.0, 1.0]))
        assert circular != pytest.approx(linear)
        assert circular == pytest.approx(0.0, abs=1e-10)


class TestCircularStd:
    """Tests for ``circular_std()``."""

    def test_all_same_value(self) -> None:
        """All identical values have zero circular std."""
        result = analysis.circular_std(np.array([14.0, 14.0, 14.0]))
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_wider_distribution(self) -> None:
        """Wider spread -> larger circular std."""
        tight = analysis.circular_std(np.array([13.0, 14.0, 15.0]))
        wide = analysis.circular_std(np.array([10.0, 14.0, 18.0]))
        assert wide > tight

    def test_all_nan(self) -> None:
        """All NaN returns NaN."""
        result = analysis.circular_std(np.array([np.nan, np.nan]))
        assert np.isnan(result)

    def test_single_value(self) -> None:
        """Single value has zero circular std."""
        result = analysis.circular_std(np.array([8.0]))
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_some_nan(self) -> None:
        """Partial NaN ignores NaN entries."""
        result = analysis.circular_std(np.array([14.0, np.nan, 14.0]))
        assert result == pytest.approx(0.0, abs=1e-10)


class TestVonMisesRegression:
    """Tests for ``von_mises_regression()``."""

    def test_returns_dict_with_status(self) -> None:
        """Returns a dict with status key."""
        df = pd.DataFrame({"year": [2000], "peak_hour": [14]})
        result = analysis.von_mises_regression(df)
        assert isinstance(result, dict)
        assert "status" in result

    def test_empty_dataframe_raises(self) -> None:
        """Empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            analysis.von_mises_regression(pd.DataFrame())

    def test_missing_column_raises(self) -> None:
        """Missing required column raises ValueError."""
        with pytest.raises(ValueError, match="year"):
            analysis.von_mises_regression(pd.DataFrame({"foo": [1]}))

    def test_synthetic_zero_trend(self) -> None:
        """Synthetic data with zero trend returns beta_1 ≈ 0."""
        df = _synthetic_peak_hours(
            beta_1_hours_per_year=0.0,
            kappa=20.0,
            n_years=20,
            seed=123,
        )
        result = analysis.von_mises_regression(df)
        assert result["status"] == "ok"
        # With high kappa and zero trend, beta_1 should be very close to 0
        assert abs(result["beta_1_hours_per_decade"]) < 0.05

    def test_synthetic_positive_trend(self) -> None:
        """Synthetic data with known positive trend recovers the slope.

        This is the critical regression test (Guideline #8 from AGENTS.md):
        verifies that the von Mises regression returns the expected
        coefficient on synthetic circular data.
        """
        true_beta_hours_per_year = 0.02  # 0.2 h/decade
        df = _synthetic_peak_hours(
            beta_1_hours_per_year=true_beta_hours_per_year,
            kappa=15.0,
            n_years=30,
            days_per_year=365,
            seed=42,
        )
        result = analysis.von_mises_regression(df)
        assert result["status"] == "ok"
        assert result["converged"] is True

        # Recover beta_1 in hours/year
        recovered = result["beta_1_hours_per_decade"] / 10.0
        # With n=10950 obs and kappa=15, the estimate should be close
        assert recovered == pytest.approx(true_beta_hours_per_year, abs=0.005)

    def test_synthetic_negative_trend(self) -> None:
        """Synthetic data with known negative trend recovers the slope."""
        true_beta_hours_per_year = -0.015  # -0.15 h/decade
        df = _synthetic_peak_hours(
            beta_1_hours_per_year=true_beta_hours_per_year,
            kappa=12.0,
            n_years=25,
            days_per_year=365,
            seed=99,
        )
        result = analysis.von_mises_regression(df)
        assert result["status"] == "ok"
        assert result["converged"] is True

        recovered = result["beta_1_hours_per_decade"] / 10.0
        assert recovered == pytest.approx(true_beta_hours_per_year, abs=0.008)

    def test_all_identical_peak_hours(self) -> None:
        """All identical peak hours returns special status."""
        years = np.arange(2000, 2010)
        df = pd.DataFrame({"year": years, "peak_hour": [14.0] * 10})
        result = analysis.von_mises_regression(df)
        assert result["status"] == "all_peak_hours_identical"

    def test_single_year_returns_insufficient(self) -> None:
        """Single unique year returns insufficient status."""
        df = pd.DataFrame(
            {"year": [2000] * 100, "peak_hour": np.random.default_rng(42).uniform(0, 24, 100)}
        )
        result = analysis.von_mises_regression(df)
        assert result["status"] == "insufficient_unique_years"

    def test_high_kappa_low_noise(self) -> None:
        """With very high kappa, estimate is very precise."""
        true_beta = 0.01  # hours/year = 0.1 h/decade
        df = _synthetic_peak_hours(
            beta_1_hours_per_year=true_beta,
            kappa=50.0,
            n_years=20,
            days_per_year=365,
            seed=777,
        )
        result = analysis.von_mises_regression(df)
        assert result["status"] == "ok"
        recovered = result["beta_1_hours_per_decade"] / 10.0
        assert recovered == pytest.approx(true_beta, abs=0.002)


class TestBootstrapTrend:
    """Tests for ``bootstrap_trend()`` (placeholder for M5)."""

    def test_returns_tuple(self) -> None:
        """Placeholder returns a 3-tuple of floats."""
        df = pd.DataFrame({"year": [2000, 2001], "peak_hour": [14.0, 14.1]})
        result = analysis.bootstrap_trend(df)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)


class TestSensitivityAnalysis:
    """Tests for ``sensitivity_analysis()``."""

    def test_returns_dict_of_variants(self) -> None:
        """Returns a dict of variant results (6 expected)."""
        config = EpinalPeakConfig()
        df = pd.DataFrame({"year": [2000, 2001], "peak_hour": [14.0, 15.0]})
        result = analysis.sensitivity_analysis(df, config)
        assert isinstance(result, dict)
        # Each variant entry should have a status key
        for variant_name, variant_result in result.items():
            assert "status" in variant_result, f"{variant_name} missing status"


class TestMain:
    """Tests for ``main()`` entry point."""

    @pytest.mark.slow
    def test_main_runs_without_error(self) -> None:
        """``main()`` executes and returns None.

        .. note::

            This is a slow integration test (~minutes) that requires
            the processed data file and runs 1000 bootstrap iterations.
            Use ``pytest --runslow`` to execute.
        """
        result = analysis.main()
        assert result is None


class TestHelpers:
    """Tests for internal helper functions."""

    def test_hours_to_radians_known(self) -> None:
        """Known conversion: 12 hours -> pi, 6 hours -> pi/2."""
        assert analysis._hours_to_radians(np.array([12.0])) == pytest.approx(np.pi)
        assert analysis._hours_to_radians(np.array([6.0])) == pytest.approx(np.pi / 2)
        assert analysis._hours_to_radians(np.array([0.0])) == pytest.approx(0.0)

    def test_radians_to_hours_known(self) -> None:
        """Known conversion: pi -> 12, 2*pi -> 0."""
        assert analysis._radians_to_hours(np.array([np.pi])) == pytest.approx(12.0)
        assert analysis._radians_to_hours(np.array([2.0 * np.pi - 1e-10])) == pytest.approx(24.0 - 1e-10)


class TestSeasonalStratification:
    """Tests for ``seasonal_stratification()``."""

    def _seasonal_synthetic_data(self, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic data spanning all four seasons."""
        rng = np.random.default_rng(seed)
        dfs = []
        for season_name, month_range in [
            ("spring", (3, 4, 5)),
            ("summer", (6, 7, 8)),
            ("autumn", (9, 10, 11)),
            ("winter", (12, 1, 2)),
        ]:
            months = list(month_range)
            n_days = 300 if season_name != "winter" else 270
            season_dates = []
            for _ in range(n_days):
                m = rng.choice(months)
                d = int(rng.integers(1, 29))
                y = int(rng.integers(2000, 2020))
                season_dates.append(pd.Timestamp(year=y, month=m, day=d))
            df = pd.DataFrame({"date": pd.DatetimeIndex(season_dates).sort_values()})
            df["year"] = df["date"].dt.year
            df["peak_hour"] = rng.uniform(10, 16, len(df))
            df["season"] = season_name
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def test_returns_dict_with_four_seasons(self) -> None:
        """Returns a dict with all four season keys."""
        config = EpinalPeakConfig(n_bootstrap=20)
        df = self._seasonal_synthetic_data()
        result = analysis.seasonal_stratification(df, config)
        assert isinstance(result, dict)
        for s in ("spring", "summer", "autumn", "winter"):
            assert s in result, f"Missing season: {s}"

    def test_each_season_has_status(self) -> None:
        """Each season entry contains a status key."""
        config = EpinalPeakConfig(n_bootstrap=20)
        df = self._seasonal_synthetic_data()
        result = analysis.seasonal_stratification(df, config)
        for s in ("spring", "summer", "autumn", "winter"):
            assert "status" in result[s], f"{s} missing status"

    def test_empty_dataframe(self) -> None:
        """Empty df returns all seasons as no_data."""
        config = EpinalPeakConfig()
        df = pd.DataFrame(columns=["year", "peak_hour", "season", "date"])
        result = analysis.seasonal_stratification(df, config)
        for s in ("spring", "summer", "autumn", "winter"):
            assert result[s].get("status") == "no_data"

    def test_no_season_column_raises(self) -> None:
        """DataFrame without season column raises KeyError."""
        config = EpinalPeakConfig()
        df = pd.DataFrame({"year": [2000], "peak_hour": [14.0]})
        with pytest.raises(KeyError):
            analysis.seasonal_stratification(df, config)


class TestSeasonalSensitivity:
    """Tests for ``seasonal_sensitivity_analysis()``."""

    def test_returns_nested_dict(self) -> None:
        """Returns outer dict of seasons, each with variant dicts."""
        config = EpinalPeakConfig(n_bootstrap=10)
        dates = pd.date_range("2000-01-01", periods=400, freq="D")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "date": dates,
            "year": dates.year,
            "peak_hour": rng.uniform(10, 16, 400),
            "diurnal_amplitude": rng.uniform(2.0, 8.0, 400),
            "season": ["spring"] * 100 + ["summer"] * 100 + ["autumn"] * 100 + ["winter"] * 100,
        })
        df["peak_hour_sensitivity"] = df["peak_hour"]
        result = analysis.seasonal_sensitivity_analysis(df, config)
        assert isinstance(result, dict)
        for s in ("spring", "summer", "autumn", "winter"):
            assert s in result, f"Missing season: {s}"

    def test_variant_keys_present_when_data_exists(self) -> None:
        """Each season with data has expected variant keys."""
        config = EpinalPeakConfig(n_bootstrap=10)
        dates = pd.date_range("2000-01-01", periods=400, freq="D")
        rng = np.random.default_rng(123)
        df = pd.DataFrame({
            "date": dates,
            "year": dates.year,
            "peak_hour": rng.uniform(10, 16, 400),
            "diurnal_amplitude": rng.uniform(2.0, 8.0, 400),
            "season": ["spring"] * 100 + ["summer"] * 100 + ["autumn"] * 100 + ["winter"] * 100,
        })
        df["peak_hour_sensitivity"] = df["peak_hour"]
        result = analysis.seasonal_sensitivity_analysis(df, config)
        for s in ("spring", "summer", "autumn", "winter"):
            season_result = result[s]
            if isinstance(season_result, dict) and "status" not in season_result:
                for v in ("tie_rule_latest",):
                    assert v in season_result, f"{s} missing variant {v}"

    def test_empty_dataframe(self) -> None:
        """Empty df returns all seasons as no_data."""
        config = EpinalPeakConfig()
        df = pd.DataFrame(columns=["year", "peak_hour", "season", "date", "diurnal_amplitude"])
        result = analysis.seasonal_sensitivity_analysis(df, config)
        for s in ("spring", "summer", "autumn", "winter"):
            assert s in result


class TestSeasonalSupplementary:
    """Tests for per-season supplementary analysis wrappers."""

    def _four_season_df(self) -> pd.DataFrame:
        """Synthetic data with one season per quarter."""
        dates = pd.date_range("2000-01-01", periods=400, freq="D")
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "date": dates,
            "year": dates.year,
            "peak_hour": rng.uniform(10, 16, 400),
            "diurnal_amplitude": rng.uniform(2.0, 10.0, 400),
            "season": ["spring"] * 100 + ["summer"] * 100
                       + ["autumn"] * 100 + ["winter"] * 100,
        })

    def test_seasonal_temperature_weighted_returns_all_seasons(self) -> None:
        """_seasonal_temperature_weighted returns 4 seasons + pooled."""
        config = EpinalPeakConfig(n_bootstrap=10)
        df = self._four_season_df()
        result = analysis._seasonal_temperature_weighted(df, config)
        for s in ("spring", "summer", "autumn", "winter", "pooled"):
            assert s in result, f"Missing key: {s}"

    def test_seasonal_temperature_weighted_each_has_status(self) -> None:
        """Each season entry has a status key."""
        config = EpinalPeakConfig(n_bootstrap=10)
        df = self._four_season_df()
        result = analysis._seasonal_temperature_weighted(df, config)
        for s in ("spring", "summer", "autumn", "winter"):
            assert "status" in result[s], f"{s} missing status"

    @pytest.mark.slow
    def test_seasonal_correlation_returns_all_seasons(self) -> None:
        """_seasonal_circular_linear_correlation returns 4 seasons + pooled."""
        config = EpinalPeakConfig(n_bootstrap_fallback=50)
        df = self._four_season_df()
        result = analysis._seasonal_circular_linear_correlation(df, config)
        for s in ("spring", "summer", "autumn", "winter", "pooled"):
            assert s in result, f"Missing key: {s}"

    @pytest.mark.slow
    def test_seasonal_correlation_each_has_rho_c(self) -> None:
        """Each season entry has a rho_c or status key."""
        config = EpinalPeakConfig(n_bootstrap_fallback=50)
        df = self._four_season_df()
        result = analysis._seasonal_circular_linear_correlation(df, config)
        for s in ("spring", "summer", "autumn", "winter"):
            entry = result[s]
            assert "rho_c" in entry or "status" in entry, f"{s} missing both"


class TestAutocorrelationDiagnostic:
    """Tests for _seasonal_autocorrelation()."""

    def test_returns_all_seasons_plus_pooled(self) -> None:
        """Returns dict with spring/summer/autumn/winter + pooled."""
        dates = pd.date_range("2000-01-01", periods=400, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "season": ["spring"] * 100 + ["summer"] * 100
                      + ["autumn"] * 100 + ["winter"] * 100,
            "peak_hour": np.random.default_rng(42).uniform(10, 16, 400),
        })
        result = analysis._seasonal_autocorrelation(df)
        for s in ("spring", "summer", "autumn", "winter", "pooled"):
            assert s in result

    def test_acf_has_seven_lags(self) -> None:
        """Each season's ACF list has exactly 7 values."""
        dates = pd.date_range("2000-01-01", periods=400, freq="D")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "date": dates,
            "season": ["spring"] * 100 + ["summer"] * 100
                      + ["autumn"] * 100 + ["winter"] * 100,
            "peak_hour": rng.uniform(10, 16, 400),
        })
        result = analysis._seasonal_autocorrelation(df)
        for s in ("spring", "summer", "autumn", "winter", "pooled"):
            acf = result[s]
            assert acf is not None, f"{s} returned None"
            assert len(acf) == 7, f"{s} has {len(acf)} lags, expected 7"

    def test_white_noise_acf_near_zero(self) -> None:
        """Synthetic white noise yields ACF values near zero."""
        dates = pd.date_range("2000-01-01", periods=1000, freq="D")
        rng = np.random.default_rng(123)
        df = pd.DataFrame({
            "date": dates,
            "season": "spring",
            "peak_hour": rng.uniform(10, 16, 1000),
        })
        result = analysis._seasonal_autocorrelation(df)
        acf = result.get("spring", result.get("pooled"))
        assert acf is not None
        for val in acf:
            assert abs(val) < 0.15, f"ACF value {val} too far from zero for white noise"
