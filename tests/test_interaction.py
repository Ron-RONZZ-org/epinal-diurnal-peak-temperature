"""Tests for the year × season interaction test module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epinal_peak._analysis_interaction import (
    _build_design_matrix,
    _neg_log_likelihood_design,
    _numerical_hessian,
    _init_params,
    _fit_vm_design,
    interaction_lr_test,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_interaction_data() -> pd.DataFrame:
    """Generate 10 years of daily data with a strong year × season interaction.

    Spring/summer have a positive trend (+0.5 h/decade), autumn/winter have
    a negative trend (-0.5 h/decade).  The interaction should be clearly
    detectable with n = 10 × 365 = 3650 observations.
    """
    rng = np.random.default_rng(2024)
    n_years = 10
    days_per_year = 365
    total = n_years * days_per_year

    dates = pd.date_range("2010-01-01", periods=total, freq="D")
    years = dates.year.values.astype(float)
    year_center = float(np.mean(years))
    months = dates.month.values

    # Map month to season
    def _month_to_season(m: int) -> str:
        if 3 <= m <= 5:
            return "spring"
        if 6 <= m <= 8:
            return "summer"
        if 9 <= m <= 11:
            return "autumn"
        return "winter"

    seasons = np.array([_month_to_season(m) for m in months])

    # True slopes (hours/year) for each season
    # Spring: +0.05 h/yr = +0.5 h/decade
    # Summer: +0.05 h/yr
    # Autumn: -0.05 h/yr
    # Winter: -0.05 h/yr
    beta_season = {
        "spring": 0.05,
        "summer": 0.05,
        "autumn": -0.05,
        "winter": -0.05,
    }

    # True intercepts (hours) per season
    alpha_season = {
        "spring": 14.0,
        "summer": 15.0,
        "autumn": 13.0,
        "winter": 12.0,
    }

    kappa = 8.0
    hours = np.zeros(total)
    for season_name, slope in beta_season.items():
        mask = seasons == season_name
        n_mask = mask.sum()
        yr = years[mask]
        mu_hours = alpha_season[season_name] + slope * (yr - year_center)
        mu_rad = mu_hours * 2.0 * np.pi / 24.0
        # Generate von Mises noise
        from scipy.stats import vonmises

        noise = vonmises.rvs(kappa, size=n_mask, random_state=rng)
        theta = (mu_rad + noise) % (2.0 * np.pi)
        hours[mask] = theta * 24.0 / (2.0 * np.pi)

    return pd.DataFrame({
        "date": dates,
        "year": years,
        "peak_hour": hours,
        "season": seasons,
    })


@pytest.fixture
def synthetic_no_interaction_data() -> pd.DataFrame:
    """10 years of daily data with a common slope across seasons.

    All seasons have β = +0.03 h/yr = +0.3 h/decade (same slope).
    The interaction test should NOT be significant.
    """
    rng = np.random.default_rng(2025)
    n_years = 10
    days_per_year = 365
    total = n_years * days_per_year

    dates = pd.date_range("2010-01-01", periods=total, freq="D")
    years = dates.year.values.astype(float)
    year_center = float(np.mean(years))
    months = dates.month.values

    def _month_to_season(m: int) -> str:
        if 3 <= m <= 5:
            return "spring"
        if 6 <= m <= 8:
            return "summer"
        if 9 <= m <= 11:
            return "autumn"
        return "winter"

    seasons = np.array([_month_to_season(m) for m in months])

    common_slope = 0.03  # h/yr
    alpha_season = {"spring": 14.0, "summer": 15.0, "autumn": 13.0, "winter": 12.0}
    kappa = 10.0

    hours = np.zeros(total)
    for season_name, alpha in alpha_season.items():
        mask = seasons == season_name
        n_mask = mask.sum()
        yr = years[mask]
        mu_hours = alpha + common_slope * (yr - year_center)
        mu_rad = mu_hours * 2.0 * np.pi / 24.0
        from scipy.stats import vonmises

        noise = vonmises.rvs(kappa, size=n_mask, random_state=rng)
        theta = (mu_rad + noise) % (2.0 * np.pi)
        hours[mask] = theta * 24.0 / (2.0 * np.pi)

    return pd.DataFrame({
        "date": dates,
        "year": years,
        "peak_hour": hours,
        "season": seasons,
    })


# ── Tests for _build_design_matrix ───────────────────────────────────


class TestBuildDesignMatrix:
    def test_reduced_shape(self) -> None:
        """Reduced (additive) model: 5 columns."""
        season = pd.Series(["spring", "summer", "autumn", "winter"])
        year_c = np.array([-1.0, 0.0, 1.0, 2.0])
        X = _build_design_matrix(season, year_c, include_interaction=False)
        assert X.shape == (4, 5)

    def test_full_shape(self) -> None:
        """Full (interaction) model: 8 columns."""
        season = pd.Series(["spring", "summer", "autumn", "winter"])
        year_c = np.array([-1.0, 0.0, 1.0, 2.0])
        X = _build_design_matrix(season, year_c, include_interaction=True)
        assert X.shape == (4, 8)

    def test_one_hot_correct(self) -> None:
        """Each row has exactly one ``1`` in the first 4 columns."""
        season = pd.Series(["spring", "summer", "autumn", "winter", "spring"])
        year_c = np.zeros(5)
        X = _build_design_matrix(season, year_c, include_interaction=False)
        row_sums = X[:, :4].sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_interaction_columns(self) -> None:
        """Interaction columns are season_dummy × year_centered."""
        season = pd.Series(["spring", "summer"])
        year_c = np.array([2.0, 3.0])
        X_full = _build_design_matrix(season, year_c, include_interaction=True)
        # Column 4 (spring×yr) should be [2, 0]
        assert X_full[0, 4] == pytest.approx(2.0)
        assert X_full[1, 4] == pytest.approx(0.0)
        # Column 5 (summer×yr) should be [0, 3]
        assert X_full[0, 5] == pytest.approx(0.0)
        assert X_full[1, 5] == pytest.approx(3.0)


# ── Tests for _neg_log_likelihood_design ────────────────────────────


class TestNegLogLikelihoodDesign:
    def test_basic_call(self) -> None:
        """Simple call with a known design matrix returns a float."""
        theta = np.array([0.0, 1.0])
        X = np.eye(2)
        params = np.array([0.0, 0.0, 1.0])  # 2 betas + log_kappa
        nll = _neg_log_likelihood_design(params, theta, X)
        assert isinstance(nll, float)
        assert np.isfinite(nll)

    def test_higher_kappa_reduces_nll(self) -> None:
        """For perfectly fitting mu, higher kappa increases likelihood
        (decreases NLL)."""
        theta = np.array([1.0, 1.0])
        X = np.ones((2, 1))
        nll_low = _neg_log_likelihood_design(np.array([1.0, 0.0]), theta, X)
        nll_high = _neg_log_likelihood_design(np.array([1.0, 5.0]), theta, X)
        assert nll_high < nll_low


# ── Tests for _numerical_hessian ────────────────────────────────────


class TestNumericalHessian:
    def test_quadratic(self) -> None:
        """Hessian of f(x, y) = x² + 3y² should be diag(2, 6)."""

        def f(x: np.ndarray) -> float:
            return float(x[0] ** 2 + 3.0 * x[1] ** 2)

        H = _numerical_hessian(f, np.array([1.0, 2.0]), eps=1e-4)
        assert H[0, 0] == pytest.approx(2.0, abs=1e-3)
        assert H[1, 1] == pytest.approx(6.0, abs=1e-3)
        assert H[0, 1] == pytest.approx(0.0, abs=1e-3)

    def test_symmetric(self) -> None:
        """Hessian is symmetric."""

        def f(x: np.ndarray) -> float:
            return float(x[0] * x[1] + x[0] ** 2)

        H = _numerical_hessian(f, np.array([1.0, 2.0]), eps=1e-4)
        assert H[0, 1] == pytest.approx(H[1, 0], abs=1e-10)


# ── Tests for _init_params ──────────────────────────────────────────


class TestInitParams:
    def test_reduced_length(self) -> None:
        """Reduced initial vector has length 6."""
        theta = np.array([0.0, 1.0, 2.0, 3.0])
        yr = np.array([0.0, 0.0, 1.0, 1.0])
        season = pd.Series(["spring", "summer", "autumn", "winter"])
        inits = _init_params(theta, yr, season)
        assert len(inits["reduced"]) == 6

    def test_full_length(self) -> None:
        """Full initial vector has length 9."""
        theta = np.array([0.0, 1.0, 2.0, 3.0])
        yr = np.array([0.0, 0.0, 1.0, 1.0])
        season = pd.Series(["spring", "summer", "autumn", "winter"])
        inits = _init_params(theta, yr, season)
        assert len(inits["full"]) == 9


# ── Tests for interaction_lr_test ───────────────────────────────────


class TestInteractionLRTest:
    def test_returns_dict(self) -> None:
        """Returns a dict with required keys."""
        df = pd.DataFrame({
            "year": [2000, 2001],
            "peak_hour": [14.0, 14.1],
            "season": ["spring", "summer"],
        })
        result = interaction_lr_test(df)
        assert isinstance(result, dict)
        assert "status" in result

    def test_missing_column(self) -> None:
        """Missing season column returns error status."""
        df = pd.DataFrame({"year": [2000], "peak_hour": [14.0]})
        result = interaction_lr_test(df)
        assert "missing" in result["status"]

    def test_detect_interaction(self, synthetic_interaction_data: pd.DataFrame) -> None:
        """Synthetic data with strong interaction returns significant result."""
        result = interaction_lr_test(synthetic_interaction_data)
        assert result["status"] == "ok"
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert "per_season" in result
        assert result["per_season"]["n_seasons_rejected"] >= 2

    def test_no_interaction(self, synthetic_no_interaction_data: pd.DataFrame) -> None:
        """Synthetic data with common slope returns non-significant result."""
        result = interaction_lr_test(synthetic_no_interaction_data)
        assert result["status"] == "ok"
        # With common slope, should NOT be significant
        assert result["significant"] is False

    def test_per_season_slopes_sign(self, synthetic_interaction_data: pd.DataFrame) -> None:
        """Signs of per-season slopes match the known true slopes.

        Spring/summer: positive; autumn/winter: negative.
        """
        result = interaction_lr_test(synthetic_interaction_data)
        assert result["status"] == "ok"
        slopes = {s["season"]: s["slope_hours_per_decade"] for s in result["per_season"]["slopes"]}
        assert slopes.get("spring", 0) > 0
        assert slopes.get("summer", 0) > 0
        assert slopes.get("autumn", 0) < 0
        assert slopes.get("winter", 0) < 0

    def test_returns_lr_stat_and_df(self) -> None:
        """Result always contains lr_stat and df fields."""
        df = pd.DataFrame({
            "year": [2000, 2001, 2002, 2003],
            "peak_hour": [14.0, 14.1, 13.9, 14.0],
            "season": ["spring", "summer", "autumn", "winter"],
        })
        result = interaction_lr_test(df)
        if result["status"] == "ok":
            assert isinstance(result["lr_stat"], float)
            assert result["df"] == 3


# ── Tests for _fit_vm_design ────────────────────────────────────────


class TestFitVmDesign:
    def test_converges_on_simple_data(self) -> None:
        """Fitting converges on well-behaved data."""
        theta = np.array([0.0, 0.1, -0.1])
        X = np.ones((3, 1))
        x0 = np.array([0.0, 1.0])
        result = _fit_vm_design(theta, X, x0)
        assert result["status"] == "ok"
        assert result["converged"] is True
