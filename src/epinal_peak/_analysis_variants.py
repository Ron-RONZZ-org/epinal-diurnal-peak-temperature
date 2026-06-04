"""Sensitivity variants, seasonal stratification, and supplementary regressions."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import i0

from epinal_peak._analysis_bootstrap import _bootstrap_engine

_logger = logging.getLogger(__name__)

# ── Local conversion helpers (avoid circular import with analysis.py) ──

_HOURS_TO_RAD = 2.0 * np.pi / 24.0
_RAD_TO_HOURS = 24.0 / (2.0 * np.pi)


def _hours_to_radians(hours: np.ndarray) -> np.ndarray:
    """Convert hour-of-day values to radians."""
    return hours * _HOURS_TO_RAD


def _radians_to_hours(radians: np.ndarray | float) -> np.ndarray | float:
    """Convert radians to hour-of-day values, normalised to [0, 24)."""
    return (radians * _RAD_TO_HOURS) % 24.0


def _apply_subsampling(hours: np.ndarray, bin_size: int) -> np.ndarray:
    """Round peak hours to the nearest N-hour bin.

    Args:
        hours: Peak hour values (0--23).
        bin_size: Bin width in hours (e.g., 3 or 6).

    Returns:
        Rounded hours in [0, 24).
    """
    half = bin_size / 2.0
    return ((hours + half) // bin_size * bin_size) % 24.0


def _run_variant(
    df: pd.DataFrame,
    overrides: dict[str, Any],
    n_iter: int = 1000,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Run von Mises regression with config overrides for a sensitivity variant.

    Args:
        df: Full peak-hour DataFrame.
        overrides: Dict of modifications (see note below).
        n_iter: Bootstrap iterations.
        ci_level: Bootstrap CI level.

    Supported overrides:

    - ``tie_col``: Column name for peak hour (default ``"peak_hour"``).
    - ``min_years``: Minimum unique years required (default 30).
    - ``exclude_post_2000``: If True, keep only rows with ``year <= 2000``.
    - ``amplitude_threshold``: Minimum diurnal amplitude filter value.
    - ``subsampling_bin``: Bin size in hours for subsampling.
    - ``note``: Human-readable label for the variant.

    Returns:
        Dict with regression result + bootstrap + variant metadata.
    """
    variant_df = df.copy()

    peak_col = overrides.get("tie_col", "peak_hour")
    min_years = overrides.get("min_years", 30)
    exclude_post_2000 = overrides.get("exclude_post_2000", False)
    amplitude_threshold = overrides.get("amplitude_threshold", None)

    # ── Apply column override ──────────────────────────────────────
    if peak_col != "peak_hour":
        if peak_col not in variant_df.columns:
            _logger.warning(
                "Column '%s' not found, falling back to 'peak_hour'", peak_col
            )
        else:
            variant_df["peak_hour"] = variant_df[peak_col]

    # ── Apply amplitude filter ─────────────────────────────────────
    if amplitude_threshold is not None and "diurnal_amplitude" in variant_df.columns:
        variant_df = variant_df[
            variant_df["diurnal_amplitude"] >= amplitude_threshold
        ].copy()

    # ── Apply post-2000 exclusion ───────────────────────────────────
    if exclude_post_2000:
        variant_df = variant_df[variant_df["year"] <= 2000].copy()

    # ── Apply subsampling ───────────────────────────────────────────
    subsampling_bin = overrides.get("subsampling_bin")
    if subsampling_bin is not None:
        variant_df["peak_hour"] = _apply_subsampling(
            variant_df["peak_hour"].values, subsampling_bin
        )

    # ── Check min years ─────────────────────────────────────────────
    unique_years = variant_df["year"].nunique() if "year" in variant_df.columns else 0
    if unique_years < min_years:
        return {
            "status": "insufficient_unique_years",
            "n_years": int(unique_years),
            "min_years_required": min_years,
            "variant_note": overrides.get("note", ""),
        }

    # ── Run regression (lazy import to avoid circular dep) ──────────
    from epinal_peak.analysis import von_mises_regression as _vm_reg

    try:
        result = _vm_reg(variant_df)
    except (ValueError, RuntimeError) as exc:
        return {
            "status": "error",
            "error": str(exc),
            "variant_note": overrides.get("note", ""),
        }

    # ── Bootstrap ───────────────────────────────────────────────────
    bootstrap = _bootstrap_engine(
        variant_df, n_iter=n_iter, ci_level=ci_level
    )
    result["bootstrap"] = bootstrap
    result["variant_note"] = overrides.get("note", "")
    result["n_years"] = int(unique_years)

    return result


def sensitivity_analysis(
    df: pd.DataFrame,
    n_iter: int = 1000,
    ci_level: float = 0.95,
    amplitude_thresholds: tuple[float, ...] | None = None,
    subsampling_bins: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Run all pre-registered sensitivity variants.

    Args:
        df: Daily peak-hour DataFrame.
        n_iter: Bootstrap iterations per variant.
        ci_level: Bootstrap CI level.
        amplitude_thresholds: Alternative amplitude thresholds to test.
        subsampling_bins: Bin sizes in hours for subsampling variants.

    Returns:
        Dict mapping variant names to result dicts.
    """
    at = amplitude_thresholds or (1.0, 3.0)
    sb = subsampling_bins or (3, 6)

    variants: dict[str, Any] = {}

    # ── Tie rule: latest ────────────────────────────────────────────
    variants["tie_rule_latest"] = _run_variant(
        df,
        {"tie_col": "peak_hour_sensitivity", "note": "tie_rule=latest"},
        n_iter=n_iter,
        ci_level=ci_level,
    )

    # ── Min years: 20 ───────────────────────────────────────────────
    variants["min_years_20"] = _run_variant(
        df,
        {"min_years": 20, "note": "min_years=20"},
        n_iter=n_iter,
        ci_level=ci_level,
    )

    # ── Subsampling variants ────────────────────────────────────────
    for bin_size in sb:
        variants[f"subsampling_{bin_size}hr"] = _run_variant(
            df,
            {"subsampling_bin": bin_size, "note": f"subsampling_bin={bin_size}"},
            n_iter=n_iter,
            ci_level=ci_level,
        )

    # ── Amplitude threshold variants ────────────────────────────────
    for threshold in at:
        variants[f"amplitude_threshold_{threshold}"] = _run_variant(
            df,
            {"amplitude_threshold": threshold, "note": f"min_amplitude={threshold}"},
            n_iter=n_iter,
            ci_level=ci_level,
        )

    return variants


def seasonal_stratification(
    df: pd.DataFrame,
    n_iter: int = 1000,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Run primary analysis per meteorological season (exploratory).

    Splits data by the ``season`` column (spring / summer / autumn /
    winter), runs von Mises regression + bootstrap on each subset.

    Args:
        df: DataFrame with ``season`` column.
        n_iter: Bootstrap iterations per season.
        ci_level: Bootstrap CI level.

    Returns:
        Dict with keys ``spring``, ``summer``, ``autumn``, ``winter``.
    """
    seasons = {"spring": "MAM", "summer": "JJA", "autumn": "SON", "winter": "DJF"}
    results: dict[str, Any] = {}

    for season_name, season_label in seasons.items():
        subset = df[df["season"] == season_name].copy()
        if subset.empty:
            results[season_name] = {
                "status": "no_data",
                "label": season_label,
                "n_obs": 0,
                "n_years": 0,
            }
            continue

        from epinal_peak.analysis import von_mises_regression as _vm_reg

        result = _vm_reg(subset)
        bootstrap = _bootstrap_engine(subset, n_iter=n_iter, ci_level=ci_level)
        result["bootstrap"] = bootstrap
        result["label"] = season_label
        result["n_years"] = int(subset["year"].nunique())
        results[season_name] = result

    return results


def _weighted_neg_log_likelihood(
    params: np.ndarray,
    theta: np.ndarray,
    year_centered: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted negative log-likelihood for von Mises regression.

    Each observation contributes ``w_i`` times the log-likelihood.
    The effective sample size is ``sum(w_i)``.
    """
    beta_0, beta_1, log_kappa = params
    kappa = np.exp(log_kappa)
    mu = beta_0 + beta_1 * year_centered
    n_eff = weights.sum()
    ll = np.sum(weights * kappa * np.cos(theta - mu)) - n_eff * np.log(
        2.0 * np.pi * i0(kappa)
    )
    return float(-ll)


def temperature_weighted_regression(
    df: pd.DataFrame,
    n_iter: int = 1000,
    ci_level: float = 0.95,
    weight_col: str = "diurnal_amplitude",
) -> dict[str, Any]:
    """Von Mises regression weighted by diurnal amplitude (supplementary).

    Days with larger diurnal ranges have more clearly defined peaks and
    receive more weight in the log-likelihood.

    Args:
        df: DataFrame with ``peak_hour``, ``year``, and *weight_col*.
        n_iter: Bootstrap iterations.
        ci_level: Bootstrap CI level.
        weight_col: Column name for observation weights.

    Returns:
        Dict with regression coefficients, bootstrap CI, and weight summary.
    """
    hours = df["peak_hour"].values.astype(float)
    year = df["year"].values.astype(float)
    weights = df[weight_col].values.astype(float)

    valid = ~(np.isnan(hours) | np.isnan(year) | np.isnan(weights) | (weights <= 0))
    if not valid.any():
        return {"status": "no_valid_observations"}

    hours = hours[valid]
    year = year[valid]
    weights = weights[valid]

    theta = _hours_to_radians(hours)
    year_center = float(np.mean(year))
    year_centered = year - year_center

    # ── Initial guesses ─────────────────────────────────────────────
    sin_sum = np.sin(theta).sum()
    cos_sum = np.cos(theta).sum()
    beta_0_init = np.arctan2(sin_sum, cos_sum)
    n = float(len(theta))
    r = np.sqrt(sin_sum**2 + cos_sum**2) / n
    if r < 0.99:
        kappa_init = max(r * (2.0 - r**2) / (1.0 - r**2), 0.01)
    else:
        kappa_init = 100.0

    opt_result = minimize(
        _weighted_neg_log_likelihood,
        x0=np.array([beta_0_init, 0.0, np.log(kappa_init)]),
        args=(theta, year_centered, weights),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )

    beta_0, beta_1, log_kappa = opt_result.x
    kappa = np.exp(log_kappa)
    beta_1_hpd = beta_1 * _radians_to_hours(np.array([1.0]))[0] * 10.0

    # ── Bootstrap (resample with weights in df) ─────────────────────
    # Build a DataFrame subset that includes weights
    weighted_df = df.iloc[valid].copy()
    weighted_df["_weight"] = weights
    bootstrap_result = _bootstrap_weighted_engine(
        weighted_df, n_iter=n_iter, ci_level=ci_level
    )

    return {
        "status": "ok" if opt_result.success else "mle_did_not_converge",
        "beta_0": float(beta_0),
        "beta_1": float(beta_1),
        "beta_1_hours_per_decade": float(beta_1_hpd),
        "kappa": float(kappa),
        "year_center": year_center,
        "n_obs": int(len(theta)),
        "n_years": int(year_centered.size),  # approximate
        "converged": bool(opt_result.success),
        "log_likelihood": float(-opt_result.fun),
        "bootstrap": bootstrap_result,
        "weight_var": weight_col,
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
    }


def _bootstrap_weighted_engine(
    df: pd.DataFrame,
    n_iter: int = 1000,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap resampling for the temperature-weighted regression.

    Resamples rows and extracts the weighted regression's slope.
    """
    alpha = 1.0 - ci_level
    lower_pct = 100.0 * alpha / 2.0
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    estimates: list[float] = []
    n = len(df)

    for i in range(n_iter):
        if (i + 1) % 100 == 0:
            _logger.info("Weighted bootstrap iteration %d / %d", i + 1, n_iter)

        resample = df.sample(n=n, replace=True, random_state=i)
        try:
            result = _run_weighted(resample)
        except (ValueError, RuntimeError):
            continue

        if result.get("status") != "ok":
            continue
        estimates.append(result["beta_1_hours_per_decade"])

    if len(estimates) < 2:
        return {
            "n_iter": n_iter,
            "ci_level": ci_level,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "median": np.nan,
            "std_error": np.nan,
            "status": "too_few_valid_resamples",
        }

    arr = np.array(estimates)
    return {
        "n_iter": len(estimates),
        "ci_level": ci_level,
        "ci_lower": float(np.nanpercentile(arr, lower_pct)),
        "ci_upper": float(np.nanpercentile(arr, upper_pct)),
        "median": float(np.nanmedian(arr)),
        "std_error": float(np.nanstd(arr, ddof=1)),
        "status": "ok",
    }


def _run_weighted(df: pd.DataFrame) -> dict[str, Any]:
    """Fit weighted von Mises regression on a DataFrame with ``_weight`` column."""
    hours = df["peak_hour"].values.astype(float)
    year = df["year"].values.astype(float)
    weights = df["_weight"].values.astype(float)

    valid = ~(np.isnan(hours) | np.isnan(year) | np.isnan(weights) | (weights <= 0))
    if not valid.any():
        return {"status": "no_valid_observations"}

    hours = hours[valid]
    year = year[valid]
    weights = weights[valid]

    theta = _hours_to_radians(hours)
    year_center = float(np.mean(year))
    year_centered = year - year_center

    sin_sum = np.sin(theta).sum()
    cos_sum = np.cos(theta).sum()
    beta_0_init = np.arctan2(sin_sum, cos_sum)
    n = float(len(theta))
    r = np.sqrt(sin_sum**2 + cos_sum**2) / n
    if r < 0.99:
        kappa_init = max(r * (2.0 - r**2) / (1.0 - r**2), 0.01)
    else:
        kappa_init = 100.0

    opt_result = minimize(
        _weighted_neg_log_likelihood,
        x0=np.array([beta_0_init, 0.0, np.log(kappa_init)]),
        args=(theta, year_centered, weights),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )

    beta_0, beta_1, log_kappa = opt_result.x
    kappa = np.exp(log_kappa)
    beta_1_hpd = beta_1 * _radians_to_hours(np.array([1.0]))[0] * 10.0

    return {
        "status": "ok" if opt_result.success else "mle_did_not_converge",
        "beta_0": float(beta_0),
        "beta_1": float(beta_1),
        "beta_1_hours_per_decade": float(beta_1_hpd),
        "kappa": float(kappa),
        "year_center": year_center,
        "n_obs": int(len(theta)),
        "converged": bool(opt_result.success),
        "log_likelihood": float(-opt_result.fun),
    }
