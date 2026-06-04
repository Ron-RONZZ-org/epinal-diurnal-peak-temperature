"""Circular statistics and von Mises regression for peak-hour trends.

Implements circular mean, circular standard deviation, and a von Mises GLM
with year as a linear predictor.  Bootstrap confidence intervals and
sensitivity analyses are also provided.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import i0

import epinal_peak
from epinal_peak._analysis_bootstrap import _bootstrap_engine
from epinal_peak._analysis_corr import circular_linear_correlation, _mardia_correlation
from epinal_peak import _analysis_variants as _variants
from epinal_peak.config import EpinalPeakConfig

_logger = logging.getLogger(__name__)

_HOURS_TO_RAD = 2.0 * np.pi / 24.0
_RAD_TO_HOURS = 24.0 / (2.0 * np.pi)


def _hours_to_radians(hours: np.ndarray) -> np.ndarray:
    """Convert hour-of-day values to radians."""
    return hours * _HOURS_TO_RAD


def _radians_to_hours(radians: np.ndarray | float) -> np.ndarray | float:
    """Convert radians to hour-of-day values, normalised to [0, 24)."""
    return (radians * _RAD_TO_HOURS) % 24.0


def circular_mean(hours: np.ndarray) -> float:
    """Compute the circular mean of hourly data (0-23 range).

    Converts hours to radians, computes the mean direction from the
    summed unit vectors, and converts back to hours.

    Args:
        hours: Array of hour-of-day values (0--23).  NaNs are ignored.

    Returns:
        Circular mean in hours (0--23).  Returns NaN if all values are
        NaN or if the resultant vector length is zero (uniform circular
        distribution).

    Examples:
        >>> circular_mean(np.array([0.0, 0.0]))
        0.0
        >>> circular_mean(np.array([23.0, 1.0]))
        0.0
    """
    hours = np.asarray(hours, dtype=float)
    valid = ~np.isnan(hours)
    if not valid.any():
        return np.nan

    radians = _hours_to_radians(hours[valid])
    sin_sum = np.sin(radians).sum()
    cos_sum = np.cos(radians).sum()
    mean_rad = np.arctan2(sin_sum, cos_sum)
    # Normalise to [0, 2π). Avoid Python's % trap on tiny negative floats
    # (e.g. -4e-16 % 2π → 2π, not 0), then convert to [0, 24) hours.
    if mean_rad < 0.0:
        mean_rad += 2.0 * np.pi
    hours_out = mean_rad * _RAD_TO_HOURS
    # Normalise to [0, 24) — catches cases where mean_rad ≈ 2π
    return float(hours_out % 24.0)


def circular_std(hours: np.ndarray) -> float:
    """Compute the circular standard deviation of hourly data (0-23 range).

    Uses the mean resultant length *R*:

    .. math::

        \\sigma = \\sqrt{-2 \\ln(R)} \\times \\frac{24}{2\\pi}

    where *R* = :math:`\\sqrt{(\\sum \\cos \\theta)^2 + (\\sum \\sin \\theta)^2} / n`.

    Args:
        hours: Array of hour-of-day values (0--23).  NaNs are ignored.

    Returns:
        Circular standard deviation in hours (0--23).  Returns NaN if all
        values are NaN or R = 0 (uniform circular distribution).

    Examples:
        >>> np.round(circular_std(np.array([0.0, 0.0])), 6)
        0.0
    """
    hours = np.asarray(hours, dtype=float)
    valid = ~np.isnan(hours)
    if not valid.any():
        return np.nan

    radians = _hours_to_radians(hours[valid])
    n = float(len(radians))
    sin_sum = np.sin(radians).sum()
    cos_sum = np.cos(radians).sum()
    r = np.sqrt(sin_sum**2 + cos_sum**2) / n

    if r <= 0.0:
        return np.nan

    # Clamp R to [eps, 1.0] to avoid floating-point edge cases:
    #   - R slightly > 1.0 → log(R) > 0 → sqrt(negative) → NaN
    #   - R slightly < 1.0 (e.g. 1 - 1e-16) →  sqrt(-2*log(R)) gives
    #     tiny non-zero (~1e-8 h) instead of 0 for all-identical data.
    if r > 1.0:
        r = 1.0
    elif 1.0 - r < 1e-12:
        r = 1.0
    circular_std_rad = np.sqrt(-2.0 * np.log(r))
    return float(circular_std_rad * _RAD_TO_HOURS)


def _neg_log_likelihood(
    params: np.ndarray, theta: np.ndarray, year_centered: np.ndarray
) -> float:
    """Negative log-likelihood of the von Mises regression model.

    Args:
        params: [beta_0, beta_1, log_kappa] where kappa = exp(log_kappa).
        theta: Circular response in radians.
        year_centered: Centered year predictor.

    Returns:
        Negative log-likelihood value.
    """
    beta_0, beta_1, log_kappa = params
    kappa = np.exp(log_kappa)
    mu = beta_0 + beta_1 * year_centered
    n = len(theta)

    # log L = sum(kappa * cos(theta - mu)) - n * log(2 * pi * I0(kappa))
    ll = np.sum(kappa * np.cos(theta - mu)) - n * np.log(2.0 * np.pi * i0(kappa))
    return float(-ll)


def von_mises_regression(df: pd.DataFrame) -> dict:
    """Fit a von Mises GLM with year as a linear predictor.

    Maximises the von Mises log-likelihood directly using
    ``scipy.optimize.minimize`` (Fisher & Lee 1992).

    The model is:

    .. math::

        \\theta_i = 2\\pi \\times \\text{peak\\_hour}_i / 24
        \\mu_i = \\beta_0 + \\beta_1 \\times \\text{year\\_centered}_i
        \\log L = \\sum_i \\kappa \\cos(\\theta_i - \\mu_i)
                  - n \\log(2\\pi I_0(\\kappa))

    where :math:`I_0` is the modified Bessel function
    (``scipy.special.i0``).  :math:`\\kappa = \\exp(\\text{log\\_kappa})`
    ensures :math:`\\kappa > 0`.

    Args:
        df: DataFrame with columns ``year`` and ``peak_hour``.

    Returns:
        Dictionary with keys:
            - ``beta_0``: Intercept (radians).
            - ``beta_1``: Slope (radians/year).
            - ``beta_1_hours_per_decade``: Slope in hours/decade.
            - ``kappa``: Estimated concentration parameter.
            - ``year_center``: Year centering value.
            - ``n_obs``: Number of observations used.
            - ``converged``: Whether the MLE optimisation converged.
            - ``log_likelihood``: Maximised log-likelihood.
            - ``status``: ``"ok"`` or ``"error"``.

    Raises:
        ValueError: If the DataFrame is empty or missing required columns.
    """
    if df.empty:
        raise ValueError("DataFrame is empty — nothing to fit.")
    for col in ("year", "peak_hour"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    peak_hour = df["peak_hour"].values.astype(float)
    year = df["year"].values.astype(float)

    valid = ~(np.isnan(peak_hour) | np.isnan(year))
    if not valid.any():
        raise ValueError("No valid observations after NaN removal.")

    peak_hour = peak_hour[valid]
    year = year[valid]

    # Check for zero variance in peak hours (all identical)
    if np.ptp(peak_hour) == 0.0:
        return {
            "beta_0": _hours_to_radians(peak_hour[0]),
            "beta_1": 0.0,
            "beta_1_hours_per_decade": 0.0,
            "kappa": np.nan,
            "year_center": float(np.mean(year)),
            "n_obs": len(peak_hour),
            "converged": True,
            "log_likelihood": np.nan,
            "status": "all_peak_hours_identical",
        }

    # Fewer than 2 unique years
    if len(np.unique(year)) < 2:
        return {
            "beta_0": np.nan,
            "beta_1": np.nan,
            "beta_1_hours_per_decade": np.nan,
            "kappa": np.nan,
            "year_center": float(year.mean()),
            "n_obs": len(peak_hour),
            "converged": False,
            "log_likelihood": np.nan,
            "status": "insufficient_unique_years",
        }

    theta = _hours_to_radians(peak_hour)
    year_center = float(np.mean(year))
    year_centered = year - year_center

    # Initial guesses
    # beta_0: circular mean of theta
    sin_sum = np.sin(theta).sum()
    cos_sum = np.cos(theta).sum()
    beta_0_init = np.arctan2(sin_sum, cos_sum)

    # beta_1: rough linear-circular correlation / year range
    beta_1_init = 0.0

    # kappa: approximate from circular std
    n = float(len(theta))
    r = np.sqrt(sin_sum**2 + cos_sum**2) / n
    # Fisher approximation: kappa ~ (R*(2 - R^2)) / (1 - R^2) for small R
    if r < 0.99:
        kappa_init = max(r * (2.0 - r**2) / (1.0 - r**2), 0.01)
    else:
        kappa_init = 100.0
    log_kappa_init = np.log(kappa_init)

    result = minimize(
        _neg_log_likelihood,
        x0=np.array([beta_0_init, beta_1_init, log_kappa_init]),
        args=(theta, year_centered),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )

    beta_0, beta_1, log_kappa = result.x
    kappa = np.exp(log_kappa)

    # Slope in hours per decade
    beta_1_hours_per_decade = beta_1 * _RAD_TO_HOURS * 10.0

    return {
        "beta_0": float(beta_0),
        "beta_1": float(beta_1),
        "beta_1_hours_per_decade": float(beta_1_hours_per_decade),
        "kappa": float(kappa),
        "year_center": year_center,
        "n_obs": int(len(theta)),
        "converged": bool(result.success),
        "log_likelihood": float(-result.fun),
        "status": "ok" if result.success else "mle_did_not_converge",
    }


def bootstrap_trend(
    df: pd.DataFrame, n_iter: int = 1000, ci_level: float = 0.95
) -> tuple[float, float, float]:
    """Bootstrap the trend coefficient and return percentile CIs.

    Resamples daily observations with replacement, refits the von Mises
    regression each iteration, and extracts percentile confidence
    intervals for ``beta_1_hours_per_decade``.

    Args:
        df: DataFrame with columns ``year`` and ``peak_hour``.
        n_iter: Number of bootstrap resamples.
        ci_level: Confidence level for percentile intervals.

    Returns:
        Tuple ``(lower_ci, median_coefficient, upper_ci)`` in hours/decade.
    """
    result = _bootstrap_engine(df, n_iter=n_iter, ci_level=ci_level)
    return (result["ci_lower"], result["median"], result["ci_upper"])


def sensitivity_analysis(df: pd.DataFrame, config: EpinalPeakConfig) -> dict:
    """Run all pre-registered sensitivity variants of the trend analysis.

    Delegates to :mod:`_analysis_variants` for the actual computation.
    The 8 variants are:

    1. ``tie_rule_latest`` — use latest-peak tie-breaking
    2. ``min_years_20`` — reduce minimum years to 20
    3. ``subsampling_3hr`` — round peak hours to 3-hour bins
    4. ``subsampling_6hr`` — round peak hours to 6-hour bins
    5. ``amplitude_threshold_1`` — min diurnal amplitude 1.0 °C
    6. ``amplitude_threshold_3`` — min diurnal amplitude 3.0 °C

    .. note::

        The ``mad_threshold_3`` and ``exclude_post_2000`` variants
        are defined in the OSF pre-registration but require data
        columns not available in the processed CSV (``circular_outlier_flag``)
        or are time-range filters that are applied upstream.

    Args:
        df: DataFrame with daily peak hour data.
        config: Pipeline configuration with sensitivity settings.

    Returns:
        Dictionary mapping sensitivity variant names to result dicts.
    """
    _logger.info("Running sensitivity analysis — %d variants", 8)
    return _variants.sensitivity_analysis(
        df,
        n_iter=config.n_bootstrap,
        ci_level=config.bootstrap_ci_level,
        amplitude_thresholds=tuple(config.amplitude_thresholds),
        subsampling_bins=tuple(config.subsampling_bins),
    )


def seasonal_stratification(
    df: pd.DataFrame, config: EpinalPeakConfig
) -> dict[str, Any]:
    """Run primary analysis per meteorological season (exploratory).

    Splits data by ``season`` column, runs von Mises regression + bootstrap
    on each of spring / summer / autumn / winter.

    Args:
        df: DataFrame with ``season`` column.
        config: Pipeline configuration.

    Returns:
        Dict with keys ``spring``, ``summer``, ``autumn``, ``winter``.
    """
    return _variants.seasonal_stratification(
        df, n_iter=config.n_bootstrap, ci_level=config.bootstrap_ci_level
    )


def temperature_weighted_regression(
    df: pd.DataFrame, config: EpinalPeakConfig
) -> dict[str, Any]:
    """Von Mises regression weighted by diurnal amplitude (supplementary).

    Days with larger diurnal ranges (``T_max - T_min``) have more clearly
    defined peaks and receive more weight in the log-likelihood.

    Args:
        df: DataFrame with ``peak_hour``, ``year``, ``diurnal_amplitude``.
        config: Pipeline configuration.

    Returns:
        Dict with regression coefficients, bootstrap CI, weight summary.
    """
    return _variants.temperature_weighted_regression(
        df,
        n_iter=config.n_bootstrap,
        ci_level=config.bootstrap_ci_level,
    )


def _season_from_month(month: int) -> str:
    """Map calendar month to meteorological season."""
    if 3 <= month <= 5:
        return "spring"
    if 6 <= month <= 8:
        return "summer"
    if 9 <= month <= 11:
        return "autumn"
    return "winter"


def _load_and_prepare_data(config: EpinalPeakConfig) -> pd.DataFrame:
    """Load processed peak-hour CSV, derive columns, apply filters.

    Args:
        config: Pipeline configuration.

    Returns:
        DataFrame with columns ``date``, ``year``, ``peak_hour``,
        ``peak_temperature``, ``season``, and ``diurnal_amplitude``.

    Raises:
        FileNotFoundError: If the processed data file does not exist.
        ValueError: If no valid observations remain after filtering.
    """
    input_path = Path(config.processed_dir) / config.peak_data_filename
    if not input_path.exists():
        raise FileNotFoundError(f"Processed data not found: {input_path}")

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].apply(_season_from_month)

    # Filter to valid, non-excluded days
    valid = df[df["qc_excluded"] == False].copy()
    _logger.info(
        "Loaded %d rows, %d valid after QC exclusion", len(df), len(valid)
    )

    if valid.empty:
        raise ValueError("No valid observations after QC exclusion.")

    # Apply diurnal amplitude filter
    amplitude_ok = valid["diurnal_amplitude"] >= config.min_diurnal_amplitude
    valid = valid[amplitude_ok].copy()
    _logger.info(
        "%d rows after diurnal amplitude filter (>= %.1f°C)",
        len(valid),
        config.min_diurnal_amplitude,
    )

    if valid.empty:
        raise ValueError("No valid observations after diurnal amplitude filter.")

    # Drop helper column
    valid.drop(columns=["month"], inplace=True)

    return valid


def _build_output_json(
    primary: dict[str, Any],
    seasonal: dict[str, Any],
    sensitivity: dict[str, Any],
    supplementary: dict[str, Any],
    fallback: dict[str, Any] | None,
    config: EpinalPeakConfig,
) -> dict[str, Any]:
    """Assemble the full nested output JSON.

    Args:
        primary: Primary von Mises regression result.
        seasonal: Seasonal stratification results.
        sensitivity: Sensitivity analysis results.
        supplementary: Supplementary analysis results.
        fallback: Mardia fallback result (or None).
        config: Pipeline configuration.

    Returns:
        Nested dict ready for JSON serialisation.
    """
    return {
        "metadata": {
            "pipeline_version": epinal_peak.__version__,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "config_snapshot": {
                "primary_tie_rule": config.primary_tie_rule,
                "min_years_for_trend": config.min_years_for_trend,
                "outlier_mad_threshold": config.outlier_mad_threshold,
                "min_diurnal_amplitude": config.min_diurnal_amplitude,
                "n_bootstrap": config.n_bootstrap,
                "bootstrap_ci_level": config.bootstrap_ci_level,
                "year_start": config.year_start,
                "year_end": config.year_end,
            },
        },
        "primary": primary,
        "seasonal": seasonal,
        "sensitivity": sensitivity,
        "supplementary": supplementary,
        "fallback": fallback,
    }


def main() -> None:
    """CLI entry point for the analysis stage.

    Reads the processed peak-hour CSV, fits all analyses (primary,
    seasonal, sensitivity, supplementary), and writes the complete
    nested results to ``config.results_dir`` as JSON.
    """
    config = EpinalPeakConfig()
    epinal_peak.setup_logging(config)

    _logger.info("Starting analysis stage")

    try:
        df = _load_and_prepare_data(config)
    except (FileNotFoundError, ValueError) as exc:
        _logger.error("Data loading failed: %s", exc)
        return

    # ── Primary analysis ────────────────────────────────────────────
    _logger.info("Primary: von Mises regression — %d obs", len(df))
    primary = von_mises_regression(df)

    _logger.info(
        "Primary result: beta_1=%.6f rad/yr (%.4f h/decade), "
        "kappa=%.2f, converged=%s",
        primary["beta_1"],
        primary["beta_1_hours_per_decade"],
        primary["kappa"],
        primary["converged"],
    )

    # Bootstrap CI for primary
    _logger.info("Primary: bootstrap CI (%d iter)", config.n_bootstrap)
    primary["bootstrap"] = _bootstrap_engine(
        df, n_iter=config.n_bootstrap, ci_level=config.bootstrap_ci_level
    )

    # ── Fallback (if MLE failed) ────────────────────────────────────
    fallback: dict[str, Any] | None = None
    if primary.get("status") in ("mle_did_not_converge",):
        _logger.info("Primary MLE failed — computing Mardia fallback correlation")
        fallback = _mardia_correlation(
            df["peak_hour"].values,
            df["year"].values,
            n_bootstrap=config.n_bootstrap_fallback,
        )

    # ── Seasonal stratification ─────────────────────────────────────
    _logger.info("Seasonal stratification")
    seasonal = seasonal_stratification(df, config)

    # ── Sensitivity analysis ────────────────────────────────────────
    _logger.info("Sensitivity analysis — 8 variants")
    sensitivity = sensitivity_analysis(df, config)

    # ── Supplementary analyses ──────────────────────────────────────
    supplementary: dict[str, Any] = {}

    _logger.info("Supplementary: temperature-weighted regression")
    supplementary["temperature_weighted"] = temperature_weighted_regression(df, config)

    _logger.info("Supplementary: circular-linear correlation")
    supplementary["circular_linear_correlation"] = circular_linear_correlation(
        df["peak_hour"].values,
        df["year"].values,
        n_bootstrap=config.n_bootstrap_fallback,
    )

    # ── Assemble output ─────────────────────────────────────────────
    output = _build_output_json(
        primary=primary,
        seasonal=seasonal,
        sensitivity=sensitivity,
        supplementary=supplementary,
        fallback=fallback,
        config=config,
    )

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / config.analysis_results_filename

    # Custom encoder to handle NaN → null in JSON
    class _NanEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            return super().default(o)

        def encode(self, o: Any) -> str:
            return super().encode(self._replace_nan(o))

        @staticmethod
        def _replace_nan(obj: Any) -> Any:
            if isinstance(obj, float):
                if np.isnan(obj):
                    return None
                return obj
            if isinstance(obj, dict):
                return {k: _NanEncoder._replace_nan(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_NanEncoder._replace_nan(v) for v in obj]
            return obj

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NanEncoder)
    _logger.info("Results written to %s", output_path)

    _logger.info("Completed analysis stage")


if __name__ == "__main__":
    main()
