"""Circular statistics and von Mises regression for peak-hour trends.

This module is the **public API** — it re-exports low-level functions
from private submodules and orchestrates the full analysis pipeline
(primary regression, seasonal stratification, sensitivity variants,
supplementary analyses, and the year × season interaction test).
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import epinal_peak
from epinal_peak import _analysis_variants as _variants
from epinal_peak._analysis_bootstrap import _bootstrap_engine
from epinal_peak._analysis_corr import circular_linear_correlation, _mardia_correlation
from epinal_peak._analysis_regression import (
    _hours_to_radians,
    _radians_to_hours,
    _HOURS_TO_RAD,
    _RAD_TO_HOURS,
    circular_mean,
    circular_std,
    von_mises_regression,
    bootstrap_trend,
)
from epinal_peak._analysis_interaction import (
    interaction_lr_test,
)
from epinal_peak.config import EpinalPeakConfig

_logger = logging.getLogger(__name__)


# ── Seasonal helpers ────────────────────────────────────────────────


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

    valid.drop(columns=["month"], inplace=True)

    return valid


# ── Public analysis functions (re-exported orchestrators) ───────────


def sensitivity_analysis(df: pd.DataFrame, config: EpinalPeakConfig) -> dict:
    """Run pooled sensitivity variants of the trend analysis.

    Delegates to :mod:`_analysis_variants` for the actual computation.
    The 6 variants are:

    1. ``tie_rule_latest`` — use latest-peak tie-breaking
    2. ``min_years_20`` — reduce minimum years to 20
    3. ``subsampling_3hr`` — round peak hours to 3-hour bins
    4. ``subsampling_6hr`` — round peak hours to 6-hour bins
    5. ``amplitude_threshold_1`` — min diurnal amplitude 1.0 °C
    6. ``amplitude_threshold_3`` — min diurnal amplitude 3.0 °C

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
    """Run primary analysis per meteorological season.

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
    from epinal_peak._analysis_weighted import temperature_weighted_regression as _twr

    return _twr(
        df,
        n_iter=config.n_bootstrap,
        ci_level=config.bootstrap_ci_level,
    )


def seasonal_sensitivity_analysis(
    df: pd.DataFrame, config: EpinalPeakConfig
) -> dict[str, dict[str, Any]]:
    """Run sensitivity variants per meteorological season.

    For each season, runs the standard sensitivity variants
    (tie-breaking, subsampling, amplitude thresholds) independently.

    Args:
        df: DataFrame with ``season`` column.
        config: Pipeline configuration.

    Returns:
        Nested dict: outer keys are seasons, inner are variant names.
    """
    _logger.info("Seasonal sensitivity analysis — 4 seasons × 6 variants")
    return _variants.seasonal_sensitivity_analysis(
        df,
        n_iter=config.n_bootstrap,
        ci_level=config.bootstrap_ci_level,
        amplitude_thresholds=tuple(config.amplitude_thresholds),
        subsampling_bins=tuple(config.subsampling_bins),
    )


def _seasonal_temperature_weighted(
    df: pd.DataFrame, config: EpinalPeakConfig
) -> dict[str, Any]:
    """Run temperature-weighted regression per season.

    Args:
        df: DataFrame with ``season`` column.
        config: Pipeline configuration.

    Returns:
        Dict mapping season names to their regression results.
        Includes a ``pooled`` key for the all-season reference.
    """
    from epinal_peak._analysis_weighted import temperature_weighted_regression as _twr

    results: dict[str, Any] = {}
    for season_name in ["spring", "summer", "autumn", "winter"]:
        subset = df[df["season"] == season_name]
        if subset.empty:
            results[season_name] = {"status": "no_data"}
        else:
            results[season_name] = _twr(
                subset,
                n_iter=config.n_bootstrap,
                ci_level=config.bootstrap_ci_level,
            )
    results["pooled"] = _twr(
        df,
        n_iter=config.n_bootstrap,
        ci_level=config.bootstrap_ci_level,
    )
    return results


def _seasonal_circular_linear_correlation(
    df: pd.DataFrame, config: EpinalPeakConfig
) -> dict[str, Any]:
    """Run circular-linear correlation per season.

    Args:
        df: DataFrame with ``season``, ``peak_hour``, ``year`` columns.
        config: Pipeline configuration.

    Returns:
        Dict mapping season names to their correlation results.
        Includes a ``pooled`` key for the all-season reference.
    """
    results: dict[str, Any] = {}
    for season_name in ["spring", "summer", "autumn", "winter"]:
        subset = df[df["season"] == season_name]
        if subset.empty:
            results[season_name] = {"status": "no_data"}
        else:
            results[season_name] = circular_linear_correlation(
                subset["peak_hour"].values,
                subset["year"].values,
                n_bootstrap=config.n_bootstrap_fallback,
            )
    results["pooled"] = circular_linear_correlation(
        df["peak_hour"].values,
        df["year"].values,
        n_bootstrap=config.n_bootstrap_fallback,
    )
    return results


def _seasonal_autocorrelation(df: pd.DataFrame) -> dict[str, Any]:
    """Compute lag-1 through lag-7 autocorrelation per season.

    Args:
        df: DataFrame with ``season``, ``date``, ``peak_hour`` columns.

    Returns:
        Dict mapping season names to a list of lag-1..7 ACF values.
        Includes a ``pooled`` entry.
    """
    results: dict[str, Any] = {}
    for season_name in ["spring", "summer", "autumn", "winter"]:
        subset = df[df["season"] == season_name].sort_values("date")
        ph = subset["peak_hour"].values
        if len(ph) < 8:
            results[season_name] = None
            continue
        acf = []
        for lag in range(1, 8):
            acf.append(float(np.corrcoef(ph[:-lag], ph[lag:])[0, 1]))
        results[season_name] = acf
    # Pooled
    df_sorted = df.sort_values("date")
    ph = df_sorted["peak_hour"].values
    acf = []
    for lag in range(1, 8):
        acf.append(float(np.corrcoef(ph[:-lag], ph[lag:])[0, 1]))
    results["pooled"] = acf
    return results


def _build_output_json(
    primary: dict[str, Any],
    seasonal: dict[str, Any],
    sensitivity: dict[str, Any],
    supplementary: dict[str, Any],
    fallback: dict[str, Any] | None,
    interaction: dict[str, Any] | None,
    seasonal_sensitivity: dict[str, Any] | None,
    autocorrelation: dict[str, Any] | None,
    config: EpinalPeakConfig,
) -> dict[str, Any]:
    """Assemble the full nested output JSON.

    Args:
        primary: Pooled von Mises regression result (supplementary).
        seasonal: Seasonal stratification results (primary).
        sensitivity: Pooled sensitivity analysis results.
        supplementary: Supplementary analysis results (includes
            per-season temperature-weighted regression and
            circular-linear correlation).
        fallback: Mardia fallback result (or None).
        interaction: Interaction test result (or None).
        seasonal_sensitivity: Per-season sensitivity results (or None).
        autocorrelation: Per-season ACF diagnostic (or None).
        config: Pipeline configuration.

    Returns:
        Nested dict ready for JSON serialisation.
    """
    output: dict[str, Any] = {
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
        "interaction": interaction,
        "seasonal": seasonal,
    }
    if seasonal_sensitivity is not None:
        output["seasonal_sensitivity"] = seasonal_sensitivity
    if autocorrelation is not None:
        output["autocorrelation"] = autocorrelation
    output["pooled"] = primary
    output["sensitivity"] = sensitivity
    output["supplementary"] = supplementary
    if fallback is not None:
        output["fallback"] = fallback
    return output


# ── CLI entry point ───────────────────────────────────────────────


def main() -> None:
    """CLI entry point for the analysis stage.

    Reads the processed peak-hour CSV and runs:

    1. Year--season interaction test (diagnostic justification)
    2. Seasonal stratification (**primary** analysis)
    3. Seasonal sensitivity variants
    4. Pooled regression (supplementary)
    5. Pooled sensitivity (supplementary)
    6. Supplementary analyses (pooled + per-season)
    7. Autocorrelation diagnostic (ACF lag-1..7 per season)
    """
    config = EpinalPeakConfig()
    epinal_peak.setup_logging(config)

    _logger.info("Starting analysis stage")

    try:
        df = _load_and_prepare_data(config)
    except (FileNotFoundError, ValueError) as exc:
        _logger.error("Data loading failed: %s", exc)
        return

    # ── 1. Interaction test: year × season (diagnostic) ─────────────
    _logger.info("Year × season interaction LR test (diagnostic)")
    interaction = interaction_lr_test(df)
    if interaction["status"] != "ok":
        _logger.warning("Interaction test failed: %s", interaction["status"])
    elif interaction["significant"]:
        _logger.info(
            "Interaction significant (χ²=%.2f, p=%.4f) — "
            "%d / 4 seasons rejected at FDR=%.2f — "
            "justifying seasonal primary analysis",
            interaction["lr_stat"],
            interaction["p_value"],
            interaction["per_season"]["n_seasons_rejected"],
            interaction["per_season"]["fdr_alpha"],
        )
    else:
        _logger.info(
            "Interaction not significant (χ²=%.2f, p=%.4f)",
            interaction["lr_stat"],
            interaction["p_value"],
        )

    # ── 2. Seasonal stratification (PRIMARY analysis) ───────────────
    _logger.info("PRIMARY: Seasonal stratification — von Mises regression per season")
    seasonal = seasonal_stratification(df, config)
    for s in ["spring", "summer", "autumn", "winter"]:
        r = seasonal.get(s, {})
        if r.get("status") == "ok":
            _logger.info(
                "  %s: β=%.4f h/decade [%.4f, %.4f], n=%d",
                s,
                r["beta_1_hours_per_decade"],
                r["bootstrap"]["ci_lower"],
                r["bootstrap"]["ci_upper"],
                r["n_obs"],
            )

    # ── 3. Seasonal sensitivity ─────────────────────────────────────
    _logger.info("Seasonal sensitivity — 4 seasons × 6 variants")
    seas_sens = seasonal_sensitivity_analysis(df, config)

    # ── 4. Pooled regression (supplementary) ────────────────────────
    _logger.info("SUPPLEMENTARY: Pooled von Mises regression — %d obs", len(df))
    primary = von_mises_regression(df)

    _logger.info(
        "Pooled result: beta_1=%.6f rad/yr (%.4f h/decade), "
        "kappa=%.2f, converged=%s",
        primary["beta_1"],
        primary["beta_1_hours_per_decade"],
        primary["kappa"],
        primary["converged"],
    )

    # Bootstrap CI for pooled
    _logger.info("Pooled: bootstrap CI (%d iter)", config.n_bootstrap)
    primary["bootstrap"] = _bootstrap_engine(
        df, n_iter=config.n_bootstrap, ci_level=config.bootstrap_ci_level
    )

    # ── Fallback (if MLE failed) ────────────────────────────────────
    fallback: dict[str, Any] | None = None
    if primary.get("status") in ("mle_did_not_converge",):
        _logger.info("Pooled MLE failed — computing Mardia fallback correlation")
        fallback = _mardia_correlation(
            df["peak_hour"].values,
            df["year"].values,
            n_bootstrap=config.n_bootstrap_fallback,
        )

    # ── 5. Pooled sensitivity (supplementary) ───────────────────────
    _logger.info("SUPPLEMENTARY: Pooled sensitivity analysis — 8 variants")
    sensitivity = sensitivity_analysis(df, config)

    # ── 6. Supplementary analyses ───────────────────────────────────
    supplementary: dict[str, Any] = {}

    _logger.info("Supplementary: temperature-weighted regression (pooled)")
    supplementary["temperature_weighted"] = temperature_weighted_regression(df, config)

    _logger.info("Supplementary: circular-linear correlation (pooled)")
    supplementary["circular_linear_correlation"] = circular_linear_correlation(
        df["peak_hour"].values,
        df["year"].values,
        n_bootstrap=config.n_bootstrap_fallback,
    )

    # Per-season supplementary analyses
    _logger.info(
        "Supplementary: seasonal temperature-weighted regression — 4 seasons + pooled"
    )
    supplementary["temperature_weighted_seasonal"] = _seasonal_temperature_weighted(
        df, config
    )

    _logger.info(
        "Supplementary: seasonal circular-linear correlation — 4 seasons + pooled"
    )
    supplementary["correlation_seasonal"] = _seasonal_circular_linear_correlation(
        df, config
    )

    # ── 7. Autocorrelation diagnostic ──────────────────────────────
    _logger.info("Autocorrelation diagnostic: ACF lag-1..7 per season")
    autocorrelation = _seasonal_autocorrelation(df)

    # ── Assemble output ─────────────────────────────────────────────
    output = _build_output_json(
        primary=primary,
        seasonal=seasonal,
        sensitivity=sensitivity,
        supplementary=supplementary,
        fallback=fallback,
        interaction=interaction,
        seasonal_sensitivity=seas_sens,
        autocorrelation=autocorrelation,
        config=config,
    )

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / config.analysis_results_filename

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
