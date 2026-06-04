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
from epinal_peak._analysis_bootstrap import _bootstrap_engine
from epinal_peak._analysis_corr import circular_linear_correlation, _mardia_correlation
from epinal_peak import _analysis_variants as _variants
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
    return _variants.temperature_weighted_regression(
        df,
        n_iter=config.n_bootstrap,
        ci_level=config.bootstrap_ci_level,
    )


# ── Output assembly ────────────────────────────────────────────────


def _build_output_json(
    primary: dict[str, Any],
    seasonal: dict[str, Any],
    sensitivity: dict[str, Any],
    supplementary: dict[str, Any],
    fallback: dict[str, Any] | None,
    interaction: dict[str, Any] | None,
    config: EpinalPeakConfig,
) -> dict[str, Any]:
    """Assemble the full nested output JSON.

    Args:
        primary: Primary von Mises regression result.
        seasonal: Seasonal stratification results.
        sensitivity: Sensitivity analysis results.
        supplementary: Supplementary analysis results.
        fallback: Mardia fallback result (or None).
        interaction: Interaction test result (or None).
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
        "primary": primary,
        "seasonal": seasonal,
        "sensitivity": sensitivity,
        "supplementary": supplementary,
    }
    if fallback is not None:
        output["fallback"] = fallback
    if interaction is not None:
        output["interaction"] = interaction
    return output


# ── CLI entry point ───────────────────────────────────────────────


def main() -> None:
    """CLI entry point for the analysis stage.

    Reads the processed peak-hour CSV, fits all analyses (primary,
    seasonal, sensitivity, supplementary, interaction), and writes the
    complete nested results to ``config.results_dir`` as JSON.
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

    # ── Interaction test: year × season ─────────────────────────────
    _logger.info("Interaction test: year × season von Mises LR test")
    interaction = interaction_lr_test(df)
    if interaction["status"] != "ok":
        _logger.warning("Interaction test failed: %s", interaction["status"])
    elif interaction["significant"]:
        _logger.info(
            "Interaction significant (χ²=%.2f, p=%.4f) — "
            "%d / 4 seasons rejected at FDR=%.2f",
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
        interaction=interaction,
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
