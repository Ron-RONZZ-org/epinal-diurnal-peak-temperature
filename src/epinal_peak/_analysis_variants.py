"""Sensitivity variants and seasonal stratification.

Weighted regression functions have been extracted to
:mod:`_analysis_weighted` to keep this module under 500 lines.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

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

    # ── Run regression ──────────────────────────────────────────────
    from epinal_peak._analysis_regression import von_mises_regression as _vm_reg

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
    """Run all pooled sensitivity variants.

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


def seasonal_sensitivity_analysis(
    df: pd.DataFrame,
    n_iter: int = 1000,
    ci_level: float = 0.95,
    amplitude_thresholds: tuple[float, ...] | None = None,
    subsampling_bins: tuple[int, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run sensitivity variants per meteorological season.

    For each season (spring / summer / autumn / winter), runs the
    standard sensitivity variants independently.  This tests whether
    the per-season trend estimates are robust to methodological
    choices.

    Args:
        df: DataFrame with ``season`` column.
        n_iter: Bootstrap iterations per variant-season.
        ci_level: Bootstrap CI level.
        amplitude_thresholds: Alternative amplitude thresholds.
        subsampling_bins: Bin sizes for subsampling variants.

    Returns:
        Nested dict: outer keys are season names, inner keys are
        variant names mapping to result dicts.
    """
    at = amplitude_thresholds or (1.0, 3.0)
    sb = subsampling_bins or (3, 6)
    seasons = {"spring": "MAM", "summer": "JJA", "autumn": "SON", "winter": "DJF"}
    results: dict[str, dict[str, Any]] = {}

    for season_name, season_label in seasons.items():
        subset = df[df["season"] == season_name].copy()
        if subset.empty:
            results[season_name] = {"status": "no_data"}
            continue

        season_results: dict[str, Any] = {}
        n_years = int(subset["year"].nunique())

        # Tie rule: latest
        if "peak_hour_sensitivity" in subset.columns:
            season_results["tie_rule_latest"] = _run_variant(
                subset,
                {"tie_col": "peak_hour_sensitivity", "note": "tie_rule=latest"},
                n_iter=n_iter,
                ci_level=ci_level,
            )
            if "n_years" not in season_results["tie_rule_latest"]:
                season_results["tie_rule_latest"]["n_years"] = n_years

        # Subsampling variants
        for bin_size in sb:
            key = f"subsampling_{bin_size}hr"
            season_results[key] = _run_variant(
                subset,
                {"subsampling_bin": bin_size, "note": f"subsampling_bin={bin_size}"},
                n_iter=n_iter,
                ci_level=ci_level,
            )
            if "n_years" not in season_results[key]:
                season_results[key]["n_years"] = n_years

        # Amplitude threshold variants
        for threshold in at:
            key = f"amplitude_threshold_{threshold}"
            season_results[key] = _run_variant(
                subset,
                {
                    "amplitude_threshold": threshold,
                    "note": f"min_amplitude={threshold}",
                },
                n_iter=n_iter,
                ci_level=ci_level,
            )
            if "n_years" not in season_results[key]:
                season_results[key]["n_years"] = n_years

        results[season_name] = season_results

    return results


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

        from epinal_peak._analysis_regression import von_mises_regression as _vm_reg

        result = _vm_reg(subset)
        bootstrap = _bootstrap_engine(subset, n_iter=n_iter, ci_level=ci_level)
        result["bootstrap"] = bootstrap
        result["label"] = season_label
        result["n_years"] = int(subset["year"].nunique())
        results[season_name] = result

    return results



