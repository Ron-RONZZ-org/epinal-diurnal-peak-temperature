"""Summary statistics and tabular reporting.

Computes descriptive statistics (N, circular mean, circular standard
deviation) per decade, formats primary and sensitivity result tables
from the analysis JSON, and exports to CSV and JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import epinal_peak
from epinal_peak.config import EpinalPeakConfig

_logger = logging.getLogger(__name__)


def _make_decade_label(year: int) -> str:
    """Return a decade label for a given year, e.g. 1992 -> '1986-1995'.

    Labels are non-overlapping bins of width *decade_width* (default 10
    years) aligned to the config's ``year_start``.
    """
    start = EpinalPeakConfig().year_start  # 1986
    y = int(year)
    decade = start + ((y - start) // 10) * 10
    return f"{decade}-{decade + 9}"


def load_peak_data(config: EpinalPeakConfig) -> pd.DataFrame:
    """Load the processed daily peak-hour CSV and derive derived columns.

    Args:
        config: Pipeline configuration.

    Returns:
        DataFrame with columns ``date``, ``year``, ``month``, ``season``,
        ``decade``, and the analysis-relevant columns from the CSV.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    path = Path(config.processed_dir) / config.peak_data_filename
    if not path.exists():
        raise FileNotFoundError(f"Peak-hour CSV not found: {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["decade"] = df["year"].apply(_make_decade_label)

    # Season
    def _season(m: int) -> str:
        if 3 <= m <= 5:
            return "spring"
        if 6 <= m <= 8:
            return "summer"
        if 9 <= m <= 11:
            return "autumn"
        return "winter"

    df["season"] = df["month"].apply(_season)

    _logger.info("Loaded %d records from %s", len(df), path)
    return df


def load_analysis_results(config: EpinalPeakConfig) -> dict[str, Any]:
    """Load the analysis results JSON.

    Args:
        config: Pipeline configuration.

    Returns:
        Nested dict with keys ``metadata``, ``primary``, ``seasonal``,
        ``sensitivity``, ``supplementary``.

    Raises:
        FileNotFoundError: If the JSON does not exist.
    """
    path = Path(config.results_dir) / config.analysis_results_filename
    if not path.exists():
        raise FileNotFoundError(f"Analysis results not found: {path}")

    with open(path) as f:
        results = json.load(f)

    _logger.info("Loaded analysis results from %s", path)
    return results


def summarize_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute descriptive statistics from peak-hour data.

    Groups data by decade and computes N, circular mean, and circular
    standard deviation of peak hours.  Also computes overall (all-years)
    statistics.

    Args:
        df: DataFrame with ``peak_hour`` and ``decade`` columns.
            Only rows with a non-null ``peak_hour`` and
            ``qc_excluded == False`` are used.

    Returns:
        Dictionary with keys ``overall`` and ``by_decade``.
        ``by_decade`` is a list of dicts ordered by decade.
    """
    from epinal_peak.analysis import circular_mean as _circ_mean
    from epinal_peak.analysis import circular_std as _circ_std

    valid = df[df["qc_excluded"] == False].copy()

    # Ensure decade column exists (may already be present from load_peak_data)
    if "decade" not in valid.columns and "year" in valid.columns:
        valid["decade"] = valid["year"].apply(_make_decade_label)

    # Overall
    peak = valid["peak_hour"].dropna().values
    overall: dict[str, Any] = {
        "n_observations": int(len(peak)),
        "n_years": int(valid["year"].nunique()),
        "circular_mean_hour": _safe_circ(_circ_mean, peak),
        "circular_std_hours": _safe_circ(_circ_std, peak),
        "year_min": int(valid["year"].min()),
        "year_max": int(valid["year"].max()),
    }

    # By decade
    decade_order = sorted(valid["decade"].unique(), key=_label_sort_key)
    by_decade: list[dict[str, Any]] = []
    for label in decade_order:
        subset = valid[valid["decade"] == label]
        pk = subset["peak_hour"].dropna().values
        by_decade.append({
            "decade": label,
            "n_observations": int(len(pk)),
            "n_years": int(subset["year"].nunique()),
            "circular_mean_hour": _safe_circ(_circ_mean, pk),
            "circular_std_hours": _safe_circ(_circ_std, pk),
        })

    return {"overall": overall, "by_decade": by_decade}


def _safe_circ(func: Any, values: np.ndarray) -> float:
    """Call a circular statistic function, returning NaN on failure."""
    if len(values) == 0:
        return float("nan")
    try:
        result = func(values)
        return float(result) if result is not None else float("nan")
    except (ValueError, RuntimeError):
        return float("nan")


def _label_sort_key(label: str) -> int:
    """Extract start year from a decade label like '1986-1995'."""
    return int(label.split("-")[0])


def _extract_ci(bootstrap: dict[str, Any] | None) -> tuple[float, float]:
    """Extract lower/upper CI from a bootstrap result dict."""
    if not bootstrap:
        return (float("nan"), float("nan"))
    return (
        bootstrap.get("ci_lower", float("nan")),
        bootstrap.get("ci_upper", float("nan")),
    )


def format_primary_table(
    primary: dict[str, Any],
) -> str:
    """Format the primary analysis result as a Markdown table.

    Args:
        primary: Dict with keys ``beta_1_hours_per_decade``, ``kappa``,
            ``n_obs``, ``status``, and optionally ``bootstrap``.

    Returns:
        A Markdown-formatted string.
    """
    beta = primary.get("beta_1_hours_per_decade")
    kappa = primary.get("kappa")
    n_obs = primary.get("n_obs", 0)
    status = primary.get("status", "unknown")
    boot = primary.get("bootstrap")
    ci_low, ci_upp = _extract_ci(boot)

    lines = [
        "| Parameter | Value |",
        "|-----------|-------|",
    ]

    if beta is not None and not (isinstance(beta, float) and np.isnan(beta)):
        lines.append(f"| β (hours/decade) | {beta:.4f} |")
    else:
        lines.append("| β (hours/decade) | — |")

    if not (isinstance(ci_low, float) and np.isnan(ci_low)):
        lines.append(f"| 95 % CI | ({ci_low:.4f}, {ci_upp:.4f}) |")
    else:
        lines.append("| 95 % CI | (not available) |")

    if kappa is not None and not (isinstance(kappa, float) and np.isnan(kappa)):
        lines.append(f"| κ (concentration) | {kappa:.2f} |")
    else:
        lines.append("| κ (concentration) | — |")

    lines.append(f"| N (valid days) | {n_obs} |")
    lines.append(f"| Status | {status} |")

    return "\n".join(lines) + "\n"


def format_sensitivity_table(
    sensitivity: dict[str, Any] | None,
) -> str:
    """Format sensitivity analysis results as a Markdown table.

    Args:
        sensitivity: Dict mapping variant names to result dicts, each
            containing ``beta_1_hours_per_decade``, ``n_obs``, and
            optionally ``bootstrap``.

    Returns:
        A Markdown-formatted string.  Returns a "no data" message if
        sensitivity is None or empty.
    """
    if not sensitivity:
        return "*No sensitivity analysis results available.*\n"

    lines = [
        "| Variant | β (h/decade) | 95 % CI | n | Status |",
        "|---------|-------------|---------|---|--------|",
    ]

    for variant_name, variant_result in sensitivity.items():
        if not isinstance(variant_result, dict):
            continue

        beta = variant_result.get("beta_1_hours_per_decade")
        n_obs = variant_result.get("n_obs", "—")
        status = variant_result.get("status", "—")
        boot = variant_result.get("bootstrap")
        ci_low, ci_upp = _extract_ci(boot)

        beta_str = (
            f"{beta:.4f}"
            if beta is not None and not (isinstance(beta, float) and np.isnan(beta))
            else "—"
        )
        ci_str = (
            f"({ci_low:.4f}, {ci_upp:.4f})"
            if not (isinstance(ci_low, float) and np.isnan(ci_low))
            else "—"
        )

        # Human-readable label
        label = variant_name.replace("_", " ").title()
        lines.append(f"| {label} | {beta_str} | {ci_str} | {n_obs} | {status} |")

    return "\n".join(lines) + "\n"


def format_seasonal_table(
    seasonal: dict[str, Any] | None,
) -> str:
    """Format seasonal stratification results as a Markdown table.

    Args:
        seasonal: Dict with keys ``spring``, ``summer``, ``autumn``,
            ``winter``, each containing regression results.

    Returns:
        A Markdown-formatted string.
    """
    if not seasonal:
        return "*No seasonal analysis results available.*\n"

    lines = [
        "| Season | β (h/decade) | 95 % CI | n | Status |",
        "|--------|-------------|---------|---|--------|",
    ]

    for season_name in ("spring", "summer", "autumn", "winter"):
        result = seasonal.get(season_name)
        if not isinstance(result, dict):
            continue

        label = result.get("label", season_name.title())
        beta = result.get("beta_1_hours_per_decade")
        n_obs = result.get("n_obs", "—")
        status = result.get("status", "—")
        boot = result.get("bootstrap")
        ci_low, ci_upp = _extract_ci(boot)

        beta_str = (
            f"{beta:.4f}"
            if beta is not None and not (isinstance(beta, float) and np.isnan(beta))
            else "—"
        )
        ci_str = (
            f"({ci_low:.4f}, {ci_upp:.4f})"
            if not (isinstance(ci_low, float) and np.isnan(ci_low))
            else "—"
        )
        lines.append(f"| {label} | {beta_str} | {ci_str} | {n_obs} | {status} |")

    return "\n".join(lines) + "\n"


def format_descriptive_table(stats: dict[str, Any]) -> str:
    """Format descriptive statistics as a Markdown table.

    Args:
        stats: Dictionary returned by :func:`summarize_statistics`.

    Returns:
        A Markdown-formatted string.
    """
    overall = stats.get("overall", {})
    lines = [
        "| Period | n | n_years | Circular mean (h) | Circular SD (h) |",
        "|--------|---|---------|-------------------|-----------------|",
    ]

    # Overall row
    lines.append(
        f"| **Overall** ({overall.get('year_min', '')}"
        f"–{overall.get('year_max', '')})"
        f" | {overall.get('n_observations', 0)}"
        f" | {overall.get('n_years', 0)}"
        f" | {_fmt_stat(overall.get('circular_mean_hour'))}"
        f" | {_fmt_stat(overall.get('circular_std_hours'))} |"
    )

    for period in stats.get("by_decade", []):
        lines.append(
            f"| {period.get('decade', '')}"
            f" | {period.get('n_observations', 0)}"
            f" | {period.get('n_years', 0)}"
            f" | {_fmt_stat(period.get('circular_mean_hour'))}"
            f" | {_fmt_stat(period.get('circular_std_hours'))} |"
        )

    return "\n".join(lines) + "\n"


def _fmt_stat(value: Any) -> str:
    """Format a statistic for display, handling NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):.2f}"


def export_stats(
    stats: dict[str, Any],
    config: EpinalPeakConfig,
) -> None:
    """Export summary statistics to CSV and JSON.

    Writes ``summary_stats.csv`` (by-decade rows) and
    ``summary_stats.json`` (full nested dict including overall)
    to the results directory.

    Args:
        stats: Dictionary from :func:`summarize_statistics`.
        config: Pipeline configuration.
    """
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # CSV — by-decade rows
    rows = stats.get("by_decade", [])
    if rows:
        csv_path = results_dir / "summary_stats.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        _logger.info("Exported %d decade rows to %s", len(rows), csv_path)

    # JSON — full nested dict
    json_path = results_dir / "summary_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    _logger.info("Exported full stats to %s", json_path)


def table_names() -> dict[str, str]:
    """Return a mapping of table names to their markdown content.

    Returns:
        Dict with keys ``primary``, ``sensitivity``, ``seasonal``,
        ``descriptive``, each containing the markdown string.
    """
    # This is a placeholder for the user to capture output after calling
    # the format functions directly.  The real tables are generated
    # dynamically in :func:`main`.
    return {
        "primary": "",
        "sensitivity": "",
        "seasonal": "",
        "descriptive": "",
    }


def main() -> None:
    """CLI entry point for the reporting stage.

    Loads the processed peak-hour CSV and analysis results JSON,
    computes descriptive statistics, formats all tables, and prints
    them.  Exports statistics to ``results/summary_stats.csv`` and
    ``results/summary_stats.json``.
    """
    config = EpinalPeakConfig()
    epinal_peak.setup_logging(config)

    _logger.info("Starting report stage")

    # ── Load data ───────────────────────────────────────────────────
    try:
        df = load_peak_data(config)
    except FileNotFoundError as exc:
        _logger.error("Data loading failed: %s", exc)
        return

    try:
        results = load_analysis_results(config)
    except FileNotFoundError as exc:
        _logger.warning("Analysis results not available: %s", exc)
        results = {}

    # ── Descriptive statistics ──────────────────────────────────────
    _logger.info("Computing descriptive statistics")
    stats = summarize_statistics(df)
    export_stats(stats, config)

    print("\n" + "=" * 72)
    print("DESCRIPTIVE STATISTICS — Peak hour by decade")
    print("=" * 72)
    print(format_descriptive_table(stats))

    # ── Primary result ─────────────────────────────────────────────
    primary = results.get("primary", {})
    if primary:
        print("\n" + "=" * 72)
        print("PRIMARY ANALYSIS — Von Mises regression")
        print("=" * 72)
        print(format_primary_table(primary))

    # ── Sensitivity analysis ────────────────────────────────────────
    sensitivity = results.get("sensitivity", {})
    if sensitivity:
        print("\n" + "=" * 72)
        print("SENSITIVITY ANALYSIS — All variants")
        print("=" * 72)
        print(format_sensitivity_table(sensitivity))

    # ── Seasonal stratification ─────────────────────────────────────
    seasonal = results.get("seasonal", {})
    if seasonal:
        print("\n" + "=" * 72)
        print("SEASONAL STRATIFICATION — By meteorological season")
        print("=" * 72)
        print(format_seasonal_table(seasonal))

    _logger.info("Completed report stage")


if __name__ == "__main__":
    main()
