"""Pipeline configuration.

All tunable parameters, thresholds, and file paths are declared in the
:class:`EpinalPeakConfig` dataclass.  No module in this package hardcodes
a threshold or a path — they all read from a config instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EpinalPeakConfig:
    """Configuration for the Epinal diurnal peak temperature pipeline.

    All pipeline parameters are defined here.  Modules import the config
    class rather than hardcoding values.  Path fields are relative to the
    project root directory (assumed to be the current working directory at
    runtime).

    The dataclass is frozen (immutable) to prevent accidental modification
    after construction.  Sensible defaults are provided so that
    ``EpinalPeakConfig()`` works out of the box.
    """

    # ── Station metadata ──────────────────────────────────────────────
    station_id: str = "88136001"
    """Météo-France RADOME station identifier for Épinal."""
    station_name: str = "EPINAL"
    """Official Météo-France station name (uppercase)."""
    latitude: float = 48.210833
    longitude: float = 6.451667
    elevation_m: int = 317
    timezone: str = "Europe/Paris"
    """IANA timezone string.  CET/CEST handled by ``zoneinfo``."""

    # ── Data acquisition ──────────────────────────────────────────────
    pooch_registry_file: str = "registry.txt"
    """Filename in *external_dir* for pooch registry (URL -> hash mappings)."""
    local_fallback_dir: str = "fallback"
    """Subdirectory under *external_dir* for local backup data files."""

    # ── QC thresholds ─────────────────────────────────────────────────
    max_missing_hours_per_day: int = 6
    """Maximum missing hourly observations per day for a valid day."""
    min_valid_fraction: float = 0.75
    """Minimum fraction of observations that must be non-null for validity."""
    outlier_mad_threshold: float = 5.0
    """Number of median absolute deviations from median for outlier flagging."""
    min_years_for_trend: int = 30
    """Minimum years of data required for meaningful trend analysis."""
    min_diurnal_amplitude: float = 2.0
    """Minimum diurnal temperature range (°C) for a valid day.
    Days with peak minus minimum temperature below this threshold
    are excluded (common instrument or radiation-error flag)."""

    # ── Analysis period ───────────────────────────────────────────────
    year_start: int = 1986
    """First calendar year to include in analysis.
    Station MF88136001 (Épinal) opened 1 June 1986 — no sub-daily data
    exists before this date.
    """
    year_end: int = 2025
    """Last calendar year to include in analysis (inclusive)."""

    # ── Peak extraction ───────────────────────────────────────────────
    primary_tie_rule: str = "earliest"
    """Tie-breaking rule for primary analysis.  One of ``'earliest'`` or ``'latest'``."""
    sensitivity_tie_rule: str = "latest"
    """Tie-breaking rule for sensitivity analysis."""

    # ── Sensitivity analysis ──────────────────────────────────────────
    sensitivity_min_years: int = 20
    """Alternative *min_years_for_trend* for the 'reduced years' sensitivity
    variant.  Used to test whether the trend is robust to a shorter record."""
    sensitivity_mad_threshold: float = 3.0
    """Alternative *outlier_mad_threshold* for the 'strict outlier' sensitivity
    variant.  Flags outliers more aggressively than the primary threshold."""
    sensitivity_exclude_post_2000: bool = False
    """If ``True``, excludes all data after year 2000 for the 'post-2000
    exclusion' sensitivity variant.  Tests whether recent decades drive
    the trend."""

    # ── Primary endpoint ──────────────────────────────────────────────
    primary_endpoint: str = "slope_beta_hours_per_decade"
    """Primary outcome measure: slope β from von Mises circular GLM
    expressed in hours per decade.  A positive value indicates the daily
    maximum temperature occurs later in the day over time."""

    # ── Seasonal definitions (Northern Hemisphere, meteorological) ────
    spring_start: int = 3
    spring_end: int = 5
    """March, April, May — meteorological spring."""
    summer_start: int = 6
    summer_end: int = 8
    """June, July, August — meteorological summer."""
    autumn_start: int = 9
    autumn_end: int = 11
    """September, October, November — meteorological autumn."""
    winter_start: int = 12  # December of year N-1
    winter_end: int = 2  # February of year N
    """December, January, February — meteorological winter (DJF)."""

    # ── Circular regression ───────────────────────────────────────────
    n_bootstrap: int = 1000
    """Number of bootstrap resamples for confidence intervals."""
    bootstrap_ci_level: float = 0.95
    """Confidence level for bootstrap percentile intervals."""

    # ── Paths (relative to project root) ──────────────────────────────
    data_dir: Path = field(default_factory=lambda: Path("data"))
    raw_dir: Path = field(default_factory=lambda: Path("data") / "raw")
    processed_dir: Path = field(default_factory=lambda: Path("data") / "processed")
    external_dir: Path = field(default_factory=lambda: Path("data") / "external")
    results_dir: Path = field(default_factory=lambda: Path("results"))
    figures_dir: Path = field(default_factory=lambda: Path("results") / "figures")
    logs_dir: Path = field(default_factory=lambda: Path("logs"))

    # ── Output filenames ──────────────────────────────────────────────
    raw_data_filename: str = "epinal_temperature_raw.csv"
    qc_data_filename: str = "epinal_temperature_qc.csv"
    hourly_data_filename: str = "epinal_temperature_hourly.csv"
    peak_data_filename: str = "epinal_daily_peak_hour.csv"
    analysis_results_filename: str = "epinal_analysis_results.json"
