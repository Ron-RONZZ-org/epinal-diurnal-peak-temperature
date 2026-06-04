"""Quality control, UTC-to-local conversion, and daily peak extraction.

Transforms raw sub-daily temperature records into a cleaned dataset of daily
peak temperature hours.  Handles DST transitions, missing data, and quality
control thresholds defined in :class:`~epinal_peak.config.EpinalPeakConfig`.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import epinal_peak
from epinal_peak._preprocess_qc import (
    filter_physical_limits,
    flag_circular_outliers,
    flag_flatline,
)
from epinal_peak._preprocess_schemas import (
    DailyPeaksSchema,
    RawInputSchema,
    validate_input_schema,
    validate_output_schema,
)
from epinal_peak.config import EpinalPeakConfig

__all__: list[str] = [
    "DailyPeaksSchema",
    "RawInputSchema",
    "extract_daily_peak_hour",
    "filter_physical_limits",
    "flag_circular_outliers",
    "flag_flatline",
    "main",
    "utc_to_local",
    "validate_input_schema",
    "validate_output_schema",
]

_logger = logging.getLogger(__name__)

_HOURS_IN_DAY = 24


# ── Time conversion ─────────────────────────────────────────────────────


def utc_to_local(df: pd.DataFrame, tz: str = "Europe/Paris") -> pd.DataFrame:
    """Convert UTC timestamps to local time and add convenience columns.

    Parses the ``AAAAMMJJHH`` column (YYYYMMDDHH format), converts from
    UTC to the target IANA timezone via ``zoneinfo``, and adds three
    new columns: ``local_time`` (full datetime), ``local_date`` (date
    only), and ``local_hour`` (int 0--23).

    DST transitions are handled transparently by ``zoneinfo``: the 25-hour
    autumn-back day uses the ``fold`` attribute to distinguish the two
    02:00--03:00 CET/CEST blocks.  The 23-hour spring-forward day simply
    has fewer records.

    Args:
        df: DataFrame with a ``AAAAMMJJHH`` column (str, YYYYMMDDHH).
        tz: IANA timezone string (default ``'Europe/Paris'``).

    Returns:
        DataFrame with added ``local_time``, ``local_date``, and
        ``local_hour`` columns.  The original ``AAAAMMJJHH`` column
        is preserved.

    Raises:
        ValueError: If ``AAAAMMJJHH`` is missing.
    """
    from zoneinfo import ZoneInfo

    if "AAAAMMJJHH" not in df.columns:
        raise ValueError("Input must contain 'AAAAMMJJHH' column.")

    df = df.copy()

    utc_dt = pd.to_datetime(df["AAAAMMJJHH"], format="%Y%m%d%H", errors="coerce")
    utc_dt = utc_dt.dt.tz_localize("UTC")
    local_tz = ZoneInfo(tz)
    local_dt = utc_dt.dt.tz_convert(local_tz)

    df["local_time"] = local_dt
    df["local_date"] = local_dt.dt.date
    df["local_hour"] = local_dt.dt.hour

    _logger.info(
        "Time conversion: utc→%s (%d records, %d unique dates)",
        tz,
        len(df),
        df["local_date"].nunique(),
    )
    return df


# ── Daily peak extraction ───────────────────────────────────────────────


def extract_daily_peak_hour(
    df: pd.DataFrame, config: EpinalPeakConfig,
) -> pd.DataFrame:
    """Aggregate hourly data to daily records with QC flags.

    For each local calendar date:

    * Computes: Tmax, Tmin, peak hours (earliest/latest tie-breaking),
      number of valid obs, diurnal amplitude.
    * Carries forward any ``flatline_flag`` from the hourly data.
    * Applies **missing-data QC**: days with fewer than
      ``24 - max_missing_hours_per_day`` valid observations are excluded.
    * Applies **amplitude QC**: days with diurnal amplitude below
      ``min_diurnal_amplitude`` are excluded.
    * Flags **temporal gaps**: days that follow more than
      ``temporal_gap_days`` consecutive missing dates receive a
      ``temporal_gap_flag``.

    All-missing days (zero observations) are represented with NaN
    peak hours and ``qc_excluded = True``, rather than being dropped
    entirely.

    Args:
        df: Hourly DataFrame with ``local_date``, ``local_hour``,
            ``temperature_c``, and ``flatline_flag`` columns.
        config: Pipeline configuration with QC thresholds.

    Returns:
        Daily peak-hour DataFrame conforming to :class:`DailyPeaksSchema`.
    """
    # Build a complete date grid for temporal-gap detection.
    start = pd.Timestamp(f"{config.year_start}-01-01")
    end = pd.Timestamp(f"{config.year_end}-12-31")
    all_dates = pd.date_range(start, end, freq="D").date

    min_valid = _HOURS_IN_DAY - config.max_missing_hours_per_day

    records: list[dict] = []
    seen_dates: set = set()

    for date_key, grp in df.groupby("local_date", sort=True):
        seen_dates.add(date_key)

        temps = grp["temperature_c"].values
        hours = grp["local_hour"].values
        flatline = bool(grp["flatline_flag"].any())

        valid_mask = ~np.isnan(temps)
        n_valid = int(valid_mask.sum())

        if n_valid == 0:
            records.append(_empty_day(date_key, flatline, excluded=True))
            continue

        t_max = float(np.nanmax(temps))
        t_min = float(np.nanmin(temps))

        # Tie-breaking: find hours where temp == T_max.
        peak_hours = hours[valid_mask & (temps == t_max)]
        peak_hour_primary = int(np.min(peak_hours))
        peak_hour_latest = int(np.max(peak_hours))

        amplitude = t_max - t_min

        # Missing-data threshold.
        if n_valid < min_valid:
            records.append(
                _empty_day(
                    date_key, flatline, excluded=True,
                    n_valid=n_valid, t_min=t_min, t_max=t_max,
                )
            )
            continue

        # Amplitude filter.
        if amplitude < config.min_diurnal_amplitude:
            records.append(
                _empty_day(
                    date_key, flatline, excluded=True,
                    n_valid=n_valid, t_min=t_min, t_max=t_max,
                )
            )
            continue

        records.append(
            {
                "date": date_key,
                "peak_hour_primary": peak_hour_primary,
                "peak_hour_latest": peak_hour_latest,
                "peak_temperature": t_max,
                "peak_temperature_min": t_min,
                "n_valid_obs": n_valid,
                "diurnal_amplitude": amplitude,
                "flatline_flag": flatline,
                "temporal_gap_flag": False,
                "circular_outlier_flag": False,
                "qc_excluded": False,
            }
        )

    # ── Temporal gap detection (on original seen dates) ────────────────
    gap_flags_map: dict = {}  # date → bool
    sorted_seen = sorted(seen_dates)
    for i in range(1, len(sorted_seen)):
        prev = sorted_seen[i - 1]
        curr = sorted_seen[i]
        if isinstance(prev, datetime.date):
            delta = (curr - prev).days
        else:
            delta = (pd.Timestamp(curr) - pd.Timestamp(prev)).days
        if delta > config.temporal_gap_days + 1:
            gap_flags_map[curr] = True
            _logger.warning(
                "Temporal gap detected: %s → %s (%d missing days)",
                prev, curr, delta - 1,
            )

    # Update temporal_gap_flag for records from seen dates.
    for rec in records:
        if rec["date"] in gap_flags_map:
            rec["temporal_gap_flag"] = True

    peaks = pd.DataFrame(records)

    # Fill missing dates (days with zero hourly records) — single concat.
    missing_dates = sorted(set(all_dates) - seen_dates)
    if missing_dates:
        missing_records = [
            _empty_day(md, flatline=False, excluded=True)
            for md in missing_dates
        ]
        peaks = pd.concat(
            [peaks, pd.DataFrame(missing_records)], ignore_index=True,
        )

    # ── Post-processing ─────────────────────────────────────────────────
    if not peaks.empty:
        peaks = peaks.sort_values("date").reset_index(drop=True)
    else:
        _logger.warning("No daily records produced — edge case.")

    # Cast nullable integer columns to float64 for pandera compatibility.
    for col in ("n_valid_obs", "peak_hour_primary", "peak_hour_latest",
                "peak_temperature", "peak_temperature_min", "diurnal_amplitude"):
        peaks[col] = pd.to_numeric(peaks[col], errors="coerce").astype("float64")

    # Ensure ``date`` is datetime64[ns] for pandera compatibility.
    peaks["date"] = pd.to_datetime(peaks["date"]).astype("datetime64[ns]")

    n_excluded = peaks["qc_excluded"].sum()
    _logger.info(
        "Daily peaks extracted: %d total, %d excluded, %d valid",
        len(peaks),
        n_excluded,
        len(peaks) - n_excluded,
    )
    return peaks


def _empty_day(
    date_key,
    flatline: bool = False,
    excluded: bool = True,
    n_valid: int = 0,
    t_min: float = float("nan"),
    t_max: float = float("nan"),
) -> dict:
    """Return a record dict for a day with no valid peak."""
    return {
        "date": date_key,
        "peak_hour_primary": None,
        "peak_hour_latest": None,
        "peak_temperature": t_max if not np.isnan(t_max) else None,
        "peak_temperature_min": t_min if not np.isnan(t_min) else None,
        "n_valid_obs": n_valid,
        "diurnal_amplitude": (t_max - t_min) if not np.isnan(t_max - t_min) else None,
        "flatline_flag": flatline,
        "temporal_gap_flag": False,
        "circular_outlier_flag": False,
        "qc_excluded": excluded,
    }


def _flag_temporal_gaps(
    sorted_dates: np.ndarray,
    gap_threshold: int,
) -> list[bool]:
    """Return a boolean array flagging days after long gaps.

    A day is flagged if the previous available date is more than
    *gap_threshold* days earlier.  The first day of the dataset
    is never flagged.

    Args:
        sorted_dates: 1-D array of date-like objects, sorted ascending.
        gap_threshold: Maximum allowed consecutive missing days.

    Returns:
        Boolean list of the same length as *sorted_dates*.
    """
    if len(sorted_dates) == 0:
        return []

    flags: list[bool] = [False]
    for i in range(1, len(sorted_dates)):
        delta = (pd.Timestamp(sorted_dates[i]) - pd.Timestamp(sorted_dates[i - 1])).days
        flags.append(delta > gap_threshold + 1)

    n_gaps = sum(flags)
    if n_gaps:
        _logger.warning("Temporal gaps flagged: %d gap events detected", n_gaps)
    return flags


# ── Pipeline I/O helpers (private) ──────────────────────────────────────


def _load_raw_data(config: EpinalPeakConfig) -> pd.DataFrame:
    """Load raw temperature data from the acquire stage output.

    Args:
        config: Pipeline configuration.

    Returns:
        DataFrame with raw columns: ``NUM_POSTE``, ``AAAAMMJJHH``,
        ``T``, ``QT``.

    Raises:
        FileNotFoundError: If ``make acquire`` has not been run yet.
    """
    path = config.raw_dir / config.raw_data_filename
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}.  Run 'make acquire' first."
        )
    _logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    _logger.info("Loaded %d records", len(df))
    return df


def _save_daily_peaks(peaks: pd.DataFrame, config: EpinalPeakConfig) -> Path:
    """Write the daily peak dataset to disk.

    Backward-compat columns ``peak_hour`` and ``peak_hour_sensitivity``
    are written alongside the primary columns.

    Args:
        peaks: Daily peak DataFrame conforming to :class:`DailyPeaksSchema`.
        config: Pipeline configuration.

    Returns:
        Path to the written CSV file.
    """
    output_path = config.processed_dir / config.peak_data_filename
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    out = peaks.copy()
    out["peak_hour"] = out["peak_hour_primary"]
    out["peak_hour_sensitivity"] = out["peak_hour_latest"]

    out.to_csv(output_path, index=False)
    _logger.info("Saved %d daily peak records → %s", len(out), output_path)
    return output_path


# ── CLI entry point ─────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for the preprocessing stage.

    Orchestrates the full pipeline:

    #. Load raw data from ``data/raw/``.
    #. Validate input schema.
    #. Convert UTC timestamps to local time (Europe/Paris).
    #. Convert temperature to °C and filter physical limits.
    #. Detect and flag flatline (stuck sensor) days.
    #. Extract daily peaks with QC (missing data, amplitude, ties).
    #. Flag circular outliers.
    #. Validate output schema.
    #. Save daily peaks to ``data/processed/``.
    """
    config = EpinalPeakConfig()
    epinal_peak.setup_logging(config)

    _logger.info("=" * 60)
    _logger.info("Preprocessing stage — station %s", config.station_id)
    _logger.info("Period: %d–%d", config.year_start, config.year_end)
    _logger.info("=" * 60)

    # 1. Load raw data.
    raw = _load_raw_data(config)

    # 2. Validate input schema.
    raw = validate_input_schema(raw)

    # 3. UTC → local time.
    local = utc_to_local(raw, config.timezone)

    # 4. Physical limits filter.
    local = filter_physical_limits(local, config)

    # 5. Flatline detection.
    local = flag_flatline(local, config)

    # 6. Daily peak extraction (inc. missing-data and amplitude QC).
    peaks = extract_daily_peak_hour(local, config)

    # 7. Circular outlier flagging.
    peaks = flag_circular_outliers(peaks, config)

    # 8. Validate output schema.
    peaks = validate_output_schema(peaks)

    # 9. Save.
    _save_daily_peaks(peaks, config)

    n_valid = int((~peaks["qc_excluded"]).sum())
    n_total = len(peaks)
    _logger.info(
        "Preprocessing complete: %d / %d days valid (%.1f %%)",
        n_valid,
        n_total,
        100.0 * n_valid / n_total if n_total else 0.0,
    )


if __name__ == "__main__":
    main()
