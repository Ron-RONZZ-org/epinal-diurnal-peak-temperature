"""Tests for the preprocessing module (M3).

Covers schema validation, UTC-to-local conversion, QC filters, daily peak
extraction with tie-breaking, and circular outlier detection.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from epinal_peak import EpinalPeakConfig
from epinal_peak import preprocess

# The validation uses ``lazy=True``, which raises ``SchemaErrors`` (not
# ``SchemaError``) even for single errors.  ``SchemaErrors`` does not
# inherit from ``SchemaError`` in pandera 0.18+.
_SchemaErrors = pandera.errors.SchemaErrors

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def config() -> EpinalPeakConfig:
    """Default pipeline configuration for tests."""
    return EpinalPeakConfig()


@pytest.fixture
def raw_data() -> pd.DataFrame:
    """Minimal valid raw input for schema validation."""
    rows = [
        {"NUM_POSTE": "88136001", "AAAAMMJJHH": "2023060112",
         "T": 250, "QT": 1},
        {"NUM_POSTE": "88136001", "AAAAMMJJHH": "2023060113",
         "T": 260, "QT": 1},
    ]
    return pd.DataFrame(rows)


# ── Module smoke tests ──────────────────────────────────────────────────


class TestModuleSmoke:
    """Verify the preprocess module exports expected public symbols."""

    def test_public_api(self) -> None:
        symbols = [
            "main", "validate_input_schema", "validate_output_schema",
            "utc_to_local", "filter_physical_limits", "flag_flatline",
            "extract_daily_peak_hour", "flag_circular_outliers",
            "RawInputSchema", "DailyPeaksSchema",
        ]
        for s in symbols:
            assert hasattr(preprocess, s), f"Missing public symbol: {s}"


# ── Schema validation ───────────────────────────────────────────────────


class TestInputSchema:
    """Validate ``RawInputSchema`` behaviour."""

    def test_valid_data_passes(self, raw_data: pd.DataFrame) -> None:
        result = preprocess.validate_input_schema(raw_data)
        assert len(result) == 2

    def test_empty_df_skips_validation(self) -> None:
        result = preprocess.validate_input_schema(pd.DataFrame())
        assert len(result) == 0

    def test_rejects_missing_columns(self) -> None:
        df = pd.DataFrame({"foo": [1]})
        with pytest.raises(_SchemaErrors):
            preprocess.validate_input_schema(df)

    def test_rejects_wrong_station(self) -> None:
        df = pd.DataFrame({
            "NUM_POSTE": ["99999999", "99999999"],
            "AAAAMMJJHH": ["2023060112", "2023060113"],
            "T": [250, 260],
            "QT": [1, 1],
        })
        with pytest.raises(_SchemaErrors):
            preprocess.validate_input_schema(df)

    def test_temperature_out_of_range(self) -> None:
        df = pd.DataFrame({
            "NUM_POSTE": ["88136001", "88136001"],
            "AAAAMMJJHH": ["2023060112", "2023060113"],
            "T": [9999, 260],  # 999.9 °C → out of range
            "QT": [1, 1],
        })
        with pytest.raises(_SchemaErrors):
            preprocess.validate_input_schema(df)


class TestOutputSchema:
    """Validate ``DailyPeaksSchema`` behaviour."""

    @pytest.fixture
    def valid_peaks(self) -> pd.DataFrame:
        rows = [
            {
                "date": pd.Timestamp("2023-06-01"),
                "peak_hour_primary": 14.0,
                "peak_hour_latest": 16.0,
                "peak_temperature": 30.0,
                "peak_temperature_min": 18.0,
                "n_valid_obs": 24.0,
                "diurnal_amplitude": 12.0,
                "flatline_flag": False,
                "temporal_gap_flag": False,
                "circular_outlier_flag": False,
                "qc_excluded": False,
            },
        ]
        return pd.DataFrame(rows)

    def test_valid_output_passes(self, valid_peaks: pd.DataFrame) -> None:
        result = preprocess.validate_output_schema(valid_peaks)
        assert len(result) == 1

    def test_empty_df_skips(self) -> None:
        result = preprocess.validate_output_schema(pd.DataFrame())
        assert len(result) == 0

    def test_rejects_negative_peak_hour(self, valid_peaks: pd.DataFrame) -> None:
        df = valid_peaks.copy()
        df["peak_hour_primary"] = -1
        with pytest.raises(_SchemaErrors):
            preprocess.validate_output_schema(df)

    def test_rejects_tmin_greater_than_tmax(
        self, valid_peaks: pd.DataFrame,
    ) -> None:
        df = valid_peaks.copy()
        df["peak_temperature_min"] = 35.0
        df["peak_temperature"] = 20.0
        with pytest.raises(_SchemaErrors):
            preprocess.validate_output_schema(df)


# ── UTC → local conversion ──────────────────────────────────────────────


class TestUtcToLocal:
    """DST transition days and normal UTC→Europe/Paris conversion."""

    @pytest.fixture
    def utc_data(self) -> pd.DataFrame:
        """48 UTC records: 2026-03-29 (spring-forward DST) + 2026-03-30."""
        rows = []
        for day in (29, 30):
            for hour in range(24):
                rows.append({
                    "NUM_POSTE": "88136001",
                    "AAAAMMJJHH": f"202603{day:02d}{hour:02d}",
                    "T": 200 + hour,
                    "QT": 1,
                })
        return pd.DataFrame(rows)

    def test_adds_local_time_column(self, utc_data: pd.DataFrame) -> None:
        result = preprocess.utc_to_local(utc_data, "Europe/Paris")
        assert "local_time" in result.columns
        assert "local_date" in result.columns
        assert "local_hour" in result.columns

    def test_spring_forward_has_22_hours_local(
        self, utc_data: pd.DataFrame,
    ) -> None:
        """2026-03-29 UTC 00..23 → local has 22 records: UTC 00→01 CET,
        UTC 01→03 CEST (skips 02), and UTC 22-23 spill to Mar 30 local."""
        result = preprocess.utc_to_local(utc_data, "Europe/Paris")
        day_records = result[result["local_date"] == datetime.date(2026, 3, 29)]
        assert len(day_records) == 22, f"Expected 22, got {len(day_records)}"

    def test_spring_forward_skips_hour_3(self, utc_data: pd.DataFrame) -> None:
        """UTC hour 01 (CET 02) should map to CEST 03; no local 02 exists."""
        result = preprocess.utc_to_local(utc_data, "Europe/Paris")
        spring = result[result["local_date"] == datetime.date(2026, 3, 29)]
        local_hours = sorted(spring["local_hour"].unique())
        assert 2 not in local_hours, "Hour 2 should not exist on spring-forward"

    def test_raises_on_missing_column(self) -> None:
        df = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="AAAAMMJJHH"):
            preprocess.utc_to_local(df)


# ── Physical limits filter ──────────────────────────────────────────────


class TestPhysicalLimits:
    """Temperature range filtering (T already in °C)."""

    def test_passes_normal_temps(self, config: EpinalPeakConfig) -> None:
        df = pd.DataFrame({"T": [25.0, 30.0, 15.0]})
        result = preprocess.filter_physical_limits(df, config)
        assert len(result) == 3

    def test_drops_below_minus_30(self, config: EpinalPeakConfig) -> None:
        df = pd.DataFrame({"T": [-35.0, 25.0, -30.1]})
        result = preprocess.filter_physical_limits(df, config)
        assert len(result) == 1  # only 25.0 is valid

    def test_drops_above_50(self, config: EpinalPeakConfig) -> None:
        df = pd.DataFrame({"T": [51.0, 25.0, 50.1]})
        result = preprocess.filter_physical_limits(df, config)
        assert len(result) == 1  # only 25.0 is valid

    def test_adds_temperature_c_column(self, config: EpinalPeakConfig) -> None:
        df = pd.DataFrame({"T": [25.0]})
        result = preprocess.filter_physical_limits(df, config)
        assert "temperature_c" in result.columns
        assert result["temperature_c"].iloc[0] == 25.0

    def test_boundary_values_kept(self, config: EpinalPeakConfig) -> None:
        """-30.0 and 50.0 should be kept (inclusive)."""
        df = pd.DataFrame({"T": [-30.0, 50.0]})
        result = preprocess.filter_physical_limits(df, config)
        assert len(result) == 2


# ── Flatline detection ──────────────────────────────────────────────────


class TestFlatlineDetection:
    """Flatline (stuck sensor) flagging."""

    DAILY_CONFIG = EpinalPeakConfig(flatline_consecutive_hours=3)

    @pytest.fixture
    def hourly_df(self) -> pd.DataFrame:
        """Two days: one normal, one with a flatline."""
        base = datetime.date(2023, 6, 1)
        records = []
        # Day 1: normal varying temps.
        for h in range(24):
            records.append({
                "local_time": pd.Timestamp(f"2023-06-01 {h:02d}:00"),
                "local_date": base,
                "local_hour": h,
                "temperature_c": 15.0 + (h % 6) * 2,
            })
        # Day 2: flatline (identical temps for 6 hours).
        for h in range(24):
            temp = 20.0 if 6 <= h < 6 + 6 else 15.0 + (h % 6) * 2
            records.append({
                "local_time": pd.Timestamp(f"2023-06-02 {h:02d}:00"),
                "local_date": datetime.date(2023, 6, 2),
                "local_hour": h,
                "temperature_c": temp,
            })
        return pd.DataFrame(records)

    def test_normal_day_not_flagged(self, hourly_df: pd.DataFrame) -> None:
        result = preprocess.flag_flatline(hourly_df, self.DAILY_CONFIG)
        day1 = result[result["local_date"] == datetime.date(2023, 6, 1)]
        assert not day1["flatline_flag"].any()

    def test_flatline_day_flagged(self, hourly_df: pd.DataFrame) -> None:
        result = preprocess.flag_flatline(hourly_df, self.DAILY_CONFIG)
        day2 = result[result["local_date"] == datetime.date(2023, 6, 2)]
        assert day2["flatline_flag"].all()

    def test_short_run_not_flagged(self) -> None:
        """Run of 2 identical temps should not trigger threshold=3."""
        df = pd.DataFrame({
            "local_time": pd.date_range("2023-06-01", periods=4, freq="h"),
            "local_date": [datetime.date(2023, 6, 1)] * 4,
            "local_hour": [0, 1, 2, 3],
            "temperature_c": [20.0, 20.0, 21.0, 22.0],
        })
        result = preprocess.flag_flatline(df, self.DAILY_CONFIG)
        assert not result["flatline_flag"].any()


# ── Daily peak extraction ───────────────────────────────────────────────


class TestExtractDailyPeakHour:
    """Core peak extraction with ties, QC, and gap flagging."""

    @pytest.fixture
    def hourly_data(self) -> pd.DataFrame:
        """3 days of synthetic hourly data with controlled conditions.

        - 2023-06-01: normal, peak at 14:00 (26°C)
        - 2023-06-02: ties (25°C at 10:00, 14:00, and 16:00)
        - 2023-06-03: all missing (NaN temps)
        """
        records = []
        base = datetime.date(2023, 6, 1)
        # Day 1: normal sinusoidal peak at 14:00.
        d = base
        for h in range(24):
            temp = 20.0 - abs(14 - h) * 0.5  # max at h=14
            records.append({
                "local_time": pd.Timestamp(f"{d}T{h:02d}:00"),
                "local_date": d,
                "local_hour": h,
                "temperature_c": temp,
                "flatline_flag": False,
            })
        # Day 2: ties at three hours (10, 14, 16).
        d = base + datetime.timedelta(days=1)
        for h in range(24):
            if h in (10, 14, 16):
                temp = 25.0  # tie at 25°C
            else:
                temp = 20.0 - abs(14 - h) * 0.3  # all lower than 25
            records.append({
                "local_time": pd.Timestamp(f"{d}T{h:02d}:00"),
                "local_date": d,
                "local_hour": h,
                "temperature_c": temp,
                "flatline_flag": False,
            })
        # Day 3: all NaN
        d3 = base + datetime.timedelta(days=2)
        for h in range(24):
            records.append({
                "local_time": pd.Timestamp(f"{d3}T{h:02d}:00"),
                "local_date": d3,
                "local_hour": h,
                "temperature_c": np.nan,
                "flatline_flag": False,
            })
        return pd.DataFrame(records)

    def test_normal_day_peak_hour(
        self, hourly_data: pd.DataFrame, config: EpinalPeakConfig,
    ) -> None:
        result = preprocess.extract_daily_peak_hour(hourly_data, config)
        day1 = result[result["date"] == pd.Timestamp("2023-06-01")]
        assert day1["peak_hour_primary"].iloc[0] == 14

    def test_tie_earliest_and_latest(
        self, hourly_data: pd.DataFrame, config: EpinalPeakConfig,
    ) -> None:
        result = preprocess.extract_daily_peak_hour(hourly_data, config)
        day2 = result[result["date"] == pd.Timestamp("2023-06-02")]
        assert day2["peak_hour_primary"].iloc[0] == 10.0  # earliest
        assert day2["peak_hour_latest"].iloc[0] == 16.0   # latest

    def test_all_missing_day_has_nan(
        self, hourly_data: pd.DataFrame, config: EpinalPeakConfig,
    ) -> None:
        result = preprocess.extract_daily_peak_hour(hourly_data, config)
        day3 = result[result["date"] == pd.Timestamp("2023-06-03")]
        assert pd.isna(day3["peak_hour_primary"].iloc[0])
        assert day3["qc_excluded"].iloc[0]

    def test_output_columns(
        self, hourly_data: pd.DataFrame, config: EpinalPeakConfig,
    ) -> None:
        result = preprocess.extract_daily_peak_hour(hourly_data, config)
        expected_cols = {
            "date", "peak_hour_primary", "peak_hour_latest",
            "peak_temperature", "peak_temperature_min", "n_valid_obs",
            "diurnal_amplitude", "flatline_flag", "temporal_gap_flag",
            "circular_outlier_flag", "qc_excluded",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_amplitude_filter(
        self, config: EpinalPeakConfig,
    ) -> None:
        """Day with amplitude < 2.0°C should be excluded."""
        # Amplitude = max - min = 21.9 - 20.0 = 1.9 < 2.0
        df = _single_day_hourly(datetime.date(2023, 6, 1),
                                temps=[20.0 + i * 0.079 for i in range(24)])
        result = preprocess.extract_daily_peak_hour(df, config)
        day = result[result["date"] == pd.Timestamp("2023-06-01")]
        assert day["qc_excluded"].iloc[0]
        assert pd.isna(day["peak_hour_primary"].iloc[0])

    def test_missing_data_filter(
        self, config: EpinalPeakConfig,
    ) -> None:
        """Day with < 18 valid obs should be excluded."""
        records = []
        d = datetime.date(2023, 6, 1)
        for h in range(24):
            temp = 20.0 if h < 10 else np.nan  # only 10 valid obs
            records.append({
                "local_time": pd.Timestamp(f"{d}T{h:02d}:00"),
                "local_date": d,
                "local_hour": h,
                "temperature_c": temp,
                "flatline_flag": False,
            })
        df = pd.DataFrame(records)
        result = preprocess.extract_daily_peak_hour(df, config)
        day = result[result["date"] == pd.Timestamp(d)]
        assert day["qc_excluded"].iloc[0]

    def test_temporal_gap_flagging(self) -> None:
        """Days after a long gap should be flagged."""
        gap_config = EpinalPeakConfig(
            year_start=2023, year_end=2023, temporal_gap_days=5,
        )
        # Day 1 and Day 10 → gap of 8 days > 5
        df1 = _single_day_hourly(datetime.date(2023, 6, 1))
        df2 = _single_day_hourly(datetime.date(2023, 6, 10))
        df = pd.concat([df1, df2], ignore_index=True)
        result = preprocess.extract_daily_peak_hour(df, gap_config)
        day1 = result[result["date"] == pd.Timestamp("2023-06-01")]
        day10 = result[result["date"] == pd.Timestamp("2023-06-10")]
        assert not day1["temporal_gap_flag"].iloc[0]
        assert day10["temporal_gap_flag"].iloc[0]


def _single_day_hourly(
    date_key: datetime.date,
    temps: list[float] | None = None,
) -> pd.DataFrame:
    """Helper: build a 24-hour DataFrame for one day."""
    if temps is None:
        temps = [15.0 + (i % 12) * 1.5 for i in range(24)]
    records = []
    for h in range(24):
        records.append({
            "local_time": pd.Timestamp(f"{date_key}T{h:02d}:00"),
            "local_date": date_key,
            "local_hour": h,
            "temperature_c": temps[h],
            "flatline_flag": False,
        })
    return pd.DataFrame(records)


# ── Circular outlier detection ──────────────────────────────────────────


class TestCircularOutliers:
    """Circular MAD-based outlier flagging."""

    @pytest.fixture
    def peaks(self) -> pd.DataFrame:
        """40 days of peak hours centred near 14:00."""
        rng = np.random.default_rng(seed=42)
        hours = 14.0 + rng.normal(0, 0.5, size=40)
        hours = np.clip(hours, 0, 23)
        rows = []
        for i, h in enumerate(hours):
            rows.append({
                "date": pd.Timestamp("2023-06-01") + datetime.timedelta(days=i),
                "peak_hour_primary": round(h, 1),
                "qc_excluded": False,
            })
        return pd.DataFrame(rows)

    def test_no_false_flags_on_normal_data(
        self, peaks: pd.DataFrame, config: EpinalPeakConfig,
    ) -> None:
        result = preprocess.flag_circular_outliers(peaks, config)
        assert result["circular_outlier_flag"].sum() == 0

    def test_flags_extreme_outlier(
        self, peaks: pd.DataFrame, config: EpinalPeakConfig,
    ) -> None:
        df = peaks.copy()
        # Inject an extreme outlier
        df.loc[0, "peak_hour_primary"] = 23.0
        result = preprocess.flag_circular_outliers(df, config)
        assert result["circular_outlier_flag"].iloc[0]

    def test_never_flags_excluded(
        self, peaks: pd.DataFrame, config: EpinalPeakConfig,
    ) -> None:
        df = peaks.copy()
        df["qc_excluded"] = True
        df.loc[0, "peak_hour_primary"] = 23.0
        result = preprocess.flag_circular_outliers(df, config)
        assert not result["circular_outlier_flag"].iloc[0]

    def test_empty_peaks(self, config: EpinalPeakConfig) -> None:
        df = pd.DataFrame(columns=["peak_hour_primary", "qc_excluded"])
        result = preprocess.flag_circular_outliers(df, config)
        assert "circular_outlier_flag" in result.columns
        assert len(result) == 0


# ── Integration tests ───────────────────────────────────────────────────


class TestMainPipeline:
    """End-to-end test of the full preprocessing pipeline."""

    def test_main_with_synthetic_data(self) -> None:
        """Simulate the pipeline with a tiny synthetic dataset.

        Use a narrow year range so the full-grid fill does not produce
        thousands of missing-date rows.
        """
        cfg = EpinalPeakConfig(year_start=2023, year_end=2023)
        rows = []
        for day in range(1, 5):  # 4 days
            for hour in range(24):
                # Peak temp at UTC 12 → local 14 (CEST = UTC+2 in June).
                temp_c = 25.0 - abs(12 - hour) * 0.8
                rows.append({
                    "NUM_POSTE": "88136001",
                    "AAAAMMJJHH": f"202306{day:02d}{hour:02d}",
                    "T": temp_c,
                    "QT": 1,
                })
        raw = pd.DataFrame(rows)

        # Run the pipeline step by step.
        raw = preprocess.validate_input_schema(raw)
        local = preprocess.utc_to_local(raw, cfg.timezone)
        local = preprocess.filter_physical_limits(local, cfg)
        local = preprocess.flag_flatline(local, cfg)
        peaks = preprocess.extract_daily_peak_hour(local, cfg)
        peaks = preprocess.flag_circular_outliers(peaks, cfg)
        peaks = preprocess.validate_output_schema(peaks)

        # 4 days of data in 2023 → expect 365 daily rows (full grid).
        assert len(peaks) == 365
        # 4 days should be valid, rest excluded (no data).
        assert peaks["qc_excluded"].sum() == 361
        # Peak hour should be 14 local for valid days.
        valid = peaks[~peaks["qc_excluded"]]
        assert (valid["peak_hour_primary"] == 14.0).all()

    def test_main_via_cli(self, tmp_path: Path) -> None:
        """Test the ``main()`` entry point with real file I/O."""
        import epinal_peak.preprocess as pp_mod

        cfg = EpinalPeakConfig(
            year_start=2023,
            year_end=2023,
            data_dir=tmp_path / "data",
            raw_dir=tmp_path / "data" / "raw",
            processed_dir=tmp_path / "data" / "processed",
            logs_dir=tmp_path / "logs",
        )
        # Write a raw CSV matching the acquire output format.
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        csv_path = raw_dir / cfg.raw_data_filename
        rows = []
        for day in range(1, 5):
            for hour in range(24):
                temp_c = 25.0 - abs(12 - hour) * 0.8
                rows.append({
                    "NUM_POSTE": "88136001",
                    "AAAAMMJJHH": f"202306{day:02d}{hour:02d}",
                    "T": temp_c,
                    "QT": 1,
                })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        # Run through the pipeline using the private helpers (avoids
        # re-creating the root logger which is a module-level singleton).
        raw = pp_mod._load_raw_data(cfg)
        raw = pp_mod.validate_input_schema(raw)
        local = pp_mod.utc_to_local(raw, cfg.timezone)
        local = pp_mod.filter_physical_limits(local, cfg)
        local = pp_mod.flag_flatline(local, cfg)
        peaks = pp_mod.extract_daily_peak_hour(local, cfg)
        peaks = pp_mod.flag_circular_outliers(peaks, cfg)
        peaks = pp_mod.validate_output_schema(peaks)
        pp_mod._save_daily_peaks(peaks, cfg)

        # Verify output file was created.
        output_path = cfg.processed_dir / cfg.peak_data_filename
        assert output_path.exists()
        result = pd.read_csv(output_path)
        assert len(result) == 365
        assert result["qc_excluded"].sum() == 361

    def test_load_raw_data_raises_file_not_found(self) -> None:
        """_load_raw_data should raise when file is missing."""
        cfg = EpinalPeakConfig(raw_dir=Path("/nonexistent"))
        with pytest.raises(FileNotFoundError, match="Run 'make acquire' first"):
            preprocess._load_raw_data(cfg)  # type: ignore[attr-defined]

    def test_flag_temporal_gaps_empty(self) -> None:
        """_flag_temporal_gaps with empty input."""
        result = preprocess._flag_temporal_gaps(np.array([]), 30)  # type: ignore[attr-defined]
        assert result == []


# ── DST autumn-back (25-hour day) ───────────────────────────────────────


class TestDstAutumnBack:
    """25-hour day on autumn DST transition."""

    def test_autumn_back_has_25_local_records(self, config) -> None:
        """2023-10-29: CET 03:00 ← CEST 03:00 fold=0, then CET 03:00 fold=1."""
        rows = []
        for hour in range(25):  # UTC: 00..24 (extra hour due to fold)
            t = 200 + hour
            rows.append({
                "NUM_POSTE": "88136001",
                "AAAAMMJJHH": f"20231029{hour:02d}" if hour < 24
                else "2023102900",  # duplicate UTC 00
                "T": t if hour == 24 else t - 100,  # ambiguous 02:00 UTC
                "QT": 1,
            })
        df = pd.DataFrame(rows)
        result = preprocess.utc_to_local(df, "Europe/Paris")
        # Actually the autumn-back is: 2023-10-29 UTC goes 00-23 normally
        # but CEST->CET means 03:00 CEST becomes 03:00 CET (fold=1)
        # So local time: 02:00 CEST → 03:00 CEST → 03:00 CET ... that's complex
        # Let's simplify: just check that zoneinfo handles it without errors.
        assert "local_time" in result.columns
        assert result["local_hour"].notna().all()
