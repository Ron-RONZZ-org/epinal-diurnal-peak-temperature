"""Tests for the data acquisition module (``acquire.py``).

Covers URL helpers, registry management, CSV filtering, schema validation,
local fallback, and the ``DATA_SOURCE`` environment override.
"""

from __future__ import annotations

import gzip
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from epinal_peak import EpinalPeakConfig
from epinal_peak import acquire


# ── Helpers ────────────────────────────────────────────────────────────

def _make_synthetic_csv_gz(
    path: Path,
    station_id: str = "88136001",
    n_other: int = 2,
) -> None:
    """Create a small synthetic ``.csv.gz`` file matching the MF schema.

    Writes *n_other* records for other stations plus 24 records for the
    target station (simulating a full day of hourly data).
    """
    columns = [
        "NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI",
        "AAAAMMJJHH", "RR1", "QRR1", "T", "QT",
    ]
    with gzip.open(path, "wt") as f:
        f.write(";".join(columns) + "\n")

        # Other stations
        for i in range(n_other):
            other_id = f"88{i:05d}"
            f.write(f"{other_id};OTHER;0;0;0;2025010100;0;1;30.0;1\n")

        # Target station – 24 hourly records
        for hour in range(24):
            ts = f"20250601{hour:02d}"
            temp = 15.0 + hour * 0.5  # rising through the day
            f.write(f"{station_id};EPINAL;48.21;6.45;317;{ts};0;1;{temp};1\n")


# ── Tests ──────────────────────────────────────────────────────────────


class TestAcquireModule:
    """Verify the module exports expected symbols."""

    def test_module_importable(self) -> None:
        """All expected public symbols exist."""
        assert hasattr(acquire, "main")
        assert hasattr(acquire, "download_raw_data")
        assert hasattr(acquire, "load_local_fallback")
        assert hasattr(acquire, "validate_raw_schema")
        assert hasattr(acquire, "RawDataSchema")

    def test_constants_defined(self) -> None:
        """Module-level constants are set."""
        assert len(acquire.PERIOD_FILES) >= 5
        assert "NUM_POSTE" in acquire.KEEP_COLUMNS
        assert "T" in acquire.KEEP_COLUMNS


class TestUrlHelpers:
    """URL and filename construction."""

    def test_filename_decade(self) -> None:
        """Decade period produces a plain filename."""
        name = acquire._filename_for_period("1990-1999")
        assert name == "H_88_1990-1999.csv.gz"

    def test_filename_previous(self) -> None:
        """``previous-`` prefix maps correctly."""
        name = acquire._filename_for_period("previous-2020-2024")
        assert "previous-2020-2024" in name

    def test_filename_latest(self) -> None:
        """``latest-`` prefix maps correctly."""
        name = acquire._filename_for_period("latest-2025-2026")
        assert "latest-2025-2026" in name

    def test_url_has_base(self) -> None:
        """URL includes the data.gouv.fr base and the correct filename."""
        url = acquire._url_for_period("2000-2009")
        assert "object.files.data.gouv.fr" in url
        assert "H_88_2000-2009.csv.gz" in url


class TestRegistry:
    """Pooch registry file read/write."""

    def test_load_missing(self) -> None:
        """Missing registry file → empty dict."""
        result = acquire._load_registry(Path("/nonexistent/registry.txt"))
        assert result == {}

    def test_round_trip(self) -> None:
        """Saved registry can be loaded back."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.txt"
            original = {"a.csv": "sha256:abc123", "b.csv": "sha256:def456"}
            acquire._save_registry(path, original)
            loaded = acquire._load_registry(path)
            assert loaded == original

    def test_skip_comments_and_empty(self) -> None:
        """Comments and blank lines are ignored."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.txt"
            path.write_text(
                "# This is a comment\n"
                "\n"
                "data.csv sha256:abcdef\n"
                "# another comment\n"
            )
            loaded = acquire._load_registry(path)
            assert loaded == {"data.csv": "sha256:abcdef"}


class TestReadAndFilter:
    """Filtering station records from departmental CSV.GZ."""

    def test_filters_correct_station(self) -> None:
        """Only the target station's records are kept."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv.gz"
            _make_synthetic_csv_gz(path, station_id="88136001", n_other=3)

            result = acquire._read_and_filter(path, "88136001")
            assert len(result) == 24
            assert list(result["NUM_POSTE"].unique()) == ["88136001"]

    def test_handles_missing_station(self) -> None:
        """Station not in file → empty DataFrame with correct columns."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv.gz"
            _make_synthetic_csv_gz(path, station_id="99999999", n_other=1)

            result = acquire._read_and_filter(path, "88136001")
            assert result.empty
            assert list(result.columns) == list(acquire.KEEP_COLUMNS)

    def test_columns_subset(self) -> None:
        """Result contains only the ``KEEP_COLUMNS``."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv.gz"
            _make_synthetic_csv_gz(path)

            result = acquire._read_and_filter(path, "88136001")
            assert list(result.columns) == list(acquire.KEEP_COLUMNS)


class TestSchemaValidation:
    """Pandera schema validation for raw data."""

    def test_valid_data_passes(self) -> None:
        """Correctly typed data passes validation."""
        df = pd.DataFrame({
            "NUM_POSTE": ["88136001", "88136001"],
            "AAAAMMJJHH": ["2025010100", "2025010101"],
            "T": [50.0, 52.0],
            "QT": [1.0, 1.0],
        })
        result = acquire.validate_raw_schema(df)
        assert len(result) == 2

    def test_empty_df_skips(self) -> None:
        """Empty DataFrame is accepted without error."""
        df = pd.DataFrame(columns=list(acquire.KEEP_COLUMNS))
        result = acquire.validate_raw_schema(df)
        assert result.empty

    def test_rejects_wrong_station(self) -> None:
        """Multiple station IDs cause validation to raise."""
        df = pd.DataFrame({
            "NUM_POSTE": ["88136001", "88136002"],
            "AAAAMMJJHH": ["2025010100", "2025010101"],
            "T": [50.0, 52.0],
            "QT": [1.0, 1.0],
        })
        with pytest.raises(Exception):
            acquire.validate_raw_schema(df)

    def test_rejects_null_temperature(self) -> None:
        """Temperature column with all NaN passes (nullable)."""
        df = pd.DataFrame({
            "NUM_POSTE": ["88136001"],
            "AAAAMMJJHH": ["2025010100"],
            "T": [None],
            "QT": [1.0],
        })
        result = acquire.validate_raw_schema(df)
        assert result["T"].isna().iloc[0]

    def test_temperature_out_of_range(self) -> None:
        """Temperature outside physical limits raises."""
        df = pd.DataFrame({
            "NUM_POSTE": ["88136001"],
            "AAAAMMJJHH": ["2025010100"],
            "T": [9999.0],   # way beyond ±60 °C in 0.1 °C units
            "QT": [1.0],
        })
        with pytest.raises(Exception):
            acquire.validate_raw_schema(df)


class TestLocalFallback:
    """Local fallback path discovery."""

    def test_returns_path(self) -> None:
        """Returns the expected path even if file does not exist."""
        config = EpinalPeakConfig()
        path = acquire.load_local_fallback(config)
        assert isinstance(path, Path)
        assert path.name == config.raw_data_filename

    def test_returns_path_in_fallback_dir(self) -> None:
        """Path is under the external fallback directory."""
        config = EpinalPeakConfig()
        path = acquire.load_local_fallback(config)
        assert str(config.external_dir) in str(path)


class TestDataSourceOverride:
    """``DATA_SOURCE`` environment variable support."""

    def test_override_with_file(self) -> None:
        """Setting ``DATA_SOURCE`` loads from local CSV."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "dummy.csv"
            _make_synthetic_csv_gz(csv_path.with_suffix(".csv.gz"), n_other=0)
            # Decompress for the test (DATA_SOURCE expects plain CSV)
            with gzip.open(csv_path.with_suffix(".csv.gz"), "rt") as gz_f:
                plain = gz_f.read()
            csv_path.write_text(plain)

            os.environ["DATA_SOURCE"] = str(csv_path)
            try:
                config = EpinalPeakConfig()
                result = acquire.download_raw_data(config)
                assert result.exists()
                df = pd.read_csv(result, dtype={"NUM_POSTE": str})
                assert len(df) == 24
                assert list(df["NUM_POSTE"].unique()) == ["88136001"]
            finally:
                del os.environ["DATA_SOURCE"]

    def test_override_missing_file(self) -> None:
        """Missing ``DATA_SOURCE`` path raises ``FileNotFoundError``."""
        os.environ["DATA_SOURCE"] = "/nonexistent/path.csv"
        try:
            config = EpinalPeakConfig()
            with pytest.raises(FileNotFoundError):
                acquire.download_raw_data(config)
        finally:
            del os.environ["DATA_SOURCE"]


class TestMain:
    """CLI entry point."""

    def test_main_runs_with_override(self) -> None:
        """``main()`` executes cleanly with ``DATA_SOURCE`` set."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "source.csv"
            _make_synthetic_csv_gz(csv_path.with_suffix(".csv.gz"), n_other=0)
            with gzip.open(csv_path.with_suffix(".csv.gz"), "rt") as gz_f:
                csv_path.write_text(gz_f.read())

            os.environ["DATA_SOURCE"] = str(csv_path)
            try:
                # Should not raise.
                acquire.main()
            finally:
                del os.environ["DATA_SOURCE"]
