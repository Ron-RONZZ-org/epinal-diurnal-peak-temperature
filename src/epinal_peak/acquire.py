"""Data acquisition via pooch with local fallback.

Downloads hourly temperature records for station MF88136001 (Épinal)
from the Météo-France *Données climatologiques de base - horaires* dataset
on data.gouv.fr.

Data source
-----------
``https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/HOR/``

Department 88 (Vosges) CSV.GZ files are downloaded, then filtered to
keep only station 88136001.  The concatenated result is saved to
``data/raw/epinal_temperature_raw.csv``.

Local fallback
--------------
Set the ``DATA_SOURCE`` environment variable to a local CSV path to
bypass downloading entirely (useful for CI/testing).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame, Series

import epinal_peak
from epinal_peak.config import EpinalPeakConfig

_logger = logging.getLogger(__name__)

# ── Data source configuration ──────────────────────────────────────────

BASE_URL = (
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/HOR"
)
"""Base URL for Météo-France hourly departmental files."""

DEPARTMENT = "88"
"""Vosges department number (Épinal is in department 88)."""

# Period files to download, in chronological order.  Only periods that
# overlap with the station's operational lifetime (1986-06-01 → present)
# are included.  Earlier periods exist (e.g. 1850-1859) but are skipped.
PERIOD_FILES: tuple[str, ...] = (
    "1980-1989",            # Station opened June 1986 → mid-1986 onward
    "1990-1999",
    "2000-2009",
    "2010-2019",
    "previous-2020-2024",   # Current decade (monthly updates, Y-2 threshold)
    "latest-2025-2026",     # Recent years (daily updates for last 2 years)
)
"""Period identifiers corresponding to departmental file suffixes."""

KEEP_COLUMNS: tuple[str, ...] = (
    "NUM_POSTE",     # Station identifier (string)
    "AAAAMMJJHH",    # Timestamp in UTC (YYYYMMDDHH format)
    "T",             # Temperature in 0.1 °C (divide by 10 for °C)
    "QT",            # Quality flag (1 = good)
)
"""Columns extracted from the raw departmental CSV files."""

CHUNK_SIZE: int = 100_000
"""Rows per chunk when reading large CSV.GZ files."""


# ── URL helpers ─────────────────────────────────────────────────────────

def _filename_for_period(period: str) -> str:
    """Build the departmental filename for a given period."""
    if period.startswith("previous-"):
        return f"H_{DEPARTMENT}_previous-{period.removeprefix('previous-')}.csv.gz"
    if period.startswith("latest-"):
        return f"H_{DEPARTMENT}_latest-{period.removeprefix('latest-')}.csv.gz"
    return f"H_{DEPARTMENT}_{period}.csv.gz"


def _url_for_period(period: str) -> str:
    """Build the full download URL for a given period."""
    return f"{BASE_URL}/{_filename_for_period(period)}"


# ── Registry management (pooch hash file) ──────────────────────────────

def _load_registry(path: Path) -> dict[str, str]:
    """Load pooch registry from *path* (filename → ``sha256:...`` hash).

    Returns an empty dict if the file does not exist.
    """
    if not path.exists():
        return {}
    registry: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                registry[parts[0]] = parts[1]
    return registry


def _save_registry(path: Path, registry: dict[str, str]) -> None:
    """Persist a registry dict to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Pooch registry: filename -> sha256:<hash>\n")
        for filename in sorted(registry):
            f.write(f"{filename} {registry[filename]}\n")


def _compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ── Station filtering ──────────────────────────────────────────────────

def _read_and_filter(path: Path, station_id: str) -> pd.DataFrame:
    """Read a departmental CSV.GZ file and keep only the target station.

    Uses chunked reading to keep memory usage bounded on large decade files.

    Args:
        path: Path to a local ``.csv.gz`` file.
        station_id: Target station identifier (e.g. ``"88136001"``).

    Returns:
        DataFrame with only the target station's records.
    """
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        sep=";",
        compression="gzip",
        usecols=list(KEEP_COLUMNS),
        dtype={"NUM_POSTE": str},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        mask = chunk["NUM_POSTE"] == station_id
        station_chunk = chunk.loc[mask]
        if not station_chunk.empty:
            chunks.append(station_chunk)

    if not chunks:
        return pd.DataFrame(columns=list(KEEP_COLUMNS))

    return pd.concat(chunks, ignore_index=True)


# ── Schema validation ──────────────────────────────────────────────────

class RawDataSchema(pa.DataFrameModel):
    """Pandera schema for filtered raw hourly temperature data.

    Validates column presence, dtypes, and basic ranges after the
    station-level filter has been applied.
    """

    NUM_POSTE: Series[str] = pa.Field(
        description="Météo-France RADOME station identifier"
    )
    AAAAMMJJHH: Series[str] = pa.Field(
        description="UTC timestamp in YYYYMMDDHH format",
    )
    T: Series[float] = pa.Field(
        nullable=True,
        ge=-500.0,      # -50.0 °C in 0.1 °C units
        le=600.0,       #  60.0 °C in 0.1 °C units
        description="Hourly temperature in 0.1 °C (÷10 for °C)",
    )
    QT: Series[float] = pa.Field(
        nullable=True,
        ge=0,
        le=9,
        description="Quality flag: 1 = good, 9 = missing/unchecked",
    )

    @pa.dataframe_check
    def station_is_correct(cls, df: DataFrame) -> Series[bool]:
        """All records must share the same station ID."""
        return df["NUM_POSTE"].nunique() == 1


def validate_raw_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the filtered raw data against the expected schema.

    Args:
        df: Raw temperature data filtered for the target station.

    Returns:
        Validated DataFrame (or raises ``pandera.errors.SchemaError``).

    Raises:
        pandera.errors.SchemaError: If validation fails.
    """
    if df.empty:
        _logger.warning("Empty DataFrame passed to schema validation — skipping.")
        return df

    # ── Normalise dtypes before validation ──────────────────────────────
    df = df.copy()
    df["NUM_POSTE"] = df["NUM_POSTE"].astype(str)
    df["AAAAMMJJHH"] = df["AAAAMMJJHH"].astype(str)
    df["T"] = pd.to_numeric(df["T"], errors="coerce").astype("float64")
    df["QT"] = pd.to_numeric(df["QT"], errors="coerce").astype("float64")

    _logger.debug("Running schema validation on %d records …", len(df))
    validated = RawDataSchema.validate(df, lazy=True)
    _logger.info("Schema validation passed: %d records", len(validated))
    return validated


# ── Download / fallback logic ──────────────────────────────────────────

def download_raw_data(config: EpinalPeakConfig) -> Path:
    """Download and filter hourly temperature data for the configured station.

    Behaviour
    ---------
    1. If ``DATA_SOURCE`` env var is set → load and filter that local CSV.
    2. Otherwise → download departmental CSV.GZ files via **pooch**,
       filter for the configured station, and concatenate.
    3. Validate the merged result against ``RawDataSchema``.
    4. Write to ``data/raw/<raw_data_filename>``.

    Args:
        config: Pipeline configuration including station ID, registry
            file path, and output directory.

    Returns:
        Path to the concatenated raw CSV file.
    """
    # ── Local fallback via DATA_SOURCE ──────────────────────────────────
    data_source = os.environ.get("DATA_SOURCE")
    if data_source is not None:
        return _load_from_source(data_source, config)

    # ── Download via pooch ─────────────────────────────────────────────
    import pooch

    registry_path = config.external_dir / config.pooch_registry_file
    registry = _load_registry(registry_path)

    # Pooch cache directory (inside raw_dir to keep data co-located).
    cache_dir = config.raw_dir / ".pooch_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pup = pooch.create(
        path=cache_dir,
        base_url=BASE_URL,
        registry={},
        env="DATA_SOURCE",
    )

    frames: list[pd.DataFrame] = []

    for period in PERIOD_FILES:
        filename = _filename_for_period(period)
        url = _url_for_period(period)
        known_hash = registry.get(filename)

        _logger.info("Downloading %s …", filename)
        local_path = Path(
            pooch.retrieve(
                url=url,
                known_hash=known_hash,
                fname=filename,
                path=pup.path,
                downloader=pooch.HTTPDownloader(progressbar=False),
            )
        )

        # First download → compute and persist hash.
        if filename not in registry:
            file_hash = _compute_sha256(local_path)
            registry[filename] = f"sha256:{file_hash}"
            _logger.info("Registered hash for %s", filename)

        _logger.info("Filtering %s …", filename)
        df_station = _read_and_filter(local_path, config.station_id)
        _logger.info(
            "  → %s: %d records for station %s",
            filename,
            len(df_station),
            config.station_id,
        )
        frames.append(df_station)

    # Persist updated registry.
    _save_registry(registry_path, registry)

    # Concatenate, sort, and validate.
    df_all = pd.concat(frames, ignore_index=True)
    _logger.info(
        "Total: %d raw records across %d period files",
        len(df_all),
        len(PERIOD_FILES),
    )

    if df_all.empty:
        raise ValueError(
            f"No data found for station {config.station_id} "
            f"in any of the downloaded files."
        )

    df_all = df_all.sort_values("AAAAMMJJHH").reset_index(drop=True)
    df_all = validate_raw_schema(df_all)

    return _save_output(df_all, config)


def _load_from_source(source: str, config: EpinalPeakConfig) -> Path:
    """Load data from a local file path (``DATA_SOURCE`` override).

    The file is expected to be a semicolon-delimited CSV matching the
    Météo-France hourly schema (or a subset containing at least the
    ``KEEP_COLUMNS``).
    """
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(
            f"DATA_SOURCE is set to '{source}' but the file does not exist."
        )
    _logger.info("DATA_SOURCE override: loading from %s", source_path)

    df = pd.read_csv(
        source_path,
        sep=";",
        usecols=list(KEEP_COLUMNS),
        dtype={"NUM_POSTE": str},
        low_memory=False,
    )
    df = df[df["NUM_POSTE"] == config.station_id].copy()
    df = df.sort_values("AAAAMMJJHH").reset_index(drop=True)
    df = validate_raw_schema(df)
    return _save_output(df, config)


def _save_output(df: pd.DataFrame, config: EpinalPeakConfig) -> Path:
    """Write the filtered DataFrame to the raw data directory."""
    output_path = config.raw_dir / config.raw_data_filename
    config.raw_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    _logger.info("Saved %d records → %s", len(df), output_path)
    return output_path


# ── Legacy fallback (kept for compatibility) ───────────────────────────

def load_local_fallback(config: EpinalPeakConfig) -> Path:
    """Locate the local fallback data file.

    This is a lower-priority fallback used when ``DATA_SOURCE`` is not
    set and the remote download fails.  The file is expected at
    ``data/external/fallback/<raw_data_filename>``.

    Args:
        config: Pipeline configuration.

    Returns:
        Path to the fallback file (may not exist — caller must check).
    """
    path = config.external_dir / config.local_fallback_dir / config.raw_data_filename
    if not path.exists():
        _logger.warning("Local fallback not found at %s", path)
    else:
        _logger.info("Local fallback found at %s", path)
    return path


# ── CLI entry point ────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for the data acquisition stage.

    Logs configuration details and invokes :func:`download_raw_data`.
    """
    config = EpinalPeakConfig()
    epinal_peak.setup_logging(config)

    _logger.info("=" * 60)
    _logger.info("Data acquisition — station %s", config.station_id)
    _logger.info("Period: %d–%d (%d years)", config.year_start,
                 config.year_end, config.year_end - config.year_start + 1)
    _logger.info("Data source: Météo-France HOR (data.gouv.fr)")
    _logger.info("=" * 60)

    output_path = download_raw_data(config)

    n_records = 0
    try:
        n_records = len(pd.read_csv(output_path))
    except Exception:
        pass

    _logger.info("Acquisition complete: %s (%d records)", output_path, n_records)


if __name__ == "__main__":
    main()
