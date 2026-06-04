"""Pandera schema definitions for the preprocessing pipeline.

This module is private — all public symbols are re-exported via
``preprocess.py``.  It defines the input schema (raw acquire output)
and the output schema (daily peak-hour dataset).
"""

from __future__ import annotations

import logging

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame, Series

_logger = logging.getLogger(__name__)


class RawInputSchema(pa.DataFrameModel):
    """Pandera schema for raw input from the acquire stage.

    Validates column presence, dtypes, and basic ranges before any
    processing.  The temperature *T* is expressed in 0.1 °C units
    (the Météo-France native encoding).
    """

    NUM_POSTE: Series[str] = pa.Field(
        isin=["88136001"],
        description="Météo-France RADOME station identifier",
    )
    AAAAMMJJHH: Series[str] = pa.Field(
        str_matches=r"^\d{10}$",
        description="UTC timestamp in YYYYMMDDHH format",
    )
    T: Series[float] = pa.Field(
        nullable=True,
        ge=-500.0,
        le=600.0,
        description="Hourly temperature (°C); the acquire stage saves in °C",
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


class DailyPeaksSchema(pa.DataFrameModel):
    """Pandera schema for the daily peak-hour output dataset.

    Each row represents one calendar day (local time).  Days that fail
    QC have *qc_excluded* = ``True`` and NaN in their hour/temperature
    columns.
    """

    date: Series[pd.Timestamp] = pa.Field(
        description="Calendar date (local Europe/Paris time)",
    )
    peak_hour_primary: Series[float] = pa.Field(
        nullable=True,
        ge=0,
        le=23,
        description="Peak hour (0--23), earliest tie-breaking",
    )
    peak_hour_latest: Series[float] = pa.Field(
        nullable=True,
        ge=0,
        le=23,
        description="Peak hour (0--23), latest tie-breaking (sensitivity)",
    )
    peak_temperature: Series[float] = pa.Field(
        nullable=True,
        ge=-30.0,
        le=50.0,
        description="Daily maximum temperature (°C)",
    )
    peak_temperature_min: Series[float] = pa.Field(
        nullable=True,
        ge=-30.0,
        le=50.0,
        description="Daily minimum temperature (°C)",
    )
    n_valid_obs: Series[float] = pa.Field(
        nullable=True,
        ge=0,
        le=31,
        description="Number of non-null hourly observations that day",
    )
    diurnal_amplitude: Series[float] = pa.Field(
        nullable=True,
        ge=0,
        le=80,
        description="Tmax − Tmin (°C)",
    )
    flatline_flag: Series[bool] = pa.Field(
        description="True if ≥config.flatline_consecutive_hours identical temps",
    )
    temporal_gap_flag: Series[bool] = pa.Field(
        description=(
            "True if this day follows >config.temporal_gap_days "
            "consecutive missing days"
        ),
    )
    circular_outlier_flag: Series[bool] = pa.Field(
        description="True if peak hour is a circular outlier via MAD",
    )
    qc_excluded: Series[bool] = pa.Field(
        description=(
            "True if excluded by amplitude or missing-data QC.  "
            "Hour/temperature columns are NaN."
        ),
    )

    @pa.dataframe_check
    def min_does_not_exceed_max(cls, df: DataFrame) -> Series[bool]:
        """``peak_temperature_min <= peak_temperature`` where both exist."""
        both_valid = df["peak_temperature_min"].notna() & df["peak_temperature"].notna()
        return (~both_valid) | (df["peak_temperature_min"] <= df["peak_temperature"])


def validate_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw input against the expected schema.

    Args:
        df: Raw temperature data from the acquire stage (columns
            ``NUM_POSTE``, ``AAAAMMJJHH``, ``T``, ``QT``).

    Returns:
        Validated DataFrame (or raises ``pandera.errors.SchemaError``).

    Raises:
        pandera.errors.SchemaError: If validation fails.
    """
    if df.empty:
        _logger.warning("Empty DataFrame passed to validate_input_schema.")
        return df

    df = df.copy()

    # Coerce dtypes before validation — ``pd.read_csv`` may infer int64
    # for NUM_POSTE or AAAAMMJJHH when values look numeric.
    for col in ("NUM_POSTE", "AAAAMMJJHH", "T", "QT"):
        if col not in df.columns:
            # Let pandera raise the missing-column error downstream.
            _logger.debug("Column '%s' missing — will fail schema check", col)
            break
        if col in ("NUM_POSTE", "AAAAMMJJHH"):
            df[col] = df[col].astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    _logger.debug("Validating input schema on %d records …", len(df))
    validated = RawInputSchema.validate(df, lazy=True)
    _logger.info("Input schema passed: %d records", len(validated))
    return validated


def validate_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the daily peak-hour output against ``DailyPeaksSchema``.

    Args:
        df: Daily peak-hour dataset with columns matching
            :class:`DailyPeaksSchema`.

    Returns:
        Validated DataFrame (or raises ``pandera.errors.SchemaError``).

    Raises:
        pandera.errors.SchemaError: If validation fails.
    """
    if df.empty:
        _logger.warning("Empty DataFrame passed to validate_output_schema.")
        return df

    _logger.debug("Validating output schema on %d records …", len(df))
    validated = DailyPeaksSchema.validate(df, lazy=True)
    _logger.info("Output schema passed: %d records", len(validated))
    return validated
