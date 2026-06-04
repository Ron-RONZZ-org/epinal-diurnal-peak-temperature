"""Tests for the preprocessing module (M0 placeholder)."""

from __future__ import annotations

import pandas as pd
import pytest

from epinal_peak import EpinalPeakConfig
from epinal_peak import preprocess


class TestPreprocessingModule:
    """Basic import and smoke tests for preprocess.py."""

    def test_module_importable(self) -> None:
        """Verify the preprocess module can be imported and has expected symbols."""
        assert hasattr(preprocess, "main")
        assert hasattr(preprocess, "validate_input_schema")
        assert hasattr(preprocess, "utc_to_local")
        assert hasattr(preprocess, "qc_filter")
        assert hasattr(preprocess, "extract_daily_peak_hour")
        assert hasattr(preprocess, "validate_output_schema")

    def test_main_runs_without_error(self) -> None:
        """``main()`` should execute and return None (placeholder)."""
        result = preprocess.main()
        assert result is None

    def test_validate_input_schema_returns_df(self) -> None:
        """Placeholder returns the input DataFrame unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = preprocess.validate_input_schema(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_extract_daily_peak_hour_returns_empty(self) -> None:
        """Placeholder returns an empty DataFrame with expected columns."""
        df = pd.DataFrame({"temp": [10.0, 12.0]})
        result = preprocess.extract_daily_peak_hour(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["date", "peak_hour", "peak_temperature"]

    def test_placeholder_uses_config(self) -> None:
        """Smoke test: pass config to qc_filter."""
        config = EpinalPeakConfig()
        df = pd.DataFrame({"temp": [10.0]})
        result = preprocess.qc_filter(df, config)
        assert result is df  # identity return for placeholder
