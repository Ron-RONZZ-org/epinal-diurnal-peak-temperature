"""Tests for the analysis module (M0 placeholder)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epinal_peak import EpinalPeakConfig
from epinal_peak import analysis


class TestAnalysisModule:
    """Basic import and smoke tests for analysis.py."""

    def test_module_importable(self) -> None:
        """Verify the analysis module can be imported and has expected symbols."""
        assert hasattr(analysis, "main")
        assert hasattr(analysis, "circular_mean")
        assert hasattr(analysis, "circular_std")
        assert hasattr(analysis, "von_mises_regression")
        assert hasattr(analysis, "bootstrap_trend")
        assert hasattr(analysis, "sensitivity_analysis")

    def test_main_runs_without_error(self) -> None:
        """``main()`` should execute and return None (placeholder)."""
        result = analysis.main()
        assert result is None

    def test_circular_mean_returns_float(self) -> None:
        """Placeholder returns a float."""
        result = analysis.circular_mean(np.array([14.0, 15.0]))
        assert isinstance(result, float)
        assert result == 14.5

    def test_circular_std_returns_float(self) -> None:
        """Placeholder returns a float (currently numpy std)."""
        result = analysis.circular_std(np.array([14.0, 15.0]))
        assert isinstance(result, float)

    def test_von_mises_regression_returns_dict(self) -> None:
        """Placeholder returns a dict with status key."""
        df = pd.DataFrame({"year": [2000], "peak_hour": [14]})
        result = analysis.von_mises_regression(df)
        assert isinstance(result, dict)
        assert "status" in result

    def test_bootstrap_trend_returns_tuple(self) -> None:
        """Placeholder returns a 3-tuple of floats."""
        df = pd.DataFrame({"year": [2000, 2001], "peak_hour": [14.0, 14.1]})
        result = analysis.bootstrap_trend(df)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_sensitivity_analysis_returns_dict(self) -> None:
        """Placeholder returns a dict with status key."""
        config = EpinalPeakConfig()
        df = pd.DataFrame({"year": [2000], "peak_hour": [14]})
        result = analysis.sensitivity_analysis(df, config)
        assert isinstance(result, dict)
        assert "status" in result
