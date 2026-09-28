"""Tests for frequency analysis."""

import pytest
import pandas as pd
import numpy as np

from src.analytics.frequency import (
    frequency_analysis,
    headway_analysis,
    service_level,
    frequency_peak_analysis,
    _compute_headway_reliability_vectorized,
    _time_period_label,
)


class TestTimePeriodLabel:
    """Test time period mapping."""

    def test_morning_peak(self):
        assert _time_period_label(7) == "morning peak"
        assert _time_period_label(9) == "morning peak"

    def test_midday(self):
        assert _time_period_label(12) == "midday"
        assert _time_period_label(15) == "midday"

    def test_evening_peak(self):
        assert _time_period_label(17) == "evening peak"
        assert _time_period_label(19) == "evening peak"

    def test_night(self):
        assert _time_period_label(2) == "night"
        assert _time_period_label(22) == "night"


class TestHeadwayReliability:
    """Test headway reliability computation."""

    def test_reliability_computation(self):
        # Create test data
        df = pd.DataFrame({
            "route_id": ["R-001", "R-001", "R-001", "R-001"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"]),
            "hour": [8, 8, 8, 8],
            "scheduled_headway_min": [10, 10, 10, 10],
            "actual_headway_min": [9, 10, 11, 13],  # 9, 10, 11 within 20% (8-12), 13 outside
        })
        result = _compute_headway_reliability_vectorized(df, include_date=True)
        assert len(result) == 1
        # 3 out of 4 within 20% = 0.75
        assert result["headway_reliability"].iloc[0] == 0.75

    def test_reliability_without_date(self):
        df = pd.DataFrame({
            "route_id": ["R-001", "R-001", "R-001", "R-001"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "hour": [8, 8, 8, 8],
            "scheduled_headway_min": [10, 10, 10, 10],
            "actual_headway_min": [9, 10, 11, 13],
        })
        result = _compute_headway_reliability_vectorized(df, include_date=False)
        assert len(result) == 1
        assert result["headway_reliability"].iloc[0] == 0.75


class TestFrequencyAnalysis:
    """Test frequency analysis functions."""

    def test_frequency_analysis_structure(self):
        # This requires actual data - just verify it runs and returns expected columns
        result = frequency_analysis()
        assert isinstance(result, pd.DataFrame)
        expected_cols = ["route_id", "date", "time_period", "scheduled_trips", "trips_per_hour",
                         "avg_scheduled_headway_min", "headway_reliability", "service_gap_min"]
        for col in expected_cols:
            assert col in result.columns

    def test_frequency_analysis_has_data(self):
        result = frequency_analysis()
        assert len(result) > 0
        assert result["route_id"].nunique() > 0

    def test_headway_analysis_structure(self):
        result = headway_analysis()
        assert isinstance(result, pd.DataFrame)
        expected_cols = ["route_id", "time_period", "scheduled_mean", "actual_mean",
                         "headway_cv", "headway_reliability"]
        for col in expected_cols:
            assert col in result.columns

    def test_headway_analysis_correct_shape(self):
        result = headway_analysis()
        # Should be 100 routes * 4 time periods = 400 rows
        assert len(result) == 400
        assert result["route_id"].nunique() == 100
        assert result["time_period"].nunique() == 4

    def test_service_level_structure(self):
        result = service_level()
        assert isinstance(result, pd.DataFrame)
        expected_cols = ["route_id", "date", "time_period", "service_regularity",
                         "trips_per_hour", "on_time_rate", "service_reliability"]
        for col in expected_cols:
            assert col in result.columns

    def test_service_level_matches_frequency_rows(self):
        freq = frequency_analysis()
        service = service_level()
        # Service level should have same row count as frequency analysis
        assert len(service) == len(freq)

    def test_frequency_peak_analysis_structure(self):
        result = frequency_peak_analysis()
        assert isinstance(result, pd.DataFrame)
        expected_cols = ["route_id", "peak_freq", "offpeak_freq", "peak_ratio"]
        for col in expected_cols:
            assert col in result.columns

    def test_frequency_peak_analysis_correct_shape(self):
        result = frequency_peak_analysis()
        assert len(result) == 100  # One per route
        assert result["route_id"].nunique() == 100

    def test_peak_ratio_positive(self):
        result = frequency_peak_analysis()
        assert (result["peak_ratio"] > 0).all()

    def test_build_returns_counts(self):
        from src.analytics.frequency import build
        result = build()
        assert isinstance(result, dict)
        expected_keys = ["frequency_analysis", "headway_analysis", "service_level", "frequency_peak_analysis"]
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], int)
            assert result[key] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])