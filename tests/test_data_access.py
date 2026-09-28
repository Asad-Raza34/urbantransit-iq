"""Tests for data access layer."""

import pytest
import pandas as pd
import numpy as np

from src.services.data_access import (
    calculate_kpis,
    filter_by_route,
    filter_by_date_range,
    filter_by_severity,
    filter_by_priority,
    DataAccess,
)


class TestFilters:
    """Test filtering utilities."""

    def test_filter_by_route(self):
        df = pd.DataFrame({
            "route_id": ["R-001", "R-002", "R-003"],
            "value": [1, 2, 3],
        })

        result = filter_by_route(df, ["R-001", "R-003"])
        assert len(result) == 2
        assert set(result["route_id"]) == {"R-001", "R-003"}

    def test_filter_by_route_empty(self):
        df = pd.DataFrame({"route_id": ["R-001"], "value": [1]})
        result = filter_by_route(df, [])
        assert len(result) == 1  # No filter applied

    def test_filter_by_route_missing_column(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = filter_by_route(df, ["R-001"])
        assert len(result) == 3  # No route_id column, no filter

    def test_filter_by_date_range(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
            "value": [1, 2, 3],
        })

        result = filter_by_date_range(df, "date", "2024-01-10", "2024-01-20")
        assert len(result) == 1
        assert result["value"].iloc[0] == 2

    def test_filter_by_date_range_partial(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
            "value": [1, 2, 3],
        })

        result = filter_by_date_range(df, "date", "2024-01-10", None)
        assert len(result) == 2

        result = filter_by_date_range(df, "date", None, "2024-01-20")
        assert len(result) == 2

    def test_filter_by_severity(self):
        df = pd.DataFrame({
            "severity": ["critical", "high", "medium", "low"],
            "value": [1, 2, 3, 4],
        })

        result = filter_by_severity(df, "severity", ["critical", "high"])
        assert len(result) == 2
        assert set(result["severity"]) == {"critical", "high"}

    def test_filter_by_priority(self):
        df = pd.DataFrame({
            "priority": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            "value": [1, 2, 3, 4],
        })

        result = filter_by_priority(df, "priority", ["CRITICAL", "HIGH"])
        assert len(result) == 2


class TestDataAccess:
    """Test DataAccess class methods."""

    def test_calculate_kpis_structure(self):
        # This requires actual data, so just verify it returns a dict with expected keys
        kpis = calculate_kpis()
        assert isinstance(kpis, dict)
        expected_keys = [
            "total_routes", "active_stops", "total_passengers_daily",
            "avg_delay_min", "avg_occupancy_pct", "on_time_pct",
            "overcrowding_events", "persistent_overcrowding_routes",
            "underutilized_routes", "active_alerts", "critical_alerts",
            "bottleneck_stops", "route_score_avg",
        ]
        for key in expected_keys:
            assert key in kpis

    def test_filter_methods(self):
        # Test DataAccess static methods work
        da = DataAccess()

        # These will work if data is loaded, otherwise return empty DataFrames
        # Just verify they don't crash
        routes = da.route_metrics(["R-001"])
        assert isinstance(routes, pd.DataFrame)

        demand = da.daily_demand(["R-001"], "2024-01-01", "2024-01-31")
        assert isinstance(demand, pd.DataFrame)

        alerts = da.alerts(["new"], ["critical"])
        assert isinstance(alerts, pd.DataFrame)

        recs = da.recommendations(["CRITICAL"], ["capacity_increase"])
        assert isinstance(recs, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])