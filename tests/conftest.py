"""Test configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def sample_routes():
    """Small sample of route metrics for testing."""
    return pd.DataFrame({
        "route_id": ["R-001", "R-002", "R-003"],
        "route_name": ["Route 1", "Route 2", "Route 3"],
        "category": ["core", "feeder", "express"],
        "route_score": [85.0, 65.0, 45.0],
        "route_class": ["Top Performer", "Solid", "Action Required"],
        "passengers": [10000, 5000, 2000],
        "avg_delay_min": [2.5, 5.0, 12.0],
        "on_time_share": [0.92, 0.75, 0.55],
        "occupancy_avg_pct": [55.0, 40.0, 70.0],
        "occupancy_max_pct": [80.0, 65.0, 95.0],
        "crowding_share": [0.1, 0.05, 0.4],
        "bunching_share": [0.05, 0.02, 0.2],
        "route_rank": [1, 2, 3],
    })


@pytest.fixture(scope="session")
def sample_alerts():
    """Sample alerts for testing."""
    return pd.DataFrame({
        "alert_id": ["alt-001", "alt-002", "alt-003"],
        "alert_type": ["severe_delay", "persistent_overcrowding", "stop_bottleneck"],
        "severity": ["high", "critical", "medium"],
        "status": ["new", "new", "acknowledged"],
        "affected_route": ["R-001", "R-002", "R-003"],
        "affected_stop": [None, None, "S-001"],
        "title": ["Test Alert 1", "Test Alert 2", "Test Alert 3"],
        "description": ["Desc 1", "Desc 2", "Desc 3"],
        "metric_value": [15.5, 0.45, 85.0],
        "threshold_used": [9.0, 0.85, "60th percentile"],
        "timestamp": ["2024-01-15T10:00:00+00:00"] * 3,
    })


@pytest.fixture(scope="session")
def sample_recommendations():
    """Sample recommendations for testing."""
    return pd.DataFrame({
        "recommendation_id": ["rec-001", "rec-002", "rec-003"],
        "category": ["capacity_increase", "frequency_increase", "delay_reduction"],
        "priority": ["CRITICAL", "HIGH", "MEDIUM"],
        "affected_route": ["R-001", "R-002", "R-003"],
        "affected_stop": [None, None, None],
        "title": ["Capacity increase for R-001", "Frequency increase for R-002", "Delay reduction for R-003"],
        "description": ["Desc 1", "Desc 2", "Desc 3"],
        "evidence": [
            {"crowding_share": 0.45, "occupancy_max_pct": 95.0},
            {"bunching_share": 0.25, "avg_headway_min": 8.0},
            {"avg_delay_min": 15.0, "on_time_share": 0.55},
        ],
        "metric_value": [95.0, 0.25, 15.0],
        "threshold_used": [0.85, 0.15, 5.0],
        "suggested_action": ["Add articulated buses", "Increase frequency", "Signal priority"],
        "expected_effect": ["Reduce crowding", "Reduce bunching", "Reduce delay"],
        "confidence": [0.9, 0.85, 0.8],
        "generated_at": ["2024-01-15T10:00:00+00:00"] * 3,
    })


# Test utilities
def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype=False):
    """Assert two DataFrames are equal."""
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True), check_dtype=check_dtype)


def create_test_route_metrics(n=10):
    """Create test route metrics DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "route_id": [f"R-{i:03d}" for i in range(1, n+1)],
        "route_name": [f"Route {i}" for i in range(1, n+1)],
        "category": np.random.choice(["core", "feeder", "express"], n),
        "route_score": np.random.uniform(40, 95, n).round(1),
        "route_class": np.random.choice(["Top Performer", "Solid", "Needs Attention", "Action Required"], n),
        "passengers": np.random.randint(1000, 20000, n),
        "avg_delay_min": np.random.uniform(1, 20, n).round(1),
        "on_time_share": np.random.uniform(0.4, 0.95, n).round(3),
        "occupancy_avg_pct": np.random.uniform(20, 90, n).round(1),
        "occupancy_max_pct": np.random.uniform(40, 120, n).round(1),
        "crowding_share": np.random.uniform(0, 0.5, n).round(3),
        "bunching_share": np.random.uniform(0, 0.3, n).round(3),
        "route_rank": range(1, n+1),
        "adherence_share": np.random.uniform(0.5, 0.9, n).round(3),
        "severe_critical_share": np.random.uniform(0, 0.4, n).round(3),
        "avg_headway_min": np.random.uniform(5, 30, n).round(1),
        "headway_cv": np.random.uniform(0.1, 0.8, n).round(3),
    })