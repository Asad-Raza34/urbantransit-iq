"""Tests for Smart Alert Center."""

import pytest
import pandas as pd
import numpy as np

from src.alerts.center import (
    AlertCenter,
    _determine_severity,
    ALERT_TYPES,
    SEVERITY_ORDER,
    VALID_STATUSES,
)


class TestSeverityDetermination:
    """Test alert severity determination."""

    def test_critical_overcrowding(self):
        evidence = {"critical_days": 35, "crowded_days": 40}
        assert _determine_severity("persistent_overcrowding", evidence) == "critical"

    def test_high_overcrowding(self):
        evidence = {"crowded_days": 25}
        assert _determine_severity("persistent_overcrowding", evidence) == "high"

    def test_critical_crowding_event(self):
        evidence = {"critical_share_pct": 60}
        assert _determine_severity("critical_crowding_event", evidence) == "critical"

    def test_critical_delay(self):
        evidence = {"avg_delay": 25}
        assert _determine_severity("severe_delay", evidence) == "critical"

    def test_high_delay(self):
        evidence = {"severe_pct": 40}
        assert _determine_severity("severe_delay", evidence) == "high"

    def test_critical_bottleneck(self):
        evidence = {"score": 95}
        assert _determine_severity("stop_bottleneck", evidence) == "critical"

    def test_high_anomaly(self):
        evidence = {"z_score": 4.5}
        assert _determine_severity("demand_anomaly", evidence) == "high"

    def test_default_severity(self):
        evidence = {}
        assert _determine_severity("severe_delay", evidence) == "high"
        assert _determine_severity("stop_bottleneck", evidence) == "high"
        assert _determine_severity("demand_anomaly", evidence) == "medium"


class TestAlertCenter:
    """Test AlertCenter functionality."""

    def test_initialization(self):
        center = AlertCenter()
        assert center.alerts == []

    def test_severity_order(self):
        assert SEVERITY_ORDER == {"critical": 0, "high": 1, "medium": 2, "low": 3}

    def test_valid_statuses(self):
        assert VALID_STATUSES == {"new", "acknowledged", "resolved"}

    def test_alert_types_defined(self):
        expected_types = [
            "severe_delay", "persistent_overcrowding", "critical_crowding_event",
            "reliability_degraded", "stop_bottleneck", "demand_anomaly",
            "underutilized_route", "bunching_persistent", "schedule_adherence_poor",
        ]
        for t in expected_types:
            assert t in ALERT_TYPES
            assert "title_template" in ALERT_TYPES[t]
            assert "description_template" in ALERT_TYPES[t]
            assert "default_severity" in ALERT_TYPES[t]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])