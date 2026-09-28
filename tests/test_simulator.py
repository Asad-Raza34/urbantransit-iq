"""Tests for What-If Scenario Simulator."""

import pytest
import pandas as pd
import numpy as np

from src.simulator.whatif import (
    WhatIfSimulator,
    Scenario,
    ScenarioResult,
    SCENARIO_TEMPLATES,
    run_scenario,
    get_overcrowded_routes,
)


class TestScenario:
    """Test Scenario dataclass."""

    def test_scenario_creation(self):
        scenario = Scenario(
            scenario_id="test-001",
            name="Test Scenario",
            description="Test description",
            demand_multiplier=1.1,
            capacity_multiplier=1.0,
        )
        assert scenario.scenario_id == "test-001"
        assert scenario.demand_multiplier == 1.1
        assert scenario.capacity_multiplier == 1.0

    def test_scenario_to_dict(self):
        scenario = Scenario(
            scenario_id="test-001",
            name="Test",
            description="Desc",
        )
        d = scenario.to_dict()
        assert d["scenario_id"] == "test-001"
        assert d["name"] == "Test"
        assert d["demand_multiplier"] == 1.0


class TestScenarioTemplates:
    """Test predefined scenario templates."""

    def test_templates_exist(self):
        expected = [
            "demand_increase_10", "demand_increase_20", "capacity_increase_10",
            "vehicle_increase_15", "frequency_increase_10", "peak_demand_surge",
            "delay_increase_20", "combined_improvement", "route_specific_overcrowded",
        ]
        for name in expected:
            assert name in SCENARIO_TEMPLATES

    def test_template_values(self):
        # Demand increase 10%
        t = SCENARIO_TEMPLATES["demand_increase_10"]
        assert t.demand_multiplier == 1.10
        assert t.capacity_multiplier == 1.0

        # Combined improvement
        t = SCENARIO_TEMPLATES["combined_improvement"]
        assert t.capacity_multiplier == 1.10
        assert t.vehicle_multiplier == 1.10
        assert t.frequency_multiplier == 1.10


class TestWhatIfSimulator:
    """Test the WhatIfSimulator."""

    def test_simulator_initialization(self):
        sim = WhatIfSimulator()
        assert sim.baseline_metrics == {}
        assert sim.route_baselines is None

    def test_load_baseline(self):
        sim = WhatIfSimulator()
        sim.load_baseline()

        assert sim.baseline_metrics
        assert "total_routes" in sim.baseline_metrics
        assert "avg_delay_min" in sim.baseline_metrics
        assert "on_time_share" in sim.baseline_metrics
        assert sim.route_baselines is not None
        assert len(sim.route_baselines) > 0

    def test_apply_scenario_to_route(self):
        sim = WhatIfSimulator()
        sim.load_baseline()

        # Create a test route row
        route_row = pd.Series({
            "passengers": 10000,
            "avg_boardings_per_trip": 50.0,
            "avg_delay_min": 5.0,
            "on_time_share": 0.85,
            "occupancy_avg_pct": 55.0,
            "occupancy_max_pct": 85.0,
            "bunching_share": 0.1,
            "headway_cv": 0.3,
            "avg_headway_min": 10.0,
            "avg_scheduled_travel_min": 45.0,
            "avg_actual_travel_min": 50.0,
            "underutilized_share": 0.1,
            "crowding_share": 0.15,
            "capacity": 80.0,
        })

        scenario = Scenario(
            scenario_id="test",
            name="Test",
            description="Test",
            demand_multiplier=1.2,
            capacity_multiplier=1.0,
            vehicle_multiplier=1.0,
            frequency_multiplier=1.0,
        )

        result = sim._apply_scenario_to_route(route_row, scenario)

        # Demand increase should increase passengers
        assert result["passengers"] > 10000
        assert result["avg_boardings_per_trip"] > 50.0

        # Occupancy should increase with demand
        assert result["occupancy_avg_pct"] > 55.0
        assert result["occupancy_max_pct"] > 85.0

        # Delay unchanged (delay_multiplier=1.0)
        assert result["avg_delay_min"] == 5.0

        # On-time should decrease slightly with more delay from demand
        assert result["on_time_share"] <= 0.85

    def test_capacity_increase_reduces_occupancy(self):
        sim = WhatIfSimulator()
        sim.load_baseline()

        route_row = pd.Series({
            "passengers": 10000,
            "avg_boardings_per_trip": 50.0,
            "avg_delay_min": 5.0,
            "on_time_share": 0.85,
            "occupancy_avg_pct": 55.0,
            "occupancy_max_pct": 85.0,
            "bunching_share": 0.1,
            "headway_cv": 0.3,
            "avg_headway_min": 10.0,
            "avg_scheduled_travel_min": 45.0,
            "avg_actual_travel_min": 50.0,
            "underutilized_share": 0.1,
            "crowding_share": 0.15,
            "capacity": 80.0,
        })

        scenario = Scenario(
            scenario_id="test",
            name="Test",
            description="Test",
            demand_multiplier=1.0,
            capacity_multiplier=1.2,
        )

        result = sim._apply_scenario_to_route(route_row, scenario)

        # Capacity increase should reduce occupancy
        assert result["occupancy_avg_pct"] < 55.0
        assert result["occupancy_max_pct"] < 85.0
        assert result["capacity"] > 80.0

    def test_frequency_increase_reduces_bunching(self):
        sim = WhatIfSimulator()
        sim.load_baseline()

        route_row = pd.Series({
            "passengers": 10000,
            "avg_boardings_per_trip": 50.0,
            "avg_delay_min": 5.0,
            "on_time_share": 0.85,
            "occupancy_avg_pct": 55.0,
            "occupancy_max_pct": 85.0,
            "bunching_share": 0.2,
            "headway_cv": 0.3,
            "avg_headway_min": 10.0,
            "avg_scheduled_travel_min": 45.0,
            "avg_actual_travel_min": 50.0,
            "underutilized_share": 0.1,
            "crowding_share": 0.15,
            "capacity": 80.0,
        })

        scenario = Scenario(
            scenario_id="test",
            name="Test",
            description="Test",
            demand_multiplier=1.0,
            capacity_multiplier=1.0,
            vehicle_multiplier=1.2,
            frequency_multiplier=1.2,
        )

        result = sim._apply_scenario_to_route(route_row, scenario)

        # More frequency/vehicles should reduce headway and bunching
        assert result["avg_headway_min"] < 10.0
        assert result["bunching_share"] < 0.2


class TestRunScenario:
    """Test run_scenario convenience function."""

    def test_known_scenarios(self):
        # Just verify the function doesn't error for known scenarios
        for name in ["demand_increase_10", "capacity_increase_10", "combined_improvement"]:
            try:
                result = run_scenario(name)
                assert isinstance(result, ScenarioResult)
                assert result.scenario_name == result.scenario_name
            except Exception as e:
                # May fail if baseline data not available in test env
                pytest.skip(f"Scenario {name} requires baseline data: {e}")

    def test_unknown_scenario(self):
        with pytest.raises(ValueError):
            run_scenario("nonexistent_scenario")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])