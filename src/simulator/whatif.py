"""What-If Scenario Simulator for UrbanTransit IQ.

Allows simulation of operational changes and estimates their impact on
key metrics. Clearly separates observed historical values from scenario
assumptions and calculated outputs.
"""

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.analytics.core import read_processed
from src.audit import record
from src.log import get_logger
from src.paths import DATA_ANALYTICS
from src.storage import read_dataset, write_dataset

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scenario definition and results
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """Definition of a what-if scenario."""
    scenario_id: str
    name: str
    description: str
    # Scenario knobs (multipliers/factors)
    demand_multiplier: float = 1.0           # e.g., 1.2 = 20% demand increase
    capacity_multiplier: float = 1.0         # e.g., 1.1 = 10% more capacity
    vehicle_multiplier: float = 1.0          # e.g., 1.2 = 20% more vehicles
    frequency_multiplier: float = 1.0        # e.g., 0.9 = 10% less frequency
    headway_multiplier: float = 1.0          # e.g., 1.0 = no change
    delay_multiplier: float = 1.0            # e.g., 1.2 = 20% more delay
    peak_demand_multiplier: float = 1.0      # e.g., 1.3 = 30% more peak demand
    event_day_demand_multiplier: float = 1.0 # e.g., 1.5 = event day
    # Scope
    affected_routes: list[str] | None = None # None = all routes
    affected_time_bands: list[str] | None = None # None = all
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScenarioResult:
    """Results of a scenario simulation."""
    scenario_id: str
    scenario_name: str
    # Baseline (historical) values
    baseline: dict[str, float]
    # Scenario outputs (calculated)
    estimated: dict[str, float]
    # Delta
    delta: dict[str, float]
    delta_pct: dict[str, float]
    # Metadata
    assumptions: dict[str, float]
    calculated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


class WhatIfSimulator:
    """Simulates the impact of operational changes."""

    def __init__(self):
        self.baseline_metrics: dict[str, float] = {}
        self.route_baselines: pd.DataFrame | None = None

    def load_baseline(self) -> None:
        """Load current metrics as baseline."""
        logger.info("Loading baseline metrics for what-if simulator...")

        # Route-level metrics
        route_m = read_dataset(DATA_ANALYTICS / "route_metrics.parquet")
        crowding = read_dataset(DATA_ANALYTICS / "overcrowding_events.parquet")
        stop_m = read_dataset(DATA_ANALYTICS / "stop_metrics.parquet")
        daily_demand = read_dataset(DATA_ANALYTICS / "daily_demand.parquet")

        # Store route baselines for per-route calculations
        self.route_baselines = route_m.copy()

        # Aggregate baseline metrics
        self.baseline_metrics = {
            "total_routes": float(len(route_m)),
            "total_passengers_daily": float(daily_demand.groupby("date")["passengers"].sum().mean()),
            "avg_delay_min": float(route_m["avg_delay_min"].mean()),
            "avg_occupancy_avg_pct": float(route_m["occupancy_avg_pct"].mean()),
            "avg_occupancy_max_pct": float(route_m["occupancy_max_pct"].mean()),
            "overcrowding_events": float(len(crowding)),
            "on_time_share": float(route_m["on_time_share"].mean()),
            "avg_bunching_share": float(route_m["bunching_share"].mean()),
            "avg_headway_min": float(route_m["avg_headway_min"].mean()),
            "underutilized_routes": float((route_m["underutilized_share"] > 0.5).sum()),
            "bottleneck_stops": float((stop_m["is_bottleneck"] == True).sum()),
            "avg_revenue_daily": float(daily_demand.groupby("date")["revenue"].sum().mean()),
        }

    def _apply_scenario_to_route(self, row: pd.Series, scenario: Scenario) -> dict[str, float]:
        """Apply scenario assumptions to a single route's metrics."""
        # Extract scenario parameters
        dm = scenario.demand_multiplier
        cm = scenario.capacity_multiplier
        vm = scenario.vehicle_multiplier
        fm = scenario.frequency_multiplier
        hm = scenario.headway_multiplier
        dlm = scenario.delay_multiplier
        pdm = scenario.peak_demand_multiplier

        # Current values
        passengers = float(row.get("passengers", 0))
        avg_boardings = float(row.get("avg_boardings_per_trip", 0))
        avg_delay = float(row.get("avg_delay_min", 0))
        on_time = float(row.get("on_time_share", 0))
        occupancy_avg = float(row.get("occupancy_avg_pct", 0))
        occupancy_max = float(row.get("occupancy_max_pct", 0))
        bunching_share = float(row.get("bunching_share", 0))
        headway = float(row.get("avg_headway_min", 0))
        headway_cv = float(row.get("headway_cv", 0)) if pd.notna(row.get("headway_cv")) else 0
        scheduled_travel = float(row.get("avg_scheduled_travel_min", 0))
        actual_travel = float(row.get("avg_actual_travel_min", 0))
        capacity = float(row.get("capacity", 80))  # default if missing

        # Demand effect
        new_passengers = passengers * dm
        new_avg_boardings = avg_boardings * dm

        # Peak demand effect (additional multiplier during peak)
        peak_factor = 1.0
        if pdm != 1.0:
            # Rough approximation: peak periods get extra demand multiplier
            peak_factor = pdm

        # Capacity effect
        new_capacity = capacity * cm

        # Frequency/vehicle effect on headway and bunching
        # More frequency = lower headway, less bunching
        new_headway = headway / max(fm * vm, 0.1) if fm * vm > 0 else headway
        new_bunching_share = bunching_share * (1.0 / max(fm * vm, 0.1)) ** 0.5

        # Delay effect
        new_delay = avg_delay * dlm
        # On-time share: assume linear-ish relationship
        new_on_time = max(0.0, min(1.0, on_time - (new_delay - avg_delay) * 0.02))

        # Occupancy: demand / capacity
        # Rough: occupancy proportional to boardings/capacity
        if capacity > 0:
            occ_factor = (new_avg_boardings / new_capacity) / (avg_boardings / capacity)
        else:
            occ_factor = 1.0
        new_occupancy_avg = occupancy_avg * occ_factor
        new_occupancy_max = occupancy_max * occ_factor

        # Travel time: delay affects actual
        new_actual_travel = scheduled_travel + new_delay

        # Crowding share estimation
        crowding_share = float(row.get("crowding_share", 0))
        new_crowding_share = crowding_share * occ_factor

        # Underutilization
        underutil_share = float(row.get("underutilized_share", 0))
        new_underutil_share = underutil_share / max(occ_factor, 0.1)

        return {
            "passengers": new_passengers,
            "avg_boardings_per_trip": new_avg_boardings,
            "avg_delay_min": new_delay,
            "on_time_share": new_on_time,
            "occupancy_avg_pct": new_occupancy_avg,
            "occupancy_max_pct": new_occupancy_max,
            "bunching_share": new_bunching_share,
            "avg_headway_min": new_headway,
            "capacity": new_capacity,
            "crowding_share": new_crowding_share,
            "underutilized_share": new_underutil_share,
            "avg_actual_travel_min": new_actual_travel,
        }

    def run(self, scenario: Scenario) -> ScenarioResult:
        """Run a scenario simulation."""
        if not self.baseline_metrics:
            self.load_baseline()

        if self.route_baselines is None or self.route_baselines.empty:
            raise ValueError("No baseline data loaded")

        # Filter affected routes
        routes = self.route_baselines
        if scenario.affected_routes:
            routes = routes[routes["route_id"].isin(scenario.affected_routes)].copy()

        # Apply scenario to each route
        results = []
        for _, row in routes.iterrows():
            est = self._apply_scenario_to_route(row, scenario)
            est["route_id"] = row["route_id"]
            results.append(est)

        est_df = pd.DataFrame(results)

        # Aggregate estimated metrics
        estimated = {
            "total_passengers_daily": float(est_df["passengers"].sum() / max(len(est_df), 1) * len(self.route_baselines)),
            "avg_delay_min": float(est_df["avg_delay_min"].mean()),
            "avg_occupancy_avg_pct": float(est_df["occupancy_avg_pct"].mean()),
            "avg_occupancy_max_pct": float(est_df["occupancy_max_pct"].mean()),
            "on_time_share": float(est_df["on_time_share"].mean()),
            "avg_bunching_share": float(est_df["bunching_share"].mean()),
            "avg_headway_min": float(est_df["avg_headway_min"].mean()),
            "underutilized_routes": float((est_df["underutilized_share"] > 0.5).sum()),
            "overcrowding_events_est": float((est_df["occupancy_max_pct"] >= settings.CROWDING_CRITICAL * 100).sum()),
            "avg_actual_travel_min": float(est_df["avg_actual_travel_min"].mean()),
        }

        # Calculate deltas
        delta = {}
        delta_pct = {}
        for key in self.baseline_metrics:
            if key in estimated:
                base = self.baseline_metrics[key]
                est = estimated[key]
                delta[key] = round(est - base, 3)
                delta_pct[key] = round((est - base) / base * 100, 1) if base != 0 else 0.0

        # Assumptions used
        assumptions = {
            "demand_multiplier": scenario.demand_multiplier,
            "capacity_multiplier": scenario.capacity_multiplier,
            "vehicle_multiplier": scenario.vehicle_multiplier,
            "frequency_multiplier": scenario.frequency_multiplier,
            "headway_multiplier": scenario.headway_multiplier,
            "delay_multiplier": scenario.delay_multiplier,
            "peak_demand_multiplier": scenario.peak_demand_multiplier,
            "event_day_demand_multiplier": scenario.event_day_demand_multiplier,
        }

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            baseline=self.baseline_metrics,
            estimated=estimated,
            delta=delta,
            delta_pct=delta_pct,
            assumptions=assumptions,
        )

    def run_comparison(self, scenarios: list[Scenario]) -> list[ScenarioResult]:
        """Run multiple scenarios and return all results."""
        return [self.run(s) for s in scenarios]


# Predefined scenario templates
SCENARIO_TEMPLATES = {
    "demand_increase_10": Scenario(
        scenario_id="",
        name="Demand +10%",
        description="10% increase in passenger demand across all routes",
        demand_multiplier=1.10,
    ),
    "demand_increase_20": Scenario(
        scenario_id="",
        name="Demand +20%",
        description="20% increase in passenger demand across all routes",
        demand_multiplier=1.20,
    ),
    "capacity_increase_10": Scenario(
        scenario_id="",
        name="Capacity +10%",
        description="10% increase in vehicle capacity (e.g., more articulated buses)",
        capacity_multiplier=1.10,
    ),
    "vehicle_increase_15": Scenario(
        scenario_id="",
        name="Vehicles +15%",
        description="15% more vehicles deployed, reducing headways",
        vehicle_multiplier=1.15,
    ),
    "frequency_increase_10": Scenario(
        scenario_id="",
        name="Frequency +10%",
        description="10% increase in service frequency",
        frequency_multiplier=1.10,
    ),
    "peak_demand_surge": Scenario(
        scenario_id="",
        name="Peak Demand Surge +30%",
        description="30% increase in peak-period demand (event day simulation)",
        peak_demand_multiplier=1.30,
        demand_multiplier=1.10,
    ),
    "delay_increase_20": Scenario(
        scenario_id="",
        name="Delay +20%",
        description="20% increase in delays (e.g., construction, weather)",
        delay_multiplier=1.20,
    ),
    "combined_improvement": Scenario(
        scenario_id="",
        name="Combined Improvement",
        description="Capacity +10%, Vehicles +10%, Frequency +10%",
        capacity_multiplier=1.10,
        vehicle_multiplier=1.10,
        frequency_multiplier=1.10,
    ),
    "route_specific_overcrowded": Scenario(
        scenario_id="",
        name="Target Overcrowded Routes",
        description="Increase capacity and frequency on persistently overcrowded routes",
        capacity_multiplier=1.15,
        frequency_multiplier=1.15,
        affected_routes=[],  # Will be populated at runtime
    ),
}


def get_overcrowded_routes() -> list[str]:
    """Get list of persistently overcrowded route IDs."""
    try:
        persistent = read_dataset(DATA_ANALYTICS / "persistent_overcrowding.parquet")
        if not persistent.empty:
            return persistent["route_id"].tolist()
    except Exception:
        pass
    return []


def run_scenario(scenario_name: str, **overrides) -> ScenarioResult:
    """Convenience function to run a predefined scenario."""
    if scenario_name not in SCENARIO_TEMPLATES:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIO_TEMPLATES.keys())}")

    template = SCENARIO_TEMPLATES[scenario_name]
    scenario = Scenario(
        scenario_id=f"scn-{uuid.uuid4().hex[:8]}",
        name=template.name,
        description=template.description,
        demand_multiplier=template.demand_multiplier,
        capacity_multiplier=template.capacity_multiplier,
        vehicle_multiplier=template.vehicle_multiplier,
        frequency_multiplier=template.frequency_multiplier,
        headway_multiplier=template.headway_multiplier,
        delay_multiplier=template.delay_multiplier,
        peak_demand_multiplier=template.peak_demand_multiplier,
        event_day_demand_multiplier=template.event_day_demand_multiplier,
        affected_routes=template.affected_routes or overrides.get("affected_routes"),
        affected_time_bands=template.affected_time_bands or overrides.get("affected_time_bands"),
    )

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(scenario, key):
            setattr(scenario, key, value)

    # Special handling for route-specific scenario
    if scenario_name == "route_specific_overcrowded":
        if not scenario.affected_routes:
            scenario.affected_routes = get_overcrowded_routes()

    sim = WhatIfSimulator()
    return sim.run(scenario)


if __name__ == "__main__":
    import time
    sim = WhatIfSimulator()
    sim.load_baseline()
    print("Baseline:", json.dumps(sim.baseline_metrics, indent=2))

    # Run a few scenarios
    for name in ["demand_increase_10", "capacity_increase_10", "combined_improvement"]:
        result = run_scenario(name)
        print(f"\n--- {name} ---")
        print("Delta:", json.dumps(result.delta, indent=2))
        print("Delta %:", json.dumps(result.delta_pct, indent=2))