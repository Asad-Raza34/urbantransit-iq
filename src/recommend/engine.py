"""Evidence-based recommendation engine for UrbanTransit IQ.

Generates actionable transport recommendations from actual analytics outputs.
Every recommendation carries metrics, thresholds, and a clear rationale.
"""

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.analytics.core import read_processed
from src.analytics import route_metrics, crowding, demand, stop_metrics
from src.audit import record
from src.log import get_logger
from src.paths import DATA_ANALYTICS, MODELS_PYTHON
from src.storage import read_dataset, write_dataset

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Recommendation categories and their detection logic
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """Structured recommendation with full evidence trail."""
    recommendation_id: str
    category: str
    priority: str
    affected_route: str | None
    affected_stop: str | None
    title: str
    description: str
    evidence: dict[str, Any]
    metric_value: float | None
    threshold_used: float | str | None
    suggested_action: str
    expected_effect: str
    confidence: float
    generated_at: str
    model_version: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# Priority thresholds (documented and transparent)
PRIORITY_RULES = {
    "CRITICAL": {
        "crowding_share": 0.4,          # >40% trips crowded
        "critical_crowding_share": 0.1, # >10% trips critically crowded
        "on_time_share": 0.55,          # <55% on-time
        "severe_critical_share": 0.3,   # >30% severe/critical delays
        "bottleneck_score": 90,         # top 10% bottleneck
        "persistent_crowding_days": 30, # crowded >30 days
    },
    "HIGH": {
        "crowding_share": 0.25,         # >25% trips crowded
        "critical_crowding_share": 0.05,
        "on_time_share": 0.65,          # <65% on-time
        "severe_critical_share": 0.2,
        "bottleneck_score": 80,         # top 20% bottleneck
        "underutilized_share": 0.7,     # >70% trips underutilized
        "persistent_crowding_days": 20,
        "demand_anomaly_z": 3.0,
    },
    "MEDIUM": {
        "crowding_share": 0.15,
        "on_time_share": 0.75,          # <75% on-time (degraded)
        "bunching_share": 0.15,         # >15% trips bunched
        "bottleneck_score": 70,         # top 30% bottleneck
        "underutilized_share": 0.5,
        "persistent_crowding_days": 15,
    },
    "LOW": {
        "crowding_share": 0.1,
        "on_time_share": 0.85,
        "bunching_share": 0.1,
        "underutilized_share": 0.3,
    },
}


CATEGORY_ACTIONS = {
    "capacity_increase": {
        "action": "Increase vehicle capacity or add articulated buses",
        "effect": "Reduce crowding share and improve passenger comfort",
    },
    "frequency_increase": {
        "action": "Increase service frequency during peak periods",
        "effect": "Reduce headway, alleviate crowding and bunching",
    },
    "schedule_adjustment": {
        "action": "Adjust schedule to match actual travel times and demand patterns",
        "effect": "Improve on-time performance and schedule adherence",
    },
    "stop_intervention": {
        "action": "Investigate stop dwell times, boarding infrastructure, or signal priority",
        "effect": "Reduce stop-level delays and improve punctuality",
    },
    "delay_reduction": {
        "action": "Implement traffic signal priority, dedicated lanes, or operational controls",
        "effect": "Reduce average delay and improve reliability",
    },
    "headway_improvement": {
        "action": "Improve dispatch control and headway management",
        "effect": "Reduce bunching and gapping, improve wait times",
    },
    "crowding_mitigation": {
        "action": "Redistribute demand via express services or alternative routing",
        "effect": "Lower peak occupancy and critical crowding events",
    },
    "underutilized_review": {
        "action": "Review route alignment, frequency, or consider service reduction",
        "effect": "Improve resource allocation and cost efficiency",
    },
    "vehicle_reallocation": {
        "action": "Reallocate vehicles from underutilized to overcrowded routes",
        "effect": "Balance capacity across the network",
    },
    "operational_monitoring": {
        "action": "Increase monitoring frequency for this route/stop",
        "effect": "Early detection of degrading performance",
    },
    "demand_based_planning": {
        "action": "Incorporate demand forecasts into service planning",
        "effect": "Proactive capacity adjustments before crowding occurs",
    },
}


def _determine_priority(evidence: dict) -> str:
    """Determine priority based on transparent threshold rules."""
    for priority in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        thresholds = PRIORITY_RULES[priority]
        for key, threshold in thresholds.items():
            value = evidence.get(key)
            if value is None:
                continue
            # For "lower is worse" metrics (on_time_share), check if below threshold
            if key in ("on_time_share",):
                if value <= threshold:
                    return priority
            # For "higher is worse" metrics, check if above threshold
            elif key in ("bottleneck_score", "demand_anomaly_z"):
                if value >= threshold:
                    return priority
            else:
                if value >= threshold:
                    return priority
    return "LOW"


def _select_action(category: str, evidence: dict) -> tuple[str, str]:
    """Select appropriate action and expected effect for a recommendation category."""
    if category in CATEGORY_ACTIONS:
        return CATEGORY_ACTIONS[category]["action"], CATEGORY_ACTIONS[category]["effect"]
    return "Review and take appropriate operational action", "Improve operational performance"


def _build_explanation(rec: Recommendation) -> str:
    """Generate a human-readable explanation for the recommendation."""
    lines = []
    lines.append(f"### What happened?")
    lines.append(rec.description)
    lines.append("")
    lines.append(f"### Why is it important?")
    if rec.category == "capacity_increase":
        lines.append("Persistent high occupancy indicates demand exceeds vehicle capacity, leading to passenger discomfort and potential safety issues.")
    elif rec.category == "frequency_increase":
        lines.append("High demand combined with bunching suggests insufficient frequency to absorb passenger volume smoothly.")
    elif rec.category == "schedule_adjustment":
        lines.append("Schedule does not reflect actual operating conditions, causing systematic delays and reliability degradation.")
    elif rec.category == "stop_intervention":
        lines.append("Stop-level bottlenecks create cascading delays affecting entire route reliability.")
    elif rec.category == "delay_reduction":
        lines.append("Excessive delays reduce service attractiveness and increase operational costs.")
    elif rec.category == "headway_improvement":
        lines.append("Bunching creates uneven service intervals, increasing passenger wait times and crowding on following vehicles.")
    elif rec.category == "crowding_mitigation":
        lines.append("Critical crowding events indicate demand spikes that exceed safe capacity thresholds.")
    elif rec.category == "underutilized_review":
        lines.append("Consistently low utilization suggests resource misallocation; capacity could be better deployed elsewhere.")
    elif rec.category == "vehicle_reallocation":
        lines.append("Network-wide capacity imbalance: some routes overcrowded while others run nearly empty.")
    elif rec.category == "operational_monitoring":
        lines.append("Early warning indicators suggest potential performance degradation.")
    elif rec.category == "demand_based_planning":
        lines.append("Demand anomalies or forecast deviations indicate planning assumptions need updating.")
    else:
        lines.append("Operational metrics indicate attention is needed.")
    lines.append("")
    lines.append(f"### Evidence")
    for k, v in rec.evidence.items():
        if v is not None:
            lines.append(f"- **{k}**: {v}")
    if rec.threshold_used is not None:
        lines.append(f"- **Threshold used**: {rec.threshold_used}")
    lines.append("")
    lines.append(f"### What is suggested?")
    lines.append(rec.suggested_action)
    lines.append("")
    lines.append(f"### Why this action?")
    lines.append(f"Based on {rec.category.replace('_', ' ')} rules: metrics exceed {rec.priority.lower()} priority thresholds.")
    lines.append("")
    lines.append(f"### Expected effect")
    lines.append(rec.expected_effect)
    lines.append("")
    lines.append(f"### Confidence: {rec.confidence:.0%}")
    return "\n".join(lines)


class RecommendationEngine:
    """Generates recommendations from analytics data."""

    def __init__(self):
        self.recommendations: list[Recommendation] = []

    def load_analytics(self) -> dict[str, pd.DataFrame]:
        """Load all required analytics frames."""
        return {
            "route_metrics": read_dataset(DATA_ANALYTICS / "route_metrics.parquet"),
            "overcrowding_events": read_dataset(DATA_ANALYTICS / "overcrowding_events.parquet"),
            "persistent_overcrowding": read_dataset(DATA_ANALYTICS / "persistent_overcrowding.parquet"),
            "underutilization": read_dataset(DATA_ANALYTICS / "underutilization.parquet"),
            "bunching_events": read_dataset(DATA_ANALYTICS / "bunching_events.parquet"),
            "quality_days": read_dataset(DATA_ANALYTICS / "quality_days.parquet"),
            "stop_metrics": read_dataset(DATA_ANALYTICS / "stop_metrics.parquet"),
            "anomalies": read_dataset(DATA_ANALYTICS / "anomalies.parquet"),
            "daily_demand": read_dataset(DATA_ANALYTICS / "daily_demand.parquet"),
        }

    def generate_capacity_recommendations(self, data: dict) -> list[Recommendation]:
        """Generate capacity increase recommendations from crowding metrics."""
        recs = []
        persistent = data["persistent_overcrowding"]
        overcrowding = data["overcrowding_events"]

        if persistent.empty:
            return recs

        for _, row in persistent.iterrows():
            route_id = row["route_id"]
            crowded_days = row["crowded_days"]
            critical_days = row.get("critical_days", 0)
            if pd.isna(critical_days):
                critical_days = 0
            peak_bands = row.get("peak_crowding_bands", "")

            # Get route metrics for this route
            route_m = data["route_metrics"][data["route_metrics"]["route_id"] == route_id]
            if route_m.empty:
                continue
            rm = route_m.iloc[0]

            evidence = {
                "crowding_share": float(rm.get("crowding_share", 0)),
                "critical_crowding_share": float(rm.get("critical_crowding_share", 0)),
                "occupancy_avg_pct": float(rm.get("occupancy_avg_pct", 0)),
                "occupancy_max_pct": float(rm.get("occupancy_max_pct", 0)),
                "persistent_crowding_days": int(crowded_days),
                "critical_days": int(critical_days),
                "peak_crowding_bands": peak_bands,
                "avg_boardings_per_trip": float(rm.get("avg_boardings_per_trip", 0)),
                "capacity": float(rm.get("capacity", 0)) if "capacity" in rm else None,
            }

            priority = _determine_priority(evidence)
            action, effect = _select_action("capacity_increase", evidence)

            rec = Recommendation(
                recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
                category="capacity_increase",
                priority=priority,
                affected_route=route_id,
                affected_stop=None,
                title=f"Capacity increase for {route_id}",
                description=(
                    f"Route {route_id} has persistent overcrowding: {crowded_days} days with "
                    f"crowded share ≥20%, including {critical_days} critical days. "
                    f"Average occupancy {rm.get('occupancy_avg_pct', 0):.1f}%, "
                    f"peak {rm.get('occupancy_max_pct', 0):.1f}%."
                ),
                evidence=evidence,
                metric_value=float(rm.get("crowding_share", 0)),
                threshold_used=settings.CROWDING_HIGH,
                suggested_action=action,
                expected_effect=effect,
                confidence=0.9 if priority in ("CRITICAL", "HIGH") else 0.75,
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                model_version=None,
            )
            recs.append(rec)

        # Also check for critical crowding events not yet persistent
        if not overcrowding.empty:
            critical_events = overcrowding[overcrowding["severity"] == "critical"]
            for _, row in critical_events.iterrows():
                route_id = row["route_id"]
                if route_id in persistent["route_id"].values:
                    continue  # already covered
                route_m = data["route_metrics"][data["route_metrics"]["route_id"] == route_id]
                if route_m.empty:
                    continue
                rm = route_m.iloc[0]

                evidence = {
                    "critical_crowding_share": float(row.get("critical_share", 0)),
                    "crowding_share": float(row.get("crowded_share", 0)),
                    "occupancy_max_pct": float(row.get("avg_occupancy_max_pct", 0)),
                    "date": str(row.get("date", "")),
                    "time_band": str(row.get("time_band", "")),
                }
                priority = _determine_priority(evidence)
                action, effect = _select_action("capacity_increase", evidence)

                rec = Recommendation(
                    recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
                    category="capacity_increase",
                    priority=priority,
                    affected_route=route_id,
                    affected_stop=None,
                    title=f"Critical crowding event on {route_id}",
                    description=(
                        f"Route {route_id} on {row.get('date')} ({row.get('time_band')}) "
                        f"had critical crowding: {row.get('critical_share', 0)*100:.1f}% trips critically crowded, "
                        f"avg occupancy {row.get('avg_occupancy_max_pct', 0):.1f}%."
                    ),
                    evidence=evidence,
                    metric_value=float(row.get("critical_share", 0)),
                    threshold_used=settings.CROWDING_CRITICAL,
                    suggested_action=action,
                    expected_effect=effect,
                    confidence=0.85,
                    generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                recs.append(rec)

        return recs

    def generate_frequency_recommendations(self, data: dict) -> list[Recommendation]:
        """Generate frequency increase recommendations from bunching and demand."""
        recs = []
        bunching = data["bunching_events"]
        route_m = data["route_metrics"]

        if bunching.empty:
            return recs

        # Aggregate bunching by route
        bunch_by_route = bunching.groupby("route_id").agg(
            event_days=("date", "nunique"),
            total_bunched_trips=("bunched_trips", "sum"),
            avg_ratio=("avg_headway_ratio", "mean"),
        ).reset_index()

        for _, row in bunch_by_route.iterrows():
            route_id = row["route_id"]
            if row["event_days"] < 5:  # only persistent bunching
                continue

            rm = route_m[route_m["route_id"] == route_id]
            if rm.empty:
                continue
            rm = rm.iloc[0]

            evidence = {
                "bunching_share": float(rm.get("bunching_share", 0)),
                "bunching_event_days": int(row["event_days"]),
                "total_bunched_trips": int(row["total_bunched_trips"]),
                "avg_headway_ratio": float(row["avg_ratio"]),
                "avg_headway_min": float(rm.get("avg_headway_min", 0)),
                "headway_cv": float(rm.get("headway_cv", 0)) if pd.notna(rm.get("headway_cv")) else 0,
                "on_time_share": float(rm.get("on_time_share", 0)),
            }
            priority = _determine_priority(evidence)
            if priority == "LOW":
                continue
            action, effect = _select_action("frequency_increase", evidence)

            rec = Recommendation(
                recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
                category="frequency_increase",
                priority=priority,
                affected_route=route_id,
                affected_stop=None,
                title=f"Frequency increase for {route_id}",
                description=(
                    f"Route {route_id} has persistent bunching: {row['event_days']} days with "
                    f"bunched trips, avg headway ratio {row['avg_ratio']:.2f}. "
                    f"Bunching share {rm.get('bunching_share', 0)*100:.1f}%."
                ),
                evidence=evidence,
                metric_value=float(rm.get("bunching_share", 0)),
                threshold_used="bunching_share > 0.15",
                suggested_action=action,
                expected_effect=effect,
                confidence=0.8,
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            recs.append(rec)

        return recs

    def generate_delay_recommendations(self, data: dict) -> list[Recommendation]:
        """Generate delay reduction and schedule adjustment recommendations."""
        recs = []
        route_m = data["route_metrics"]
        quality_days = data["quality_days"]

        for _, rm in route_m.iterrows():
            route_id = rm["route_id"]
            on_time = float(rm.get("on_time_share", 0))
            avg_delay = float(rm.get("avg_delay_min", 0))
            severe_share = float(rm.get("severe_critical_share", 0))
            adherence = float(rm.get("adherence_share", 0))

            evidence = {
                "on_time_share": on_time,
                "avg_delay_min": avg_delay,
                "severe_critical_share": severe_share,
                "adherence_share": adherence,
                "p90_delay_min": float(rm.get("p90_delay_min", 0)),
            }
            priority = _determine_priority(evidence)
            if priority == "LOW" and on_time > 0.8:
                continue

            # Determine specific category
            if severe_share > 0.2 or avg_delay > 15:
                category = "delay_reduction"
                title = f"Delay reduction for {route_id}"
                desc = (
                    f"Route {route_id} has avg delay {avg_delay:.1f} min, "
                    f"on-time {on_time*100:.1f}%, severe/critical share {severe_share*100:.1f}%."
                )
            elif adherence < 0.6:
                category = "schedule_adjustment"
                title = f"Schedule adjustment for {route_id}"
                desc = (
                    f"Route {route_id} has poor schedule adherence: {adherence*100:.1f}% "
                    f"within ±2 min, avg delay {avg_delay:.1f} min."
                )
            else:
                category = "delay_reduction"
                title = f"Delay reduction for {route_id}"
                desc = (
                    f"Route {route_id} on-time {on_time*100:.1f}% (threshold "
                    f"{settings.RELIABILITY_DEGRADED*100:.0f}%), avg delay {avg_delay:.1f} min."
                )

            action, effect = _select_action(category, evidence)

            rec = Recommendation(
                recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
                category=category,
                priority=priority,
                affected_route=route_id,
                affected_stop=None,
                title=title,
                description=desc,
                evidence=evidence,
                metric_value=avg_delay,
                threshold_used=f"on_time_share < {settings.RELIABILITY_DEGRADED}",
                suggested_action=action,
                expected_effect=effect,
                confidence=0.85,
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            recs.append(rec)

        # Quality days: degraded/critical status
        if not quality_days.empty:
            degraded = quality_days[quality_days["status"].isin(["degraded", "critical"])]
            for _, row in degraded.iterrows():
                route_id = row["route_id"]
                # Check if already covered
                if any(r.affected_route == route_id and r.category in ("delay_reduction", "schedule_adjustment") for r in recs):
                    continue

                evidence = {
                    "on_time_share": float(row.get("on_time_share", 0)),
                    "avg_delay_min": float(row.get("avg_delay_min", 0)),
                    "quality_score": float(row.get("quality_score", 0)),
                    "status": str(row.get("status", "")),
                    "date": str(row.get("date", "")),
                }
                priority = "HIGH" if row.get("status") == "critical" else "MEDIUM"
                category = "operational_monitoring"
                action, effect = _select_action(category, evidence)

                rec = Recommendation(
                    recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
                    category=category,
                    priority=priority,
                    affected_route=route_id,
                    affected_stop=None,
                    title=f"Reliability monitoring for {route_id}",
                    description=(
                        f"Route {route_id} on {row.get('date')} has {row.get('status')} reliability: "
                        f"on-time {row.get('on_time_share', 0)*100:.1f}%, "
                        f"quality score {row.get('quality_score', 0):.1f}."
                    ),
                    evidence=evidence,
                    metric_value=float(row.get("quality_score", 0)),
                    threshold_used=f"RELIABILITY_DEGRADED={settings.RELIABILITY_DEGRADED}",
                    suggested_action=action,
                    expected_effect=effect,
                    confidence=0.8,
                    generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                recs.append(rec)

        return recs

    def generate_stop_recommendations(self, data: dict) -> list[Recommendation]:
        """Generate stop-level intervention recommendations."""
        recs = []
        stop_m = data["stop_metrics"]

        if stop_m.empty:
            return recs

        bottlenecks = stop_m[stop_m["is_bottleneck"] == True]
        if bottlenecks.empty:
            return recs

        # Build the stop -> routes mapping ONCE. Reading stop_facts (millions of
        # rows) inside the per-bottleneck loop re-parsed the whole parquet for
        # every stop, which made the engine take over ten minutes and never
        # finish -- the reason recommendations.parquet was never written.
        stop_facts = read_processed("stop_facts")
        if stop_facts.empty or not {"stop_id", "route_id"}.issubset(stop_facts.columns):
            routes_by_stop: dict[str, list[str]] = {}
        else:
            routes_by_stop = (
                stop_facts.groupby("stop_id")["route_id"]
                .apply(lambda s: list(pd.unique(s)))
                .to_dict()
            )

        for _, row in bottlenecks.iterrows():
            stop_id = row["stop_id"]
            routes_at_stop = routes_by_stop.get(stop_id, [])

            evidence = {
                "bottleneck_score": float(row.get("bottleneck_score", 0)),
                "avg_arr_delay_min": float(row.get("avg_arr_delay_min", 0)),
                "p95_arr_delay_min": float(row.get("p95_arr_delay_min", 0)),
                "crowding_share": float(row.get("crowding_share", 0)),
                "boardings": int(row.get("boardings", 0)),
                "peak_hour": int(row.get("peak_hour", 0)),
                "routes_serving": list(routes_at_stop[:5]),
            }
            priority = _determine_priority(evidence)
            if priority == "LOW":
                continue
            action, effect = _select_action("stop_intervention", evidence)

            rec = Recommendation(
                recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
                category="stop_intervention",
                priority=priority,
                affected_route=routes_at_stop[0] if len(routes_at_stop) > 0 else None,
                affected_stop=stop_id,
                title=f"Stop intervention at {stop_id}",
                description=(
                    f"Stop {stop_id} ({row.get('stop_name', '')}) is a bottleneck "
                    f"(score {row.get('bottleneck_score', 0):.1f}, {int(row.get('bottleneck_score', 0))}th percentile). "
                    f"Avg arrival delay {row.get('avg_arr_delay_min', 0):.1f} min, "
                    f"crowding share {row.get('crowding_share', 0)*100:.1f}%."
                ),
                evidence=evidence,
                metric_value=float(row.get("bottleneck_score", 0)),
                threshold_used=f"bottleneck_score > {BOTTLENECK_PERCENTILE*100:.0f}th percentile",
                suggested_action=action,
                expected_effect=effect,
                confidence=0.85,
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            recs.append(rec)

        return recs

    def generate_underutilized_recommendations(self, data: dict) -> list[Recommendation]:
        """Generate recommendations for underutilized routes."""
        recs = []
        underutil = data["underutilization"]
        route_m = data["route_metrics"]

        if underutil.empty:
            return recs

        # Aggregate by route
        by_route = underutil.groupby("route_id").agg(
            time_bands=("time_band", "count"),
            avg_occupancy=("avg_occupancy_pct", "mean"),
            avg_boardings=("avg_boardings", "mean"),
        ).reset_index()

        for _, row in by_route.iterrows():
            route_id = row["route_id"]
            if row["time_bands"] < 2:  # only if multiple time bands affected
                continue

            rm = route_m[route_m["route_id"] == route_id]
            if rm.empty:
                continue
            rm = rm.iloc[0]

            evidence = {
                "underutilized_share": float(rm.get("underutilized_share", 0)),
                "underutilized_time_bands": int(row["time_bands"]),
                "avg_occupancy_pct": float(row["avg_occupancy"]),
                "avg_boardings": float(row["avg_boardings"]),
                "occupancy_avg_pct": float(rm.get("occupancy_avg_pct", 0)),
                "passengers": int(rm.get("passengers", 0)),
            }
            priority = _determine_priority(evidence)
            if priority == "LOW":
                continue

            # Determine if vehicle reallocation is more appropriate
            category = "underutilized_review"
            if evidence["passengers"] < 1000:  # very low demand
                category = "vehicle_reallocation"

            action, effect = _select_action(category, evidence)

            rec = Recommendation(
                recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
                category=category,
                priority=priority,
                affected_route=route_id,
                affected_stop=None,
                title=f"Underutilized route review: {route_id}",
                description=(
                    f"Route {route_id} has {row['time_bands']} time bands with ≥50% trips underutilized. "
                    f"Avg occupancy {row['avg_occupancy']:.1f}%, avg boardings {row['avg_boardings']:.1f}."
                ),
                evidence=evidence,
                metric_value=float(rm.get("underutilized_share", 0)),
                threshold_used=f"UNDERUTILIZATION_MAX={settings.UNDERUTILIZATION_MAX}",
                suggested_action=action,
                expected_effect=effect,
                confidence=0.75,
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            recs.append(rec)

        return recs

    def generate_demand_recommendations(self, data: dict) -> list[Recommendation]:
        """Generate demand-based planning recommendations from anomalies."""
        recs = []
        anomalies = data["anomalies"]

        if anomalies.empty:
            return recs

        # Group by route
        by_route = anomalies.groupby("route_id").agg(
            anomaly_count=("date", "count"),
            high_demand_count=("anomaly_type", lambda s: (s == "high demand").sum()),
            low_demand_count=("anomaly_type", lambda s: (s == "low demand").sum()),
            max_zscore=("z_score", "max"),
        ).reset_index()

        for _, row in by_route.iterrows():
            route_id = row["route_id"]
            if row["anomaly_count"] < 3:
                continue

            evidence = {
                "anomaly_count": int(row["anomaly_count"]),
                "high_demand_anomalies": int(row["high_demand_count"]),
                "low_demand_anomalies": int(row["low_demand_count"]),
                "max_zscore": float(row["max_zscore"]),
                "demand_anomaly_z": float(row["max_zscore"]),
            }
            priority = _determine_priority(evidence)
            if priority == "LOW":
                continue

            action, effect = _select_action("demand_based_planning", evidence)

            rec = Recommendation(
                recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
                category="demand_based_planning",
                priority=priority,
                affected_route=route_id,
                affected_stop=None,
                title=f"Demand anomaly review for {route_id}",
                description=(
                    f"Route {route_id} has {row['anomaly_count']} demand anomalies "
                    f"({row['high_demand_count']} high, {row['low_demand_count']} low), "
                    f"max z-score {row['max_zscore']:.2f}."
                ),
                evidence=evidence,
                metric_value=float(row["max_zscore"]),
                threshold_used=f"ANOMALY_ZSCORE={settings.ANOMALY_ZSCORE}",
                suggested_action=action,
                expected_effect=effect,
                confidence=0.7,
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            recs.append(rec)

        return recs

    def build_all(self) -> list[Recommendation]:
        """Generate all recommendations."""
        logger.info("Loading analytics data for recommendation engine...")
        data = self.load_analytics()

        logger.info("Generating capacity recommendations...")
        self.recommendations.extend(self.generate_capacity_recommendations(data))

        logger.info("Generating frequency recommendations...")
        self.recommendations.extend(self.generate_frequency_recommendations(data))

        logger.info("Generating delay/schedule recommendations...")
        self.recommendations.extend(self.generate_delay_recommendations(data))

        logger.info("Generating stop intervention recommendations...")
        self.recommendations.extend(self.generate_stop_recommendations(data))

        logger.info("Generating underutilized route recommendations...")
        self.recommendations.extend(self.generate_underutilized_recommendations(data))

        logger.info("Generating demand-based planning recommendations...")
        self.recommendations.extend(self.generate_demand_recommendations(data))

        # Sort by priority then confidence
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        self.recommendations.sort(key=lambda r: (priority_order.get(r.priority, 4), -r.confidence))

        logger.info(f"Generated {len(self.recommendations)} recommendations")
        return self.recommendations

    def to_dataframe(self) -> pd.DataFrame:
        """Convert recommendations to DataFrame for persistence/export."""
        if not self.recommendations:
            return pd.DataFrame()
        rows = [r.to_dict() for r in self.recommendations]
        df = pd.DataFrame(rows)
        # ``threshold_used`` is float | str by design (e.g. 0.85 or
        # "on_time_share < 0.75"). Arrow cannot infer one type for such a mixed
        # object column, so the parquet write failed with ArrowInvalid and
        # recommendations.parquet was never produced. Normalise only the columns
        # that genuinely mix Python types; None values are ignored so nullable
        # string columns are left as-is.
        for col in df.columns:
            series = df[col]
            if series.dtype != object:
                continue
            kinds = {type(v).__name__ for v in series.dropna()}
            if len(kinds) > 1:
                df[col] = series.astype(str)
        return df

    def to_explainable_format(self) -> list[dict]:
        """Convert to explainable format with full rationale."""
        return [
            {
                "recommendation": r.to_dict(),
                "explanation": _build_explanation(r),
            }
            for r in self.recommendations
        ]


def build_recommendations() -> dict:
    """Main entry point: build and persist recommendations."""
    t0 = __import__("time").perf_counter()
    engine = RecommendationEngine()
    recs = engine.build_all()

    # Persist as parquet
    df = engine.to_dataframe()
    if not df.empty:
        path = write_dataset(df, DATA_ANALYTICS / "recommendations.parquet")
        # Drop the cached frame: the dashboard reruns right after regenerating
        # and would otherwise re-read the pre-regeneration results.
        from src.services.data_access import invalidate

        invalidate("recommendations")
        logger.info(f"Saved {len(df)} recommendations to {path}")

        # Also save explainable JSON
        explainable = engine.to_explainable_format()
        exp_path = DATA_ANALYTICS / "recommendations_explainable.json"
        exp_path.write_text(json.dumps(explainable, indent=2, default=str), encoding="utf-8")

    # Record audit
    record(
        action="recommendations:build",
        component="recommend",
        status="ok" if recs else "warning",
        details=json.dumps({"count": len(recs), "by_priority": _priority_counts(recs)}, default=str),
        duration_ms=round((__import__("time").perf_counter() - t0) * 1000, 1),
    )

    return {
        "count": len(recs),
        "by_priority": _priority_counts(recs),
        "by_category": _category_counts(recs),
    }


def _priority_counts(recs: list[Recommendation]) -> dict[str, int]:
    counts = {}
    for r in recs:
        counts[r.priority] = counts.get(r.priority, 0) + 1
    return counts


def _category_counts(recs: list[Recommendation]) -> dict[str, int]:
    counts = {}
    for r in recs:
        counts[r.category] = counts.get(r.category, 0) + 1
    return counts


if __name__ == "__main__":
    import time
    result = build_recommendations()
    print(json.dumps(result, indent=2, default=str))