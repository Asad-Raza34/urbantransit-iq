"""Route Comparison Tool for UrbanTransit IQ.

Allows users to compare multiple routes using actual analytics metrics.
Provides side-by-side comparison with clear metric labels and difference highlighting.
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
from src.audit import record
from src.log import get_logger
from src.paths import DATA_ANALYTICS
from src.storage import read_dataset, write_dataset

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Comparison metrics definition
# ---------------------------------------------------------------------------

# Metric display names and formatting
METRIC_SPECS = {
    "passengers": {"label": "Total Passengers", "fmt": "{:,.0f}", "higher_better": True},
    "revenue": {"label": "Revenue", "fmt": "${:,.0f}", "higher_better": True},
    "trips": {"label": "Total Trips", "fmt": "{:,.0f}", "higher_better": True},
    "avg_boardings_per_trip": {"label": "Avg Boardings/Trip", "fmt": "{:.1f}", "higher_better": True},
    "avg_delay_min": {"label": "Avg Delay (min)", "fmt": "{:.1f}", "higher_better": False},
    "p90_delay_min": {"label": "P90 Delay (min)", "fmt": "{:.1f}", "higher_better": False},
    "on_time_share": {"label": "On-Time %", "fmt": "{:.1%}", "higher_better": True},
    "adherence_share": {"label": "Adherence % (±2min)", "fmt": "{:.1%}", "higher_better": True},
    "occupancy_avg_pct": {"label": "Avg Occupancy %", "fmt": "{:.1f}%", "higher_better": None},  # target ~55%
    "occupancy_max_pct": {"label": "Max Occupancy %", "fmt": "{:.1f}%", "higher_better": False},
    "crowding_share": {"label": "Crowding Share", "fmt": "{:.1%}", "higher_better": False},
    "critical_crowding_share": {"label": "Critical Crowding %", "fmt": "{:.1%}", "higher_better": False},
    "underutilized_share": {"label": "Underutilized Share", "fmt": "{:.1%}", "higher_better": False},
    "avg_scheduled_travel_min": {"label": "Scheduled Travel (min)", "fmt": "{:.1f}", "higher_better": None},
    "avg_actual_travel_min": {"label": "Actual Travel (min)", "fmt": "{:.1f}", "higher_better": False},
    "avg_headway_min": {"label": "Avg Headway (min)", "fmt": "{:.1f}", "higher_better": None},
    "headway_std_min": {"label": "Headway Std Dev (min)", "fmt": "{:.1f}", "higher_better": False},
    "headway_cv": {"label": "Headway CV", "fmt": "{:.3f}", "higher_better": False},
    "bunching_share": {"label": "Bunching Share", "fmt": "{:.1%}", "higher_better": False},
    "gapping_share": {"label": "Gapping Share", "fmt": "{:.1%}", "higher_better": False},
    "severe_critical_share": {"label": "Severe+Critical Delay %", "fmt": "{:.1%}", "higher_better": False},
    "route_score": {"label": "Route Score (0-100)", "fmt": "{:.1f}", "higher_better": True},
    "route_rank": {"label": "Route Rank", "fmt": "{:.0f}", "higher_better": False},
    "route_class": {"label": "Classification", "fmt": "{}", "higher_better": None},
    "length_km": {"label": "Route Length (km)", "fmt": "{:.1f}", "higher_better": None},
    "category": {"label": "Category", "fmt": "{}", "higher_better": None},
}


COMPARISON_GROUPS = {
    "demand": ["passengers", "revenue", "trips", "avg_boardings_per_trip"],
    "delay": ["avg_delay_min", "p90_delay_min", "on_time_share", "adherence_share", "severe_critical_share"],
    "occupancy": ["occupancy_avg_pct", "occupancy_max_pct", "crowding_share", "critical_crowding_share", "underutilized_share"],
    "travel_time": ["avg_scheduled_travel_min", "avg_actual_travel_min"],
    "headway": ["avg_headway_min", "headway_std_min", "headway_cv", "bunching_share", "gapping_share"],
    "overall": ["route_score", "route_rank", "route_class"],
    "route_info": ["category", "length_km"],
}


def _resolve_metric_groups(metric_groups: list[str]) -> list[str]:
    """Map caller-supplied group labels onto canonical COMPARISON_GROUPS keys.

    The dashboard passes human labels such as ``"Route Info"`` while the module
    keys are ``"route_info"``. Previously that mismatch resolved to an empty
    metric list and the comparison frame lost its ``route_id`` column, raising
    ``KeyError: 'route_id'``. Accept any casing/spacing that maps to a known key.
    """
    resolved = []
    for group in metric_groups:
        key = str(group).strip().lower().replace(" ", "_").replace("-", "_")
        if key in COMPARISON_GROUPS and key not in resolved:
            resolved.append(key)
    return resolved


@dataclass
class RouteComparison:
    """Structured route comparison result."""
    comparison_id: str
    route_ids: list[str]
    metrics: dict[str, dict[str, Any]]  # route_id -> {metric -> value}
    differences: dict[str, dict[str, Any]]  # metric -> {route_id -> diff from median}
    generated_at: str
    filters: dict[str, Any] = None

    def to_dict(self) -> dict:
        return asdict(self)


def get_route_metrics(route_ids: list[str] | None = None) -> pd.DataFrame:
    """Get route metrics for specified routes (or all if None)."""
    df = read_dataset(DATA_ANALYTICS / "route_metrics.parquet")
    if route_ids:
        df = df[df["route_id"].isin(route_ids)]
    return df


def compare_routes(route_ids: list[str] | None = None,
                   metric_groups: list[str] | None = None,
                   include_all_metrics: bool = False) -> RouteComparison:
    """Compare routes across specified metric groups."""
    df = get_route_metrics(route_ids)

    if df.empty:
        return RouteComparison(
            comparison_id=f"cmp-{uuid.uuid4().hex[:8]}",
            route_ids=route_ids or [],
            metrics={},
            differences={},
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # Determine which metrics to include
    if include_all_metrics:
        metrics_to_show = [c for c in df.columns if c in METRIC_SPECS]
    elif metric_groups:
        metrics_to_show = []
        for group in _resolve_metric_groups(metric_groups):
            metrics_to_show.extend(COMPARISON_GROUPS[group])
        # De-duplicate while preserving group order, then keep only real columns.
        metrics_to_show = list(dict.fromkeys(metrics_to_show))
        metrics_to_show = [m for m in metrics_to_show if m in df.columns]
        if not metrics_to_show:
            # Unrecognised group selection: fall back to the full metric set so
            # the caller still receives a usable comparison.
            metrics_to_show = [c for c in df.columns if c in METRIC_SPECS]
    else:
        # Default: show key metrics
        metrics_to_show = [
            "route_id", "route_name", "category", "route_score", "route_class", "route_rank",
            "passengers", "revenue", "trips", "avg_boardings_per_trip",
            "avg_delay_min", "on_time_share", "adherence_share",
            "occupancy_avg_pct", "occupancy_max_pct", "crowding_share",
            "critical_crowding_share", "underutilized_share",
            "avg_headway_min", "bunching_share", "avg_actual_travel_min",
        ]
        metrics_to_show = [m for m in metrics_to_show if m in df.columns]

    # Identity columns key the comparison and are always required; a metric-group
    # selection on its own never contains them.
    identity = [c for c in ("route_id", "route_name") if c in df.columns]
    metrics_to_show = identity + [m for m in metrics_to_show if m not in identity]

    # Build comparison
    comparison_df = df[metrics_to_show].copy()
    route_ids_actual = comparison_df["route_id"].tolist()

    # Build metrics dict
    metrics_dict = {}
    for _, row in comparison_df.iterrows():
        rid = row["route_id"]
        metrics_dict[rid] = {}
        for col in metrics_to_show:
            if col == "route_id":
                continue
            val = row[col]
            if pd.isna(val):
                metrics_dict[rid][col] = None
            elif isinstance(val, (np.integer, np.floating)):
                metrics_dict[rid][col] = float(val)
            else:
                metrics_dict[rid][col] = val

    # Calculate differences from median for numeric metrics. METRIC_SPECS also
    # carries display entries for categorical metrics (route_class, category),
    # so gate on the actual dtype rather than on spec membership alone.
    differences = {}
    numeric_cols = [
        c for c in metrics_to_show
        if c in METRIC_SPECS and c in df.columns
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in numeric_cols:
        values = [metrics_dict[rid].get(col) for rid in route_ids_actual if metrics_dict[rid].get(col) is not None]
        if len(values) >= 2:
            median_val = np.median(values)
            diff_dict = {}
            for rid in route_ids_actual:
                val = metrics_dict[rid].get(col)
                if val is not None:
                    diff_dict[rid] = round(val - median_val, 3)
            differences[col] = {
                "median": round(float(median_val), 3),
                "differences": diff_dict,
                "spec": METRIC_SPECS.get(col, {}),
            }

    return RouteComparison(
        comparison_id=f"cmp-{uuid.uuid4().hex[:8]}",
        route_ids=route_ids_actual,
        metrics=metrics_dict,
        differences=differences,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def compare_routes_detailed(route_ids: list[str]) -> dict:
    """Detailed comparison with additional context from other analytics."""
    route_m = get_route_metrics(route_ids)

    # Load additional context
    stop_m = read_dataset(DATA_ANALYTICS / "stop_metrics.parquet")
    crowding = read_dataset(DATA_ANALYTICS / "overcrowding_events.parquet")
    bunching = read_dataset(DATA_ANALYTICS / "bunching_events.parquet")
    quality = read_dataset(DATA_ANALYTICS / "quality_days.parquet")
    daily_demand = read_dataset(DATA_ANALYTICS / "daily_demand.parquet")

    result = {
        "routes": {},
        "summary": {},
    }

    for _, rm in route_m.iterrows():
        rid = rm["route_id"]

        # Crowding events for this route
        route_crowding = crowding[crowding["route_id"] == rid] if not crowding.empty else pd.DataFrame()
        route_bunching = bunching[bunching["route_id"] == rid] if not bunching.empty else pd.DataFrame()
        route_quality = quality[quality["route_id"] == rid] if not quality.empty else pd.DataFrame()
        route_demand = daily_demand[daily_demand["route_id"] == rid] if not daily_demand.empty else pd.DataFrame()

        # Stop metrics for stops on this route
        route_stop_facts = read_processed("stop_facts")
        stops_on_route = route_stop_facts[route_stop_facts["route_id"] == rid]["stop_id"].unique()
        route_stop_m = stop_m[stop_m["stop_id"].isin(stops_on_route)] if not stop_m.empty else pd.DataFrame()

        result["routes"][rid] = {
            "basic": {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
                      for k, v in rm.to_dict().items()},
            "crowding_events": len(route_crowding),
            "persistent_crowding_days": int(route_crowding.groupby("date")["date"].nunique().sum()) if not route_crowding.empty else 0,
            "bunching_events": len(route_bunching),
            "avg_quality_score": float(route_quality["quality_score"].mean()) if not route_quality.empty else None,
            "reliability_status": route_quality["status"].iloc[-1] if not route_quality.empty else None,
            "avg_daily_passengers": float(route_demand["passengers"].mean()) if not route_demand.empty else 0,
            "peak_daily_passengers": float(route_demand["passengers"].max()) if not route_demand.empty else 0,
            "bottleneck_stops": int((route_stop_m["is_bottleneck"] == True).sum()) if not route_stop_m.empty else 0,
            "total_stops": int(len(stops_on_route)),
        }

    # Summary statistics
    result["summary"] = {
        "routes_compared": len(route_ids),
        "avg_route_score": float(route_m["route_score"].mean()),
        "total_passengers": int(route_m["passengers"].sum()),
        "avg_delay_min": float(route_m["avg_delay_min"].mean()),
        "avg_on_time_pct": float(route_m["on_time_share"].mean() * 100),
        "avg_occupancy_pct": float(route_m["occupancy_avg_pct"].mean()),
        "total_crowding_events": int(sum(r["crowding_events"] for r in result["routes"].values())),
        "total_bunching_events": int(sum(r["bunching_events"] for r in result["routes"].values())),
    }

    return result


def rank_routes(metric: str, ascending: bool = False, top_n: int = 10) -> pd.DataFrame:
    """Rank all routes by a specific metric."""
    df = read_dataset(DATA_ANALYTICS / "route_metrics.parquet")
    if metric not in df.columns:
        raise ValueError(f"Metric {metric} not found in route_metrics")
    return df.nlargest(top_n, metric) if not ascending else df.nsmallest(top_n, metric)


def get_route_profile(route_id: str) -> dict:
    """Get comprehensive profile for a single route."""
    route_m = get_route_metrics([route_id])
    if route_m.empty:
        return {"error": f"Route {route_id} not found"}

    rm = route_m.iloc[0]

    # Get stop-level details
    stop_facts = read_processed("stop_facts")
    stops_on_route = stop_facts[stop_facts["route_id"] == route_id]
    stop_m = read_dataset(DATA_ANALYTICS / "stop_metrics.parquet")
    route_stop_m = stop_m[stop_m["stop_id"].isin(stops_on_route["stop_id"].unique())] if not stop_m.empty else pd.DataFrame()

    # Crowding events
    crowding = read_dataset(DATA_ANALYTICS / "overcrowding_events.parquet")
    route_crowding = crowding[crowding["route_id"] == route_id] if not crowding.empty else pd.DataFrame()

    # Bunching
    bunching = read_dataset(DATA_ANALYTICS / "bunching_events.parquet")
    route_bunching = bunching[bunching["route_id"] == route_id] if not bunching.empty else pd.DataFrame()

    # Quality days
    quality = read_dataset(DATA_ANALYTICS / "quality_days.parquet")
    route_quality = quality[quality["route_id"] == route_id] if not quality.empty else pd.DataFrame()

    # Daily demand
    daily_demand = read_dataset(DATA_ANALYTICS / "daily_demand.parquet")
    route_demand = daily_demand[daily_demand["route_id"] == route_id] if not daily_demand.empty else pd.DataFrame()

    profile = {
        "route_id": route_id,
        "basic_info": {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
                       for k, v in rm.to_dict().items()},
        "performance_summary": {
            "route_score": float(rm.get("route_score", 0)),
            "route_class": str(rm.get("route_class", "")),
            "route_rank": int(rm.get("route_rank", 0)),
            "passengers": int(rm.get("passengers", 0)),
            "avg_delay_min": float(rm.get("avg_delay_min", 0)),
            "on_time_pct": float(rm.get("on_time_share", 0) * 100),
            "occupancy_avg_pct": float(rm.get("occupancy_avg_pct", 0)),
            "occupancy_max_pct": float(rm.get("occupancy_max_pct", 0)),
            "crowding_share": float(rm.get("crowding_share", 0) * 100),
            "bunching_share": float(rm.get("bunching_share", 0) * 100),
        },
        "stops": {
            "total": int(len(stops_on_route["stop_id"].unique())),
            "bottlenecks": int((route_stop_m["is_bottleneck"] == True).sum()) if not route_stop_m.empty else 0,
            "top_boarding_stops": route_stop_m.nlargest(5, "boardings")[["stop_id", "stop_name", "boardings"]].to_dict("records") if not route_stop_m.empty else [],
        },
        "crowding": {
            "events": len(route_crowding),
            "persistent_days": int(route_crowding.groupby("date")["date"].nunique().sum()) if not route_crowding.empty else 0,
            "peak_bands": route_crowding["time_band"].unique().tolist() if not route_crowding.empty else [],
            "severity_breakdown": route_crowding["severity"].value_counts().to_dict() if not route_crowding.empty else {},
        },
        "bunching": {
            "events": len(route_bunching),
            "event_days": int(route_bunching["date"].nunique()) if not route_bunching.empty else 0,
        },
        "reliability": {
            "avg_quality_score": float(route_quality["quality_score"].mean()) if not route_quality.empty else None,
            "current_status": route_quality["status"].iloc[-1] if not route_quality.empty else None,
            "degraded_days": int((route_quality["status"] == "degraded").sum()) if not route_quality.empty else 0,
            "critical_days": int((route_quality["status"] == "critical").sum()) if not route_quality.empty else 0,
        },
        "demand": {
            "avg_daily": float(route_demand["passengers"].mean()) if not route_demand.empty else 0,
            "peak_daily": float(route_demand["passengers"].max()) if not route_demand.empty else 0,
            "trend_7d": float(route_demand["passengers"].rolling(7).mean().iloc[-1]) if len(route_demand) >= 7 else None,
        },
    }

    return profile


if __name__ == "__main__":
    # Test
    df = get_route_metrics()
    print(f"Total routes: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Compare first 5 routes
    cmp = compare_routes(df["route_id"].head(5).tolist())
    print(json.dumps(cmp.to_dict(), indent=2, default=str))