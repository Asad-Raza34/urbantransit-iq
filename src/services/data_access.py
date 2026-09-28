"""Data access layer for dashboard and services.

Provides a clean interface for loading analytics, ML, and operational data
without exposing storage details to the UI layer.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from src.analytics.core import read_processed
from src.paths import (
    DATA_ANALYTICS, DATA_PROCESSED, DATA_FEATURES,
    DATA_CLEANED, MODELS_PYTHON, MODELS_SPARK, DATA_METADATA
)
from src.storage import read_dataset

# Cache for expensive loads
_cache: dict[str, pd.DataFrame] = {}


def _cached_read(path: Path, key: str) -> pd.DataFrame:
    """Read with simple in-memory caching."""
    if key not in _cache:
        _cache[key] = read_dataset(path)
    return _cache[key]


def clear_cache() -> None:
    """Clear the data access cache."""
    global _cache
    _cache.clear()


def invalidate(*keys: str) -> None:
    """Drop specific cached datasets so the next read re-reads them from disk.

    The dashboard writes ``recommendations.parquet`` and ``alerts.parquet`` and
    then reruns. ``_cache`` lives for the whole server process, so without this
    the freshly written file stayed invisible: the generator reported
    "Acknowledged 3 alerts" while the table still showed NEW, and "Regenerate
    Alerts" / "Regenerate Recommendations" appeared to do nothing. Call this
    from every writer whose output the dashboard reads back through a loader.
    """
    for key in keys:
        _cache.pop(key, None)


# ---------------------------------------------------------------------------
# Analytics data
# ---------------------------------------------------------------------------

def load_route_metrics() -> pd.DataFrame:
    """Load route metrics (100 routes x ~35 metrics)."""
    return _cached_read(DATA_ANALYTICS / "route_metrics.parquet", "route_metrics")


def load_stop_metrics() -> pd.DataFrame:
    """Load stop metrics (408 stops x ~17 metrics)."""
    return _cached_read(DATA_ANALYTICS / "stop_metrics.parquet", "stop_metrics")


def load_stop_timeband() -> pd.DataFrame:
    """Load stop x time-band metrics (1,632 rows)."""
    return _cached_read(DATA_ANALYTICS / "stop_timeband.parquet", "stop_timeband")


def load_demand_analytics() -> dict[str, pd.DataFrame]:
    """Load all demand analytics frames."""
    return {
        "daily_demand": _cached_read(DATA_ANALYTICS / "daily_demand.parquet", "daily_demand"),
        "peak_patterns": _cached_read(DATA_ANALYTICS / "peak_patterns.parquet", "peak_patterns"),
        "anomalies": _cached_read(DATA_ANALYTICS / "anomalies.parquet", "anomalies"),
        "event_impact": _cached_read(DATA_ANALYTICS / "event_impact.parquet", "event_impact"),
        "od_zone_flow": _cached_read(DATA_ANALYTICS / "od_zone_flow.parquet", "od_zone_flow"),
    }


def load_crowding_analytics() -> dict[str, pd.DataFrame]:
    """Load all crowding analytics frames."""
    return {
        "overcrowding_events": _cached_read(DATA_ANALYTICS / "overcrowding_events.parquet", "overcrowding_events"),
        "persistent_overcrowding": _cached_read(DATA_ANALYTICS / "persistent_overcrowding.parquet", "persistent_overcrowding"),
        "underutilization": _cached_read(DATA_ANALYTICS / "underutilization.parquet", "underutilization"),
        "bunching_events": _cached_read(DATA_ANALYTICS / "bunching_events.parquet", "bunching_events"),
        "quality_days": _cached_read(DATA_ANALYTICS / "quality_days.parquet", "quality_days"),
    }


def load_segmentation() -> dict[str, pd.DataFrame]:
    """Load passenger segmentation frames."""
    return {
        "segmented_passengers": _cached_read(DATA_ANALYTICS / "segmented_passengers.parquet", "segmented_passengers"),
        "segment_summary": _cached_read(DATA_ANALYTICS / "segment_summary.parquet", "segment_summary"),
    }


def load_frequency_analytics() -> dict[str, pd.DataFrame]:
    """Load all frequency analytics frames."""
    return {
        "frequency_analysis": _cached_read(DATA_ANALYTICS / "frequency_analysis.parquet", "frequency_analysis"),
        "headway_analysis": _cached_read(DATA_ANALYTICS / "headway_analysis.parquet", "headway_analysis"),
        "service_level": _cached_read(DATA_ANALYTICS / "service_level.parquet", "service_level"),
        "frequency_peak_analysis": _cached_read(DATA_ANALYTICS / "frequency_peak_analysis.parquet", "frequency_peak_analysis"),
    }


def load_spark_analytics() -> dict[str, pd.DataFrame]:
    """Load Spark SQL analytics frames."""
    return {
        "spark_route_reliability": _cached_read(DATA_ANALYTICS / "spark_route_reliability.parquet", "spark_route_reliability"),
        "spark_stop_punctuality": _cached_read(DATA_ANALYTICS / "spark_stop_punctuality.parquet", "spark_stop_punctuality"),
    }


def load_recommendations() -> pd.DataFrame:
    """Load recommendations."""
    path = DATA_ANALYTICS / "recommendations.parquet"
    if path.exists():
        return _cached_read(path, "recommendations")
    return pd.DataFrame()


def load_recommendations_explainable() -> list[dict]:
    """Load explainable recommendations with full rationale."""
    path = DATA_ANALYTICS / "recommendations_explainable.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def load_alerts() -> pd.DataFrame:
    """Load alerts."""
    path = DATA_ANALYTICS / "alerts.parquet"
    if path.exists():
        return _cached_read(path, "alerts")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Processed/Integrated data (for detailed drill-down)
# ---------------------------------------------------------------------------

def load_trip_facts() -> pd.DataFrame:
    """Load trip facts (581K rows). Use with filters."""
    return _cached_read(DATA_PROCESSED / "trip_facts.parquet", "trip_facts")


def load_stop_facts() -> pd.DataFrame:
    """Load stop facts (6.6M rows). Use with filters."""
    return _cached_read(DATA_PROCESSED / "stop_facts.parquet", "stop_facts")


def load_passenger_flow() -> pd.DataFrame:
    """Load passenger flow (7.3M rows). Use with filters."""
    return _cached_read(DATA_PROCESSED / "passenger_flow.parquet", "passenger_flow")


def load_od_matrices() -> dict[str, pd.DataFrame]:
    """Load OD matrices."""
    return {
        "od_route": _cached_read(DATA_PROCESSED / "od_route.parquet", "od_route"),
        "od_stop": _cached_read(DATA_PROCESSED / "od_stop.parquet", "od_stop"),
        "od_timeband": _cached_read(DATA_PROCESSED / "od_timeband.parquet", "od_timeband"),
        "od_top": _cached_read(DATA_PROCESSED / "od_top.parquet", "od_top"),
    }


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def load_demand_features() -> pd.DataFrame:
    """Load demand features (129K rows)."""
    return _cached_read(DATA_FEATURES / "demand_features.parquet", "demand_features")


def load_trip_features() -> pd.DataFrame:
    """Load trip features (581K rows)."""
    return _cached_read(DATA_FEATURES / "trip_features.parquet", "trip_features")


def load_stop_features() -> pd.DataFrame:
    """Load stop features (6.6M rows)."""
    return _cached_read(DATA_FEATURES / "stop_features.parquet", "stop_features")


# ---------------------------------------------------------------------------
# ML Models and Parity
# ---------------------------------------------------------------------------

# Model cache
_model_cache: dict[str, Any] = {}
_metrics_cache: dict[str, dict] = {}


def load_sklearn_model(model_name: str = "demand_forecast") -> Any:
    """Load scikit-learn model with caching."""
    import joblib
    cache_key = f"sklearn_{model_name}"
    if cache_key not in _model_cache:
        path = MODELS_PYTHON / f"{model_name}.joblib"
        if path.exists():
            _model_cache[cache_key] = joblib.load(path)
        else:
            return None
    return _model_cache[cache_key]


def load_sklearn_metrics(model_name: str = "demand_forecast") -> dict:
    """Load sklearn model metrics with caching."""
    cache_key = f"sklearn_metrics_{model_name}"
    if cache_key not in _metrics_cache:
        path = MODELS_PYTHON / f"{model_name}_metrics.json"
        if path.exists():
            _metrics_cache[cache_key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            return {}
    return _metrics_cache[cache_key]


def load_mllib_metrics(model_name: str = "demand_forecast") -> dict:
    """Load MLlib model metrics with caching."""
    cache_key = f"mllib_metrics_{model_name}"
    if cache_key not in _metrics_cache:
        path = MODELS_SPARK / f"{model_name}_metrics.json"
        if path.exists():
            _metrics_cache[cache_key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            return {}
    return _metrics_cache[cache_key]


def load_forecast_parity() -> pd.DataFrame:
    """Load forecast parity comparison."""
    path = MODELS_PYTHON / "forecast_parity.parquet"
    if path.exists():
        return _cached_read(path, "forecast_parity")
    return pd.DataFrame()


def load_pipeline_parity() -> pd.DataFrame:
    """Load pipeline parity comparison."""
    path = MODELS_PYTHON / "pipeline_parity.parquet"
    if path.exists():
        return _cached_read(path, "pipeline_parity")
    return pd.DataFrame()


def clear_model_cache() -> None:
    """Clear model and metrics caches."""
    global _model_cache, _metrics_cache
    _model_cache.clear()
    _metrics_cache.clear()


# ---------------------------------------------------------------------------
# Audit and Metadata
# ---------------------------------------------------------------------------

def load_audit_timeline(limit: int = 500, action: str = None, status: str = None) -> pd.DataFrame:
    """Load audit timeline."""
    from src.audit import timeline
    return timeline(limit=limit, action=action, status=status)


def load_audit_summary() -> pd.DataFrame:
    """Load audit summary counts grouped by action and status."""
    from src.audit import summary
    return summary()


def load_audit_summary_by_component() -> pd.DataFrame:
    """Load audit summary counts grouped by component and status."""
    from src.audit import summary_by_component
    return summary_by_component()


def load_recent_activity(hours: int = 24) -> pd.DataFrame:
    """Load audit entries recorded in the last ``hours`` hours."""
    from src.audit import recent_activity
    return recent_activity(hours=hours)


def load_analytics_summary() -> dict:
    """Load analytics summary metadata."""
    path = DATA_METADATA / "analytics_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_cleaning_summary() -> dict:
    """Load cleaning summary."""
    path = DATA_METADATA / "cleaning_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_integration_summary() -> dict:
    """Load integration summary."""
    path = DATA_METADATA / "integration_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def load_routes_reference() -> pd.DataFrame:
    """Load routes reference from cleaned data."""
    return _cached_read(DATA_CLEANED / "routes.parquet", "routes_ref")


def load_stops_reference() -> pd.DataFrame:
    """Load stops reference from cleaned data."""
    return _cached_read(DATA_CLEANED / "stops.parquet", "stops_ref")


def load_vehicles_reference() -> pd.DataFrame:
    """Load vehicles reference from cleaned data."""
    return _cached_read(DATA_CLEANED / "vehicles.parquet", "vehicles_ref")


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_by_route(df: pd.DataFrame, route_ids: list[str]) -> pd.DataFrame:
    """Filter dataframe by route_id column if present."""
    if "route_id" in df.columns and route_ids:
        return df[df["route_id"].isin(route_ids)]
    return df


def filter_by_date_range(df: pd.DataFrame, date_col: str,
                         start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Filter dataframe by date range."""
    if date_col not in df.columns:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if start_date:
        df = df[df[date_col] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.Timestamp(end_date)]
    return df


def filter_by_severity(df: pd.DataFrame, severity_col: str,
                       severities: list[str]) -> pd.DataFrame:
    """Filter dataframe by severity values."""
    if severity_col in df.columns and severities:
        return df[df[severity_col].isin(severities)]
    return df


def filter_by_priority(df: pd.DataFrame, priority_col: str,
                       priorities: list[str]) -> pd.DataFrame:
    """Filter dataframe by priority values."""
    if priority_col in df.columns and priorities:
        return df[df[priority_col].isin(priorities)]
    return df


# ---------------------------------------------------------------------------
# KPI calculation
# ---------------------------------------------------------------------------

def calculate_kpis() -> dict[str, Any]:
    """Calculate executive KPIs from analytics."""
    route_m = load_route_metrics()
    demand = load_demand_analytics()
    crowding = load_crowding_analytics()
    alerts = load_alerts()

    daily = demand.get("daily_demand", pd.DataFrame())

    return {
        "total_routes": int(len(route_m)),
        "active_stops": int(load_stop_metrics()["stop_id"].nunique()) if not load_stop_metrics().empty else 0,
        "total_passengers_daily": float(daily.groupby("date")["passengers"].sum().mean()) if not daily.empty else 0,
        "avg_delay_min": float(route_m["avg_delay_min"].mean()),
        "avg_occupancy_pct": float(route_m["occupancy_avg_pct"].mean()),
        "on_time_pct": float(route_m["on_time_share"].mean() * 100),
        "overcrowding_events": int(len(crowding.get("overcrowding_events", pd.DataFrame()))),
        "persistent_overcrowding_routes": int(len(crowding.get("persistent_overcrowding", pd.DataFrame()))),
        "underutilized_routes": int((route_m["underutilized_share"] > 0.5).sum()),
        "active_alerts": int(len(alerts[alerts["status"] == "new"])) if not alerts.empty else 0,
        "critical_alerts": int(len(alerts[(alerts["status"] == "new") & (alerts["severity"] == "critical")])) if not alerts.empty else 0,
        "bottleneck_stops": int((load_stop_metrics()["is_bottleneck"] == True).sum()),
        "route_score_avg": float(route_m["route_score"].mean()),
    }


class DataAccess:
    """Convenience class for data access."""

    @staticmethod
    def route_metrics(route_ids: list[str] = None) -> pd.DataFrame:
        df = load_route_metrics()
        return filter_by_route(df, route_ids) if route_ids else df

    @staticmethod
    def stop_metrics(stop_ids: list[str] = None) -> pd.DataFrame:
        df = load_stop_metrics()
        if stop_ids and "stop_id" in df.columns:
            return df[df["stop_id"].isin(stop_ids)]
        return df

    @staticmethod
    def daily_demand(route_ids: list[str] = None,
                     start_date: str = None, end_date: str = None) -> pd.DataFrame:
        df = load_demand_analytics()["daily_demand"]
        df = filter_by_route(df, route_ids)
        return filter_by_date_range(df, "date", start_date, end_date)

    @staticmethod
    def overcrowding_events(route_ids: list[str] = None,
                            severity: list[str] = None) -> pd.DataFrame:
        df = load_crowding_analytics()["overcrowding_events"]
        df = filter_by_route(df, route_ids)
        return filter_by_severity(df, "severity", severity)

    @staticmethod
    def alerts(status: list[str] = None, severity: list[str] = None) -> pd.DataFrame:
        df = load_alerts()
        df = filter_by_severity(df, "status", status)
        return filter_by_severity(df, "severity", severity)

    @staticmethod
    def recommendations(priority: list[str] = None,
                        category: list[str] = None) -> pd.DataFrame:
        df = load_recommendations()
        df = filter_by_priority(df, "priority", priority)
        if category and "category" in df.columns:
            df = df[df["category"].isin(category)]
        return df

    @staticmethod
    def kpis() -> dict:
        return calculate_kpis()