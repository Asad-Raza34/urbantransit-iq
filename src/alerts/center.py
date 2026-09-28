"""Smart Alert Center for UrbanTransit IQ.

Generates alerts from actual analytics metrics. Alerts represent operational
conditions requiring attention, with severity, evidence, and lifecycle states.
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
# Alert definitions
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """Structured alert with full evidence and lifecycle."""
    alert_id: str
    alert_type: str
    severity: str          # critical, high, medium, low
    title: str
    description: str
    affected_route: str | None
    affected_stop: str | None
    evidence: dict[str, Any]
    metric_value: float | None
    threshold_used: float | str | None
    timestamp: str
    status: str            # new, acknowledged, resolved
    acknowledged_at: str | None = None
    resolved_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


ALERT_TYPES = {
    "severe_delay": {
        "title_template": "Severe delays on route {route_id}",
        "description_template": (
            "Route {route_id} has average delay {avg_delay:.1f} min "
            "(on-time {on_time_pct:.1f}%, severe/critical {severe_pct:.1f}%)."
        ),
        "default_severity": "high",
    },
    "persistent_overcrowding": {
        "title_template": "Persistent overcrowding on route {route_id}",
        "description_template": (
            "Route {route_id} has been overcrowded on {crowded_days} days "
            "({critical_days} critical). Peak bands: {peak_bands}."
        ),
        "default_severity": "critical",
    },
    "critical_crowding_event": {
        "title_template": "Critical crowding event on route {route_id}",
        "description_template": (
            "Route {route_id} on {date} ({time_band}) had critical crowding: "
            "{critical_share_pct:.1f}% trips critically crowded, "
            "avg occupancy {occupancy:.1f}%."
        ),
        "default_severity": "critical",
    },
    "reliability_degraded": {
        "title_template": "Reliability degraded on route {route_id}",
        "description_template": (
            "Route {route_id} on {date} has {status} reliability: "
            "on-time {on_time_pct:.1f}%, quality score {quality_score:.1f}."
        ),
        "default_severity": "high",
    },
    "stop_bottleneck": {
        "title_template": "Stop bottleneck detected at {stop_id}",
        "description_template": (
            "Stop {stop_id} ({stop_name}) is a bottleneck (score {score:.1f}, "
            "{percentile}th percentile). Avg arrival delay {delay:.1f} min, "
            "crowding share {crowding_pct:.1f}%."
        ),
        "default_severity": "high",
    },
    "demand_anomaly": {
        "title_template": "Demand anomaly on route {route_id}",
        "description_template": (
            "Route {route_id} on {date} has {anomaly_type} demand "
            "(z-score {z_score:.2f}, {passengers} passengers vs baseline {baseline:.1f})."
        ),
        "default_severity": "medium",
    },
    "underutilized_route": {
        "title_template": "Underutilized route {route_id}",
        "description_template": (
            "Route {route_id} has {underutil_bands} time bands with ≥50% trips "
            "underutilized. Avg occupancy {avg_occ:.1f}%, avg boardings {avg_boardings:.1f}."
        ),
        "default_severity": "medium",
    },
    "bunching_persistent": {
        "title_template": "Persistent bunching on route {route_id}",
        "description_template": (
            "Route {route_id} has bunching on {event_days} days. "
            "Bunching share {bunching_share_pct:.1f}%, avg headway ratio {headway_ratio:.2f}."
        ),
        "default_severity": "high",
    },
    "schedule_adherence_poor": {
        "title_template": "Poor schedule adherence on route {route_id}",
        "description_template": (
            "Route {route_id} has {adherence_pct:.1f}% adherence (±2 min), "
            "avg delay {avg_delay:.1f} min."
        ),
        "default_severity": "high",
    },
}


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
VALID_STATUSES = {"new", "acknowledged", "resolved"}


def _determine_severity(alert_type: str, evidence: dict) -> str:
    """Determine alert severity based on type and evidence."""
    base = ALERT_TYPES.get(alert_type, {}).get("default_severity", "medium")

    # Escalate based on evidence
    if alert_type == "persistent_overcrowding":
        if evidence.get("critical_days", 0) > 30:
            return "critical"
        if evidence.get("crowded_days", 0) > 20:
            return "high"
    elif alert_type == "critical_crowding_event":
        if evidence.get("critical_share_pct", 0) > 50:
            return "critical"
    elif alert_type == "severe_delay":
        if evidence.get("avg_delay", 0) > 20:
            return "critical"
        if evidence.get("severe_pct", 0) > 30:
            return "high"
    elif alert_type == "reliability_degraded":
        if evidence.get("status") == "critical":
            return "critical"
    elif alert_type == "stop_bottleneck":
        if evidence.get("score", 0) > 90:
            return "critical"
    elif alert_type == "demand_anomaly":
        if abs(evidence.get("z_score", 0)) > 4:
            return "high"

    return base


class AlertCenter:
    """Generates and manages alerts from analytics data."""

    def __init__(self):
        self.alerts: list[Alert] = []

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
        }

    def generate_delay_alerts(self, data: dict) -> list[Alert]:
        """Generate severe delay alerts from route metrics."""
        alerts = []
        route_m = data["route_metrics"]
        for _, rm in route_m.iterrows():
            route_id = rm["route_id"]
            on_time = float(rm.get("on_time_share", 0))
            avg_delay = float(rm.get("avg_delay_min", 0))
            severe_share = float(rm.get("severe_critical_share", 0))

            if on_time < settings.RELIABILITY_DEGRADED or avg_delay > 15 or severe_share > 0.2:
                evidence = {
                    "on_time_share": on_time,
                    "avg_delay_min": avg_delay,
                    "severe_critical_share": severe_share,
                    "p90_delay_min": float(rm.get("p90_delay_min", 0)),
                    "adherence_share": float(rm.get("adherence_share", 0)),
                }
                severity = _determine_severity("severe_delay", evidence)
                tpl = ALERT_TYPES["severe_delay"]

                alerts.append(Alert(
                    alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                    alert_type="severe_delay",
                    severity=severity,
                    title=tpl["title_template"].format(route_id=route_id),
                    description=tpl["description_template"].format(
                        route_id=route_id,
                        avg_delay=avg_delay,
                        on_time_pct=on_time * 100,
                        severe_pct=severe_share * 100,
                    ),
                    affected_route=route_id,
                    affected_stop=None,
                    evidence=evidence,
                    metric_value=avg_delay,
                    threshold_used=f"RELIABILITY_DEGRADED={settings.RELIABILITY_DEGRADED}",
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    status="new",
                ))
        return alerts

    def generate_crowding_alerts(self, data: dict) -> list[Alert]:
        """Generate crowding alerts from overcrowding and persistent crowding."""
        alerts = []

        # Persistent overcrowding
        persistent = data["persistent_overcrowding"]
        for _, row in persistent.iterrows():
            route_id = row["route_id"]
            critical_days = row.get("critical_days", 0)
            if pd.isna(critical_days):
                critical_days = 0
            evidence = {
                "crowded_days": int(row.get("crowded_days", 0)),
                "critical_days": int(critical_days),
                "peak_crowding_bands": str(row.get("peak_crowding_bands", "")),
                "all_crowded_trips": int(row.get("all_crowded_trips", 0)),
            }
            severity = _determine_severity("persistent_overcrowding", evidence)
            tpl = ALERT_TYPES["persistent_overcrowding"]

            alerts.append(Alert(
                alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                alert_type="persistent_overcrowding",
                severity=severity,
                title=tpl["title_template"].format(route_id=route_id),
                description=tpl["description_template"].format(
                    route_id=route_id,
                    crowded_days=row.get("crowded_days", 0),
                    critical_days=row.get("critical_days", 0),
                    peak_bands=row.get("peak_crowding_bands", ""),
                ),
                affected_route=route_id,
                affected_stop=None,
                evidence=evidence,
                metric_value=float(row.get("crowded_days", 0)),
                threshold_used=f"PERSISTENT_CROWDING_MIN_DAYS={settings.PERSISTENT_CROWDING_MIN_DAYS}",
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                status="new",
            ))

        # Critical crowding events (non-persistent)
        overcrowding = data["overcrowding_events"]
        if not overcrowding.empty:
            critical_events = overcrowding[overcrowding["severity"] == "critical"]
            # Only alert on most recent critical events per route
            for route_id in critical_events["route_id"].unique():
                route_events = critical_events[critical_events["route_id"] == route_id]
                latest = route_events.sort_values("date").iloc[-1]

                # Skip if already covered by persistent
                if route_id in persistent["route_id"].values:
                    continue

                evidence = {
                    "critical_share_pct": float(latest.get("critical_share", 0)) * 100,
                    "crowded_share_pct": float(latest.get("crowded_share", 0)) * 100,
                    "occupancy": float(latest.get("avg_occupancy_max_pct", 0)),
                }
                severity = _determine_severity("critical_crowding_event", evidence)
                tpl = ALERT_TYPES["critical_crowding_event"]

                alerts.append(Alert(
                    alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                    alert_type="critical_crowding_event",
                    severity=severity,
                    title=tpl["title_template"].format(route_id=route_id),
                    description=tpl["description_template"].format(
                        route_id=route_id,
                        date=latest.get("date", ""),
                        time_band=latest.get("time_band", ""),
                        critical_share_pct=evidence["critical_share_pct"],
                        occupancy=evidence["occupancy"],
                    ),
                    affected_route=route_id,
                    affected_stop=None,
                    evidence=evidence,
                    metric_value=evidence["critical_share_pct"],
                    threshold_used=f"CROWDING_CRITICAL={settings.CROWDING_CRITICAL}",
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    status="new",
                ))

        return alerts

    def generate_reliability_alerts(self, data: dict) -> list[Alert]:
        """Generate reliability degraded alerts from quality_days."""
        alerts = []
        quality = data["quality_days"]

        if quality.empty:
            return alerts

        # Alert on degraded/critical status
        degraded = quality[quality["status"].isin(["degraded", "critical"])]
        for _, row in degraded.iterrows():
            route_id = row["route_id"]
            evidence = {
                "on_time_share": float(row.get("on_time_share", 0)),
                "avg_delay_min": float(row.get("avg_delay_min", 0)),
                "quality_score": float(row.get("quality_score", 0)),
                "status": str(row.get("status", "")),
            }
            severity = _determine_severity("reliability_degraded", evidence)
            tpl = ALERT_TYPES["reliability_degraded"]

            alerts.append(Alert(
                alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                alert_type="reliability_degraded",
                severity=severity,
                title=tpl["title_template"].format(route_id=route_id),
                description=tpl["description_template"].format(
                    route_id=route_id,
                    date=row.get("date", ""),
                    status=row.get("status", ""),
                    on_time_pct=row.get("on_time_share", 0) * 100,
                    quality_score=row.get("quality_score", 0),
                ),
                affected_route=route_id,
                affected_stop=None,
                evidence=evidence,
                metric_value=float(row.get("quality_score", 0)),
                threshold_used=f"RELIABILITY_DEGRADED={settings.RELIABILITY_DEGRADED}",
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                status="new",
            ))

        return alerts

    def generate_stop_alerts(self, data: dict) -> list[Alert]:
        """Generate stop bottleneck alerts."""
        alerts = []
        stop_m = data["stop_metrics"]

        if stop_m.empty:
            return alerts

        bottlenecks = stop_m[stop_m["is_bottleneck"] == True]
        for _, row in bottlenecks.iterrows():
            stop_id = row["stop_id"]
            evidence = {
                "score": float(row.get("bottleneck_score", 0)),
                "avg_arr_delay_min": float(row.get("avg_arr_delay_min", 0)),
                "p95_arr_delay_min": float(row.get("p95_arr_delay_min", 0)),
                "crowding_share": float(row.get("crowding_share", 0)),
                "boardings": int(row.get("boardings", 0)),
                "percentile": int(SEVERITY_ORDER.get("high", 1) * 100 / 4 * 100),  # approx
            }
            severity = _determine_severity("stop_bottleneck", evidence)
            if severity == "low":
                continue
            tpl = ALERT_TYPES["stop_bottleneck"]

            alerts.append(Alert(
                alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                alert_type="stop_bottleneck",
                severity=severity,
                title=tpl["title_template"].format(stop_id=stop_id),
                description=tpl["description_template"].format(
                    stop_id=stop_id,
                    stop_name=row.get("stop_name", ""),
                    score=row.get("bottleneck_score", 0),
                    percentile=int(row.get("bottleneck_score", 0)),  # approximate
                    delay=row.get("avg_arr_delay_min", 0),
                    crowding_pct=row.get("crowding_share", 0) * 100,
                ),
                affected_route=None,  # multiple routes may serve
                affected_stop=stop_id,
                evidence=evidence,
                metric_value=float(row.get("bottleneck_score", 0)),
                threshold_used=f"BOTTLENECK_PERCENTILE={SEVERITY_ORDER}",
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                status="new",
            ))

        return alerts

    def generate_anomaly_alerts(self, data: dict) -> list[Alert]:
        """Generate demand anomaly alerts."""
        alerts = []
        anomalies = data["anomalies"]

        if anomalies.empty:
            return alerts

        # Alert on significant anomalies (|z| > 3)
        significant = anomalies[anomalies["z_score"].abs() > 3.0]
        for _, row in significant.iterrows():
            route_id = row["route_id"]
            evidence = {
                "z_score": float(row.get("z_score", 0)),
                "anomaly_type": str(row.get("anomaly_type", "")),
                "passengers": int(row.get("passengers", 0)),
                "date": str(row.get("date", "")),
                "is_weekend": bool(row.get("is_weekend", False)),
            }
            severity = _determine_severity("demand_anomaly", evidence)
            tpl = ALERT_TYPES["demand_anomaly"]

            alerts.append(Alert(
                alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                alert_type="demand_anomaly",
                severity=severity,
                title=tpl["title_template"].format(route_id=route_id),
                description=tpl["description_template"].format(
                    route_id=route_id,
                    date=row.get("date", ""),
                    anomaly_type=row.get("anomaly_type", ""),
                    z_score=row.get("z_score", 0),
                    passengers=int(row.get("passengers", 0)),
                    baseline=0.0,  # would need baseline calc
                ),
                affected_route=route_id,
                affected_stop=None,
                evidence=evidence,
                metric_value=float(row.get("z_score", 0)),
                threshold_used=f"ANOMALY_ZSCORE={settings.ANOMALY_ZSCORE}",
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                status="new",
            ))

        return alerts

    def generate_underutilized_alerts(self, data: dict) -> list[Alert]:
        """Generate underutilized route alerts."""
        alerts = []
        underutil = data["underutilization"]

        if underutil.empty:
            return alerts

        by_route = underutil.groupby("route_id").agg(
            underutil_bands=("time_band", "count"),
            avg_occupancy=("avg_occupancy_pct", "mean"),
            avg_boardings=("avg_boardings", "mean"),
        ).reset_index()

        for _, row in by_route.iterrows():
            if row["underutil_bands"] < 2:
                continue
            route_id = row["route_id"]
            evidence = {
                "underutil_bands": int(row["underutil_bands"]),
                "avg_occ": float(row["avg_occupancy"]),
                "avg_boardings": float(row["avg_boardings"]),
            }
            severity = _determine_severity("underutilized_route", evidence)
            if severity == "low":
                continue
            tpl = ALERT_TYPES["underutilized_route"]

            alerts.append(Alert(
                alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                alert_type="underutilized_route",
                severity=severity,
                title=tpl["title_template"].format(route_id=route_id),
                description=tpl["description_template"].format(
                    route_id=route_id,
                    underutil_bands=row["underutil_bands"],
                    avg_occ=row["avg_occupancy"],
                    avg_boardings=row["avg_boardings"],
                ),
                affected_route=route_id,
                affected_stop=None,
                evidence=evidence,
                metric_value=float(row["avg_occupancy"]),
                threshold_used=f"UNDERUTILIZATION_MAX={settings.UNDERUTILIZATION_MAX}",
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                status="new",
            ))

        return alerts

    def generate_bunching_alerts(self, data: dict) -> list[Alert]:
        """Generate persistent bunching alerts."""
        alerts = []
        bunching = data["bunching_events"]
        route_m = data["route_metrics"]

        if bunching.empty:
            return alerts

        by_route = bunching.groupby("route_id").agg(
            event_days=("date", "nunique"),
            total_bunched=("bunched_trips", "sum"),
            avg_ratio=("avg_headway_ratio", "mean"),
        ).reset_index()

        for _, row in by_route.iterrows():
            if row["event_days"] < 5:
                continue
            route_id = row["route_id"]
            rm = route_m[route_m["route_id"] == route_id]
            if rm.empty:
                continue
            rm = rm.iloc[0]

            evidence = {
                "event_days": int(row["event_days"]),
                "total_bunched": int(row["total_bunched"]),
                "headway_ratio": float(row["avg_ratio"]),
                "bunching_share_pct": float(rm.get("bunching_share", 0)) * 100,
            }
            severity = _determine_severity("bunching_persistent", evidence)
            if severity == "low":
                continue
            tpl = ALERT_TYPES["bunching_persistent"]

            alerts.append(Alert(
                alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                alert_type="bunching_persistent",
                severity=severity,
                title=tpl["title_template"].format(route_id=route_id),
                description=tpl["description_template"].format(
                    route_id=route_id,
                    event_days=row["event_days"],
                    bunching_share_pct=evidence["bunching_share_pct"],
                    headway_ratio=row["avg_ratio"],
                ),
                affected_route=route_id,
                affected_stop=None,
                evidence=evidence,
                metric_value=evidence["bunching_share_pct"],
                threshold_used="bunching_share > 0.15",
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                status="new",
            ))

        return alerts

    def build_all(self) -> list[Alert]:
        """Generate all alerts."""
        logger.info("Loading analytics data for alert center...")
        data = self.load_analytics()

        logger.info("Generating delay alerts...")
        self.alerts.extend(self.generate_delay_alerts(data))

        logger.info("Generating crowding alerts...")
        self.alerts.extend(self.generate_crowding_alerts(data))

        logger.info("Generating reliability alerts...")
        self.alerts.extend(self.generate_reliability_alerts(data))

        logger.info("Generating stop bottleneck alerts...")
        self.alerts.extend(self.generate_stop_alerts(data))

        logger.info("Generating demand anomaly alerts...")
        self.alerts.extend(self.generate_anomaly_alerts(data))

        logger.info("Generating underutilized alerts...")
        self.alerts.extend(self.generate_underutilized_alerts(data))

        logger.info("Generating bunching alerts...")
        self.alerts.extend(self.generate_bunching_alerts(data))

        # Sort by severity then timestamp
        self.alerts.sort(key=lambda a: (SEVERITY_ORDER.get(a.severity, 4), a.timestamp))

        logger.info(f"Generated {len(self.alerts)} alerts")
        return self.alerts

    def to_dataframe(self) -> pd.DataFrame:
        """Convert alerts to DataFrame for persistence."""
        if not self.alerts:
            return pd.DataFrame()
        rows = [a.to_dict() for a in self.alerts]
        return pd.DataFrame(rows)

    def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for a in self.alerts:
            if a.alert_id == alert_id:
                a.status = "acknowledged"
                a.acknowledged_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                return True
        return False

    def resolve(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for a in self.alerts:
            if a.alert_id == alert_id:
                a.status = "resolved"
                a.resolved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                return True
        return False


def build_alerts() -> dict:
    """Main entry point: build and persist alerts."""
    import time
    t0 = time.perf_counter()
    center = AlertCenter()
    alerts = center.build_all()

    # Persist as parquet
    df = center.to_dataframe()
    if not df.empty:
        path = write_dataset(df, DATA_ANALYTICS / "alerts.parquet")
        _invalidate_alerts_cache()
        logger.info(f"Saved {len(df)} alerts to {path}")

    # Record audit
    record(
        action="alerts:build",
        component="alerts",
        status="ok" if alerts else "warning",
        details=json.dumps({"count": len(alerts), "by_severity": _severity_counts(alerts)}, default=str),
        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
    )

    return {
        "count": len(alerts),
        "by_severity": _severity_counts(alerts),
        "by_type": _type_counts(alerts),
    }


def _invalidate_alerts_cache() -> None:
    """Drop the cached alerts frame so the next read sees the new file."""
    from src.services.data_access import invalidate

    invalidate("alerts")


def update_alert_status(alert_ids: list[str], status: str) -> int:
    """Persist a lifecycle transition (new -> acknowledged -> resolved).

    The dashboard's Acknowledge/Resolve controls used to only print a success
    message while writing nothing, so the documented alert lifecycle did not
    actually exist. This updates the persisted alert store and appends an audit
    entry so the transition is traceable.

    Returns the number of alerts updated.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status {status!r}; expected one of {sorted(VALID_STATUSES)}")

    alert_ids = [str(a) for a in (alert_ids or [])]
    if not alert_ids:
        return 0

    path = DATA_ANALYTICS / "alerts.parquet"
    df = read_dataset(path)
    if df.empty or "alert_id" not in df.columns:
        return 0

    mask = df["alert_id"].astype(str).isin(alert_ids)
    updated = int(mask.sum())
    if updated == 0:
        return 0

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    df.loc[mask, "status"] = status
    if status == "acknowledged":
        df.loc[mask, "acknowledged_at"] = now
    elif status == "resolved":
        df.loc[mask, "resolved_at"] = now

    write_dataset(df, path)
    # The page reruns straight after this call, and the data-access cache lives
    # for the whole server process. Without invalidation the rerun re-reads the
    # pre-write frame, so the alert keeps showing its old status even though the
    # file on disk is correct.
    _invalidate_alerts_cache()
    record(
        action=f"alerts:{status}",
        component="alerts",
        status="ok",
        details=json.dumps({"count": updated, "alert_ids": alert_ids[:50]}),
    )
    logger.info("alert status -> %s for %d alert(s)", status, updated)
    return updated


def _severity_counts(alerts: list[Alert]) -> dict[str, int]:
    counts = {}
    for a in alerts:
        counts[a.severity] = counts.get(a.severity, 0) + 1
    return counts


def _type_counts(alerts: list[Alert]) -> dict[str, int]:
    counts = {}
    for a in alerts:
        counts[a.alert_type] = counts.get(a.alert_type, 0) + 1
    return counts


if __name__ == "__main__":
    import time
    result = build_alerts()
    print(json.dumps(result, indent=2, default=str))