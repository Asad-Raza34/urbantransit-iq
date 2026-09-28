"""Crowding, underutilisation and service-quality problem detection.

Produces the evidence frames that feed alerts and recommendations:

* ``overcrowding_events`` - (route, date, time-band) where a material share of
  trips ran above the crowding threshold;
* ``persistent_overcrowding`` - routes whose crowding problem is recurrent
  (settings.PERSISTENT_CROWDING_MIN_DAYS distinct event days);
* ``underutilization`` - (route, time-band) where capacity is barely used;
* ``bunching_events`` - (route, date, time-band) with collapsed headways;
* ``quality_days`` - per route per day service-quality score used to spot
  reliability degradation.
"""

import numpy as np
import pandas as pd

from config import settings
from src.analytics.core import read_processed, save
from src.log import get_logger

logger = get_logger(__name__)

CROWDED_TRIP_SHARE = 0.2       # share of crowded trips that marks an event
CRITICAL_TRIP_SHARE = 0.4
BUNCHING_RATIO = 0.5           # actual headway below 50% of scheduled = bunched
UNDERUTIL_TRIP_SHARE = 0.5


def overcrowding_events() -> pd.DataFrame:
    trips = read_processed("trip_facts")
    trips["crowded"] = trips["occupancy_max_pct"] >= settings.CROWDING_HIGH * 100
    trips["critical"] = trips["occupancy_max_pct"] >= settings.CROWDING_CRITICAL * 100

    g = trips.groupby(["route_id", "date", "time_band"], as_index=False).agg(
        trips_total=("trip_id", "count"),
        crowded_trips=("crowded", "sum"),
        critical_trips=("critical", "sum"),
        avg_occupancy_max_pct=("occupancy_max_pct", "mean"),
    )
    g["crowded_share"] = (g["crowded_trips"] / g["trips_total"]).round(3)
    g["critical_share"] = (g["critical_trips"] / g["trips_total"]).round(3)
    events = g[g["crowded_share"] >= CROWDED_TRIP_SHARE].copy()
    events["severity"] = np.where(
        (events["crowded_share"] >= CRITICAL_TRIP_SHARE) | (events["critical_trips"] > 0),
        "critical", "high")
    events["avg_occupancy_max_pct"] = events["avg_occupancy_max_pct"].round(1)
    events = events.sort_values(
        ["route_id", "date", "time_band"]).reset_index(drop=True)
    return events


def persistent_overcrowding() -> pd.DataFrame:
    events = overcrowding_events()
    days = events.groupby("route_id")["date"].nunique().rename("crowded_days") \
        .reset_index()
    persistent = events.groupby("route_id")["trips_total"].sum().rename("all_crowded_trips") \
        .reset_index().merge(days, on="route_id")
    persistent = persistent[persistent["crowded_days"] >=
                            settings.PERSISTENT_CROWDING_MIN_DAYS].copy()
    bands = events.groupby("route_id")["time_band"].agg(
        lambda s: ", ".join(sorted(set(s)))).rename("peak_crowding_bands").reset_index()
    persistent = persistent.merge(bands, on="route_id", how="left")
    persistent["critical_days"] = events[events["severity"] == "critical"] \
        .groupby("route_id")["date"].nunique().reindex(persistent["route_id"]).fillna(0).astype(int)
    return persistent.sort_values("crowded_days", ascending=False).reset_index(drop=True)


def underutilization() -> pd.DataFrame:
    trips = read_processed("trip_facts")
    trips["underused"] = trips["occupancy_max_pct"] <= settings.UNDERUTILIZATION_MAX * 100
    g = trips.groupby(["route_id", "time_band"], as_index=False).agg(
        trips_total=("trip_id", "count"),
        underused_trips=("underused", "sum"),
        avg_occupancy_pct=("occupancy_max_pct", "mean"),
        avg_boardings=("boardings", "mean"),
    )
    g["underused_share"] = (g["underused_trips"] / g["trips_total"]).round(3)
    flagged = g[g["underused_share"] >= UNDERUTIL_TRIP_SHARE].copy()
    flagged["avg_occupancy_pct"] = flagged["avg_occupancy_pct"].round(1)
    flagged["avg_boardings"] = flagged["avg_boardings"].round(1)
    return flagged.sort_values(
        ["route_id", "time_band"]).reset_index(drop=True)


def _bunching_events() -> pd.DataFrame:
    trips = read_processed("trip_facts")
    sched = trips["scheduled_headway_min"]
    actual = trips["actual_headway_min"]
    valid = sched.notna() & actual.notna() & (sched > 2)
    bunched = valid & (actual < BUNCHING_RATIO * sched)
    sub = trips[bunched].copy()
    sub["severity"] = "bunched"
    events = sub.groupby(["route_id", "date", "time_band"], as_index=False).agg(
        bunched_trips=("trip_id", "count"),
        avg_headway_ratio=("actual_headway_min", lambda s: (
            s / sub.loc[s.index, "scheduled_headway_min"]).mean()),
    )
    events["avg_headway_ratio"] = events["avg_headway_ratio"].round(3)
    return events.sort_values(
        ["route_id", "date", "time_band"]).reset_index(drop=True)


def quality_days() -> pd.DataFrame:
    """Daily reliability score = on-time share blended with delay severity."""
    trips = read_processed("trip_facts")
    g = trips.groupby(["route_id", "date"], as_index=False).agg(
        trips_total=("trip_id", "count"),
        on_time_trips=("on_time", "sum"),
        avg_delay_min=("departure_delay_min", "mean"),
        severe_trips=("delay_severity", lambda s: s.isin(["severe", "critical"]).sum()),
    )
    g["on_time_share"] = (g["on_time_trips"] / g["trips_total"]).round(4)
    g["delay_score"] = np.clip(100.0 - g["avg_delay_min"] * (100.0 / 30.0), 0, 100)
    g["quality_score"] = (0.7 * g["on_time_share"] * 100 + 0.3 * g["delay_score"]).round(1)
    g["status"] = np.select(
        [g["on_time_share"] <= settings.RELIABILITY_CRITICAL,
         g["on_time_share"] <= settings.RELIABILITY_DEGRADED],
        ["critical", "degraded"],
        default="healthy",
    )
    return g.sort_values(["route_id", "date"]).reset_index(drop=True)


def build() -> dict[str, int]:
    oe = save("overcrowding_events", overcrowding_events())
    po = save("persistent_overcrowding", persistent_overcrowding())
    uu = save("underutilization", underutilization())
    be = save("bunching_events", _bunching_events())
    qd = save("quality_days", quality_days())
    return {
        "overcrowding_events": len(oe),
        "persistent_overcrowding": len(po),
        "underutilization": len(uu),
        "bunching_events": len(be),
        "quality_days": len(qd),
    }