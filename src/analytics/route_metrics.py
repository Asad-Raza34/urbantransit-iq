"""Route-level performance analytics.

Produces ``route_metrics``, the per-route decision frame used by the route
comparison tool, route scores/classification and the recommendation engine.

Each metric is computed from the real integrated frames (``trip_facts`` +
``passenger_flow``); the outcome is a 0..100 composite ``route_score`` built
from the configured weights (``ROUTE_SCORE_WEIGHTS``) and a classification
band from ``ROUTE_CLASS_BANDS``.
"""

import numpy as np
import pandas as pd

from config import settings
from src.analytics.core import clip100, read_processed, save
from src.log import get_logger

logger = get_logger(__name__)


def _bunching_share(trips: pd.DataFrame) -> pd.DataFrame:
    """Share of trips whose actual headway collapsed vs the scheduled one."""
    sched = trips["scheduled_headway_min"]
    actual = trips["actual_headway_min"]
    valid = sched.notna() & actual.notna() & (sched > 2)
    bunched = valid & (actual < 0.5 * sched)
    gapped = valid & (actual > 1.75 * sched)
    agg = pd.DataFrame(
        {
            "route_id": trips["route_id"].to_numpy(),
            "bunched_trips": bunched,
            "gapped_trips": gapped,
            "valid_headway": valid,
        }
    ).groupby("route_id", as_index=False)[
        ["bunched_trips", "gapped_trips", "valid_headway"]
    ].sum()
    agg["bunching_share"] = (agg["bunched_trips"] / agg["valid_headway"].clip(1)).round(4)
    agg["gapping_share"] = (agg["gapped_trips"] / agg["valid_headway"].clip(1)).round(4)
    return agg


def _score_components(route: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """Normalise each route into 0..100 component scores."""
    out = route.copy()

    occ_peak = 55.0
    out["occupancy_score"] = clip100(
        100.0 - (out["occupancy_avg_pct"] - occ_peak).abs() * (100.0 / occ_peak))

    out["reliability_score"] = (out["on_time_share"] * 100.0).round(1)
    out["adherence_score"] = (out["adherence_share"] * 100.0).round(1)

    out["delay_score"] = clip100(100.0 - out["avg_delay_min"] * (100.0 / 30.0))
    out["bunching_score"] = clip100(100.0 - out["bunching_share"] * 100.0)

    demand_max = out["passengers"].max()
    out["demand_score"] = clip100(100.0 * out["passengers"] / demand_max)

    out["route_score"] = out.apply(
        lambda r: _composite(r), axis=1)
    out["route_class"] = out["route_score"].apply(_classify)
    return out


def _composite(row: pd.Series) -> float:
    parts = {
        "demand": row["demand_score"],
        "reliability": row["reliability_score"],
        "occupancy": row["occupancy_score"],
        "delay": row["delay_score"],
        "adherence": row["adherence_score"],
        "bunching": row["bunching_score"],
    }
    total = sum(settings.ROUTE_SCORE_WEIGHTS.get(k, 0.0) * v for k, v in parts.items())
    return round(total, 1)


def _classify(score: float) -> str:
    for label, lo, hi in settings.ROUTE_CLASS_BANDS:
        if lo <= score < hi:
            return label
    return "Action Required"


def compute() -> pd.DataFrame:
    trips = read_processed("trip_facts")
    flow = read_processed("passenger_flow")
    routes = _route_reference()

    volume = (flow.groupby("route_id")["ticket_id"].count()
              .rename("passengers").reset_index())
    revenue = (flow.groupby("route_id")["fare"].sum()
               .rename("revenue").reset_index())

    g = trips.groupby("route_id")
    metrics = pd.DataFrame(
        {
            "route_id": g["trip_id"].count().index,
            "trips": g["trip_id"].count().values,
            "avg_delay_min": g["departure_delay_min"].mean().values.round(2),
            "p90_delay_min": g["departure_delay_min"].quantile(0.9).values.round(2),
            "on_time_share": g["on_time"].mean().values.round(4),
            "occupancy_avg_pct": g["occupancy_avg_pct"].mean().values.round(1),
            "occupancy_max_pct": g["occupancy_max_pct"].mean().values.round(1),
            "crowding_share": g.apply(lambda s: (s["occupancy_max_pct"] >=
                                                 settings.CROWDING_HIGH * 100).mean())
            .values.round(4),
            "critical_crowding_share": g.apply(lambda s: (s["occupancy_max_pct"] >=
                                                          settings.CROWDING_CRITICAL * 100).mean())
            .values.round(4),
            "underutilized_share": g.apply(lambda s: (s["occupancy_avg_pct"] <=
                                                      settings.UNDERUTILIZATION_MAX * 100).mean())
            .values.round(4),
            "avg_scheduled_travel_min": g["scheduled_travel_min"].mean().values.round(1),
            "avg_actual_travel_min": g["actual_travel_min"].mean().values.round(1),
            "avg_headway_min": g["scheduled_headway_min"].mean().values.round(1),
            "headway_std_min": g["scheduled_headway_min"].std().fillna(0).values.round(1),
            "headway_cv": (g["scheduled_headway_min"].std() /
                           g["scheduled_headway_min"].mean().replace(0, np.nan))
            .round(3).values,
        }
    )

    agg = metrics.copy()
    sev = trips.groupby("route_id")["delay_severity"] \
        .apply(lambda s: s.isin(["severe", "critical"]).mean()).rename("severe_critical_share")
    agg["severe_critical_share"] = sev.reindex(agg["route_id"]).fillna(0).round(4).values

    # adherence: share of departures within +/- 2 minutes of schedule
    adh = (trips["departure_delay_min"].between(-2, 2)
           .groupby(trips["route_id"]).mean()
           .rename("adherence_share"))
    agg["adherence_share"] = adh.reindex(agg["route_id"]).fillna(0).round(4).values

    bunch = _bunching_share(trips).set_index("route_id")
    agg["bunching_share"] = bunch["bunching_share"].reindex(agg["route_id"]).fillna(0).values
    agg["gapping_share"] = bunch["gapping_share"].reindex(agg["route_id"]).fillna(0).values

    agg = agg.merge(volume, on="route_id", how="left")
    agg = agg.merge(revenue, on="route_id", how="left")
    agg = agg.merge(routes, on="route_id", how="left")
    agg["passengers"] = agg["passengers"].fillna(0).astype(int)
    agg["revenue"] = agg["revenue"].fillna(0).round(1)
    agg["avg_boardings_per_trip"] = (agg["passengers"] / agg["trips"]).round(2)
    agg["delay_norm"] = clip100(100.0 - agg["avg_delay_min"] * (100.0 / 30.0)).round(1)

    scored = _score_components(agg, trips)
    scored = scored.sort_values("route_score", ascending=False).reset_index(drop=True)
    scored["route_rank"] = np.arange(1, len(scored) + 1)

    order = ["route_id", "route_name", "category", "length_km", "route_rank",
             "route_score", "route_class", "passengers", "revenue", "trips",
             "avg_boardings_per_trip", "avg_delay_min", "p90_delay_min",
             "severe_critical_share", "on_time_share", "adherence_share",
             "occupancy_avg_pct", "occupancy_max_pct", "crowding_share",
             "critical_crowding_share", "underutilized_share",
             "avg_scheduled_travel_min", "avg_actual_travel_min",
             "avg_headway_min", "headway_std_min", "headway_cv",
             "bunching_share", "gapping_share", "demand_score",
             "reliability_score", "occupancy_score", "delay_score",
             "adherence_score", "bunching_score"]
    return scored[order]


def _route_reference() -> pd.DataFrame:
    routes = read_processed("trip_facts")[["route_id", "route_category", "route_length_km"]] \
        .drop_duplicates("route_id") \
        .rename(columns={"route_category": "category", "route_length_km": "length_km"})
    names = read_clean_route_names()
    return routes.merge(names, on="route_id", how="left")


def read_clean_route_names() -> pd.DataFrame:
    from src.analytics.core import read_clean
    return read_clean("routes")[["route_id", "route_name"]]


def build() -> pd.DataFrame:
    frame = compute()
    save("route_metrics", frame)
    return frame