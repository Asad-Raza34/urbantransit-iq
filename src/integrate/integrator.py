"""Integration layer: joins the cleaned datasets into analytical frames.

The four outputs are the backbone for the rest of the pipeline:

* ``trip_facts``  - one row per trip enriched with route/vehicle/calendar context;
* ``stop_facts``  - one row per stop-visit enriched with trip + stop context;
* ``passenger_flow`` - one row per ticket enriched with trip + stop context;
* OD matrices      - passenger movements aggregated at route / stop / time level.

Joins are 1:1 and validated - if a merge ever multiplies rows the integrator
refuses to write and raises instead of silently corrupting the analytical set.
"""

import json
from datetime import datetime, timezone

import pandas as pd

from config import settings
from src.log import get_logger
from src.paths import DATA_CLEANED, DATA_PROCESSED, DATA_METADATA
from src.storage import read_dataset, write_dataset, mirror_to_hdfs

logger = get_logger(__name__)


def _season(month: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"}[month]


def _time_band(hour: int) -> str:
    if hour < 5:
        return "night"
    if hour < 10:
        return "morning peak"
    if hour < 16:
        return "midday"
    if hour < 20:
        return "evening peak"
    return "night"


def _trip_context(trips: pd.DataFrame) -> pd.DataFrame:
    """Calendar-derived columns shared across the analytical frames.

    ``trips`` already carries ``date`` and ``hour`` from generation, so this
    only derives the columns not yet present (avoids merge-suffix collisions).
    """
    out = trips.copy()
    d = pd.to_datetime(out["scheduled_departure"])
    out["weekday"] = d.dt.weekday  # 0 = Monday
    out["month"] = d.dt.month
    out["season"] = out["month"].map(_season)
    out["is_weekend"] = out["weekday"] >= 5
    out["time_band"] = out["hour"].map(_time_band)
    out["delay_severity"] = _delay_severity(out["departure_delay_min"])
    return out[["trip_id", "weekday", "month", "season",
                "is_weekend", "time_band", "delay_severity"]]


def _delay_severity(delay) -> pd.Series:
    """Operational delay severity buckets (the platform's single definition).

    The cut points live in :mod:`config.settings` so the machine-learning
    targets in ``src.ml.delay_classification`` cannot drift away from the
    figures the analytics dashboards and alert rules report.
    """
    return pd.cut(delay,
                  bins=[-1e9, settings.DELAY_ON_TIME_MAX, settings.DELAY_MODERATE_MAX,
                        settings.DELAY_CRITICAL_MAX, 1e9],
                  labels=["on time", "moderate", "severe", "critical"]).astype(str)


def _merge_context(left: pd.DataFrame, right: pd.DataFrame, on: str) -> pd.DataFrame:
    """Left-merge ``right`` context columns, skipping any already in ``left``.

    Prevents ``_x``/``_y`` suffixes when a key column (e.g. route_id) exists
    on both sides of the join.
    """
    keep = [c for c in right.columns if c == on or c not in left.columns]
    return left.merge(right[keep], on=on, how="left")


def integrate(cleaned: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """Build and persist the integrated analytical datasets."""
    cleaned = cleaned or {
        name: read_dataset(DATA_CLEANED / f"{name}.parquet")
        for name in ["routes", "stops", "vehicles", "calendar", "trips",
                     "stop_trips", "tickets"]
    }

    routes = cleaned["routes"]
    stops = cleaned["stops"]
    vehicles = cleaned["vehicles"]
    trips = cleaned["trips"]
    stop_trips = cleaned["stop_trips"]
    tickets = cleaned["tickets"]

    validation = {}

    # -- trip facts ---------------------------------------------------------
    route_info = routes[["route_id", "category", "length_km"]].rename(
        columns={"category": "route_category", "length_km": "route_length_km"})
    veh_info = vehicles[["vehicle_id", "vehicle_type", "capacity"]].rename(
        columns={"vehicle_type": "vehicle_type", "capacity": "capacity"})

    trip_facts = trips.merge(route_info, on="route_id", how="left")
    trip_facts = trip_facts.merge(veh_info, on="vehicle_id", how="left")
    ctx = _trip_context(trips)
    trip_facts = _merge_context(trip_facts, ctx, "trip_id")
    validation["trip_facts"] = _check_no_multiply(trips, "trip_id", trip_facts, "trip_id")

    # -- stop facts ----------------------------------------------------------
    stop_ctx = trip_facts[["trip_id", "date", "route_id", "route_category",
                           "day_type", "weather", "hour", "month", "season",
                           "is_weekend", "time_band"]]
    stop_facts = _merge_context(stop_trips, stop_ctx, "trip_id")
    stop_info = stops[["stop_id", "stop_name", "zone"]]
    stop_facts = stop_facts.merge(stop_info, on="stop_id", how="left")
    validation["stop_facts"] = _check_no_multiply(stop_trips, ["trip_id", "seq"],
                                                  stop_facts, ["trip_id", "seq"])

    # -- passenger flow ------------------------------------------------------
    flow_ctx = trip_facts[["trip_id", "date", "route_id", "route_category",
                           "day_type", "weather", "hour", "month", "season",
                           "is_weekend", "time_band"]]
    flow = _merge_context(tickets, flow_ctx, "trip_id")
    stop_info = stops[["stop_id", "zone"]].rename(columns={"zone": "boarding_zone"})
    flow = flow.merge(stop_info, left_on="boarding_stop_id", right_on="stop_id", how="left")
    flow = flow.rename(columns={"boarding_zone": "boarding_zone"})
    stop_info2 = stops[["stop_id", "zone"]].rename(columns={"zone": "alighting_zone"})
    flow = flow.merge(stop_info2, left_on="alighting_stop_id", right_on="stop_id", how="left")
    flow = flow.drop(columns=[c for c in ["stop_id_x", "stop_id_y"] if c in flow.columns])
    validation["passenger_flow"] = _check_no_multiply(tickets, "ticket_id", flow, "ticket_id")

    stop_facts["arr_delay_min"] = stop_facts["arr_delay_min"].round(1)

    # -- origin-destination matrices ----------------------------------------
    route_od = flow.groupby(["route_id", "boarding_zone", "alighting_zone"],
                            as_index=False)["ticket_id"].count()
    route_od = route_od.rename(columns={"ticket_id": "passengers"})

    stop_od = flow.groupby(["boarding_stop_id", "alighting_stop_id"],
                           as_index=False)["ticket_id"].count()
    stop_od = stop_od.rename(columns={"ticket_id": "passengers"})

    timeband_od = flow.groupby(["route_id", "time_band", "boarding_zone", "alighting_zone"],
                               as_index=False)["ticket_id"].count()
    timeband_od = timeband_od.rename(columns={"ticket_id": "passengers"})

    top_od = stop_od.sort_values("passengers", ascending=False).head(30)

    outputs = {
        "trip_facts": trip_facts,
        "stop_facts": stop_facts,
        "passenger_flow": flow,
        "od_route": route_od,
        "od_stop": stop_od,
        "od_timeband": timeband_od,
        "od_top": top_od,
    }

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        write_dataset(frame, DATA_PROCESSED / f"{name}.parquet")
        mirror_to_hdfs(frame, "processed", name)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "join_validation": validation,
        "rows": {name: int(len(df)) for name, df in outputs.items()},
    }
    (DATA_METADATA / "integration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("integration: %s", validation)
    return outputs


def _check_no_multiply(source, source_key, merged, merged_key, ) -> dict:
    """Assert the merge did not change the row count, return join diagnostics."""
    n_src = len(source)
    n_merged = len(merged)
    if n_src != n_merged:
        raise ValueError(
            f"Merge multiplied/contracted rows: source had {n_src}, merged has "
            f"{n_merged} ({source_key} vs {merged_key}) - refusing to write")
    dup = int(merged.duplicated(subset=merged_key).sum()) if merged_key else 0
    null_ctx = int(merged.isna().any(axis=1).sum())
    return {"source_rows": n_src, "merged_rows": n_merged, "duplicated_keys": dup,
            "rows_with_missing_context": null_ctx}


if __name__ == "__main__":
    outputs = integrate()
    for name, df in outputs.items():
        print(f"{name:16s} rows={len(df):9d} cols={df.shape[1]}")
    print(json.dumps(outputs["od_route"].head(5).to_dict(orient="records"), default=str))