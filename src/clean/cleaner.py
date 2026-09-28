"""Cleaning layer: turns the messy raw datasets into analytics-ready frames.

Every repair is recorded (dataset, step, operation, rows affected) so the
cleaning report the app shows is a genuine account of what happened to the raw
data, not a static table. The rules mirror the business rules in
``config/settings``.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from src.log import get_logger
from src.paths import DATA_RAW, DATA_CLEANED, DATA_METADATA, DATA_GENERATED
from src.storage import read_dataset, write_dataset

logger = get_logger(__name__)

LOG_COLUMNS = ["dataset", "step", "operation", "rows_affected", "details"]


class CleaningRun:
    """Tracks repair operations performed while cleaning one layer."""

    def __init__(self):
        self.log_rows = []

    def record(self, dataset, step, operation, n, details=""):
        self.log_rows.append({
            "dataset": dataset, "step": step, "operation": operation,
            "rows_affected": int(n), "details": details,
        })

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log_rows, columns=LOG_COLUMNS)


# ---------------------------------------------------------------------------
# Small reusable helpers
# ---------------------------------------------------------------------------

def _alias_to_reference(values: pd.Series, reference_ids) -> pd.Series:
    """Map messy id spellings (r001, R-001, R001) to the canonical reference id.

    Lookup is case/space/dash insensitive against the actual id list, so zero
    padding is preserved instead of being normalised away.
    """
    s = values.astype(str).str.strip()
    aliases = {}
    for v in reference_ids.astype(str):
        key = v.replace("-", "").replace(" ", "").upper()
        if key not in aliases:
            aliases[key] = v
    return s.map(lambda x: aliases.get(x.replace("-", "").replace(" ", "").upper(), np.nan))


def _to_datetimes(frame: pd.DataFrame, columns) -> None:
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce", format="mixed")


CATEGORY_FIXES = {
    "rainy": "rain", "clearr": "clear", "thunderstorm": "storm",
    "wknd": "weekend", "holliday": "holiday", "snowy": "snow",
}


def _fix_categories(frame: pd.DataFrame, columns) -> None:
    for col in columns:
        if col in frame.columns:
            frame[col] = frame[col].replace(CATEGORY_FIXES)


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

def clean_reference_tables(raw, run: CleaningRun) -> dict[str, pd.DataFrame]:
    """Deduplicate and normalise the stable reference tables."""
    cleaned = {}

    routes = raw["routes"].copy()
    before = len(routes)
    routes = routes.drop_duplicates(subset=["route_id"])
    routes["category"] = routes["category"].replace(CATEGORY_FIXES)
    cleaned["routes"] = routes
    run.record("routes", "dedupe", "drop duplicate route_id", before - len(routes))

    stops = raw["stops"].copy()
    before = len(stops)
    stops = stops.drop_duplicates(subset=["stop_id"])
    cleaned["stops"] = stops
    run.record("stops", "dedupe", "drop duplicate stop_id", before - len(stops))

    route_stops = raw["route_stops"].copy()
    before = len(route_stops)
    route_stops = route_stops.drop_duplicates(subset=["route_id", "seq"])
    valid_routes = set(routes["route_id"])
    route_stops = route_stops[route_stops["route_id"].isin(valid_routes)].reset_index(drop=True)
    cleaned["route_stops"] = route_stops
    run.record("route_stops", "dedupe/filter", "drop duplicate or dangling rows", before - len(route_stops))

    vehicles = raw["vehicles"].copy()
    before = len(vehicles)
    vehicles = vehicles.drop_duplicates(subset=["vehicle_id"])
    std_cap = {"standard": 80, "articulated": 130, "mini": 45}
    bad_cap = vehicles["capacity"].isna() | (vehicles["capacity"] <= 0)
    vehicles.loc[bad_cap, "capacity"] = vehicles.loc[bad_cap, "vehicle_type"].map(std_cap).fillna(80)
    vehicles["capacity"] = pd.to_numeric(vehicles["capacity"], errors="coerce").fillna(80).astype(int)
    cleaned["vehicles"] = vehicles
    run.record("vehicles", "repair", "fix impossible/missing capacity", int(bad_cap.sum()))

    passengers = raw["passengers"].copy()
    before = len(passengers)
    passengers = passengers.drop_duplicates(subset=["passenger_id"])
    median_birth = passengers.groupby("age_group")["birth_year"].transform("median")
    n_birth = passengers["birth_year"].isna().sum()
    passengers["birth_year"] = passengers["birth_year"].fillna(median_birth).astype(int)
    passengers["annual_trips"] = pd.to_numeric(passengers["annual_trips"], errors="coerce") \
        .fillna(1).clip(1, 400).astype(int)
    cleaned["passengers"] = passengers
    run.record("passengers", "impute", "birth_year from age-group median", n_birth)
    run.record("passengers", "dedupe", "drop duplicate passenger_id", before - len(passengers))

    calendar = raw["calendar"].copy()
    before = len(calendar)
    calendar = calendar.drop_duplicates(subset=["date"])
    calendar["weather"] = calendar["weather"].replace(CATEGORY_FIXES)
    cleaned["calendar"] = calendar
    run.record("calendar", "dedupe/categorise", "drop dups + fix weather", before - len(calendar))

    cleaned["events"] = raw["events"].copy()
    cleaned["vehicle_pools"] = raw.get("vehicle_pools", pd.DataFrame()).copy()

    return cleaned


# ---------------------------------------------------------------------------
# Trips
# ---------------------------------------------------------------------------

def clean_trips(raw, reference, run: CleaningRun) -> pd.DataFrame:
    frame = raw.copy()
    total = len(frame)

    before_ids = frame["route_id"].copy()
    frame["route_id"] = _alias_to_reference(frame["route_id"], reference["routes"]["route_id"])
    run.record("trips", "normalise", "route_id to canonical format",
               int((before_ids.astype(str) != frame["route_id"].astype(str)).sum()))

    frame = frame.drop_duplicates(subset=["trip_id"]).reset_index(drop=True)
    run.record("trips", "dedupe", "drop duplicate trip_id", total - len(frame))

    _to_datetimes(frame, ["scheduled_departure", "actual_departure"])
    n_nodate = frame["scheduled_departure"].isna().sum()
    frame = frame[frame["scheduled_departure"].notna()].reset_index(drop=True)
    frame["actual_departure"] = frame["actual_departure"].fillna(frame["scheduled_departure"])
    run.record("trips", "drop", "drop trips without a scheduled departure", n_nodate)
    frame["date"] = frame["scheduled_departure"].dt.strftime("%Y-%m-%d")

    valid_routes = set(reference["routes"]["route_id"])
    unknown = ~frame["route_id"].isin(valid_routes)
    run.record("trips", "drop", "drop trips on unknown route", int(unknown.sum()))
    frame = frame[~unknown].reset_index(drop=True)

    valid_vids = set(reference["vehicles"]["vehicle_id"])
    bad_v = ~frame["vehicle_id"].isin(valid_vids) & frame["vehicle_id"].notna()
    if bad_v.any():
        frame.loc[bad_v, "vehicle_id"] = frame.loc[bad_v, "route_id"].map(
            reference["vehicle_pools"].groupby("route_id")["vehicle_id"].first()
        ).fillna(reference["vehicles"]["vehicle_id"].iloc[0])
        run.record("trips", "repair", "vehicle_id re-aligned via route pool", int(bad_v.sum()))
    frame["vehicle_id"] = frame["vehicle_id"].fillna(reference["vehicles"]["vehicle_id"].iloc[0])

    cap_of = reference["vehicles"].set_index("vehicle_id")["capacity"]
    frame["capacity"] = frame["vehicle_id"].map(cap_of).fillna(frame["capacity"]).fillna(80).astype(int)

    for col in ["departure_delay_min", "occupancy_avg_pct", "occupancy_max_pct",
                "boardings", "alightings", "distance_km", "scheduled_travel_min",
                "actual_travel_min"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    n_delay = frame["departure_delay_min"].isna().sum()
    med_delay = frame["departure_delay_min"].median()
    frame["departure_delay_min"] = frame["departure_delay_min"].fillna(med_delay).clip(-5, 120).round(1)
    run.record("trips", "impute", "departure delay from column median", n_delay)

    n_occ = frame["occupancy_max_pct"].isna().sum() + frame["occupancy_avg_pct"].isna().sum()
    occ_med = frame.groupby("route_id")["occupancy_max_pct"].transform("median")
    frame["occupancy_max_pct"] = frame["occupancy_max_pct"].fillna(occ_med).fillna(0).clip(0, 200)
    frame["occupancy_avg_pct"] = frame["occupancy_avg_pct"].fillna(
        frame["occupancy_avg_pct"].median()).fillna(0).clip(0, 200)
    run.record("trips", "impute", "occupancy from route median", n_occ)

    n_neg = int(((frame["boardings"] < 0) | (frame["alightings"] < 0)).sum())
    frame["boardings"] = frame["boardings"].fillna(0).clip(0, 400)
    frame["alightings"] = frame["alightings"].fillna(0).clip(0, 400)
    run.record("trips", "repair", "impossible boarding/alighting counts zeroed", n_neg)

    valid_day = {"weekday", "weekend", "holiday"}
    valid_wx = {"clear", "rain", "snow", "storm"}
    n_cat = int((~frame["day_type"].isin(valid_day) | ~frame["weather"].isin(valid_wx)).sum())
    _fix_categories(frame, ["day_type", "weather"])
    frame["day_type"] = frame["day_type"].where(frame["day_type"].isin(valid_day), "weekday")
    frame["weather"] = frame["weather"].where(frame["weather"].isin(valid_wx), "clear")
    run.record("trips", "categorise", "normalise day_type/weather", n_cat)

    frame["on_time"] = frame["departure_delay_min"] <= settings.DELAY_ON_TIME_MAX
    frame = frame[list(raw.columns)].reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# Stop-level records
# ---------------------------------------------------------------------------

def clean_stop_trips(raw, reference, run: CleaningRun) -> pd.DataFrame:
    frame = raw.copy()
    total = len(frame)

    before = len(frame)
    frame = frame.drop_duplicates(subset=["trip_id", "seq"]).reset_index(drop=True)
    run.record("stop_trips", "dedupe", "drop duplicate (trip_id, seq)", before - len(frame))

    frame["stop_id"] = _alias_to_reference(frame["stop_id"], reference["stops"]["stop_id"])
    n_bad = frame["stop_id"].isna().sum()
    frame = frame[frame["stop_id"].notna()].reset_index(drop=True)
    run.record("stop_trips", "drop", "drop rows with unknown stop_id", n_bad)

    valid_trips = set(reference["trips"]["trip_id"])
    n_fk = (~frame["trip_id"].isin(valid_trips)).sum()
    frame = frame[frame["trip_id"].isin(valid_trips)].reset_index(drop=True)
    run.record("stop_trips", "drop", "drop rows on unknown trip", n_fk)

    _to_datetimes(frame, ["scheduled_arrival", "actual_arrival"])
    n_ts = (frame["scheduled_arrival"].isna() & frame["actual_arrival"].isna()).sum()
    frame = frame[~(frame["scheduled_arrival"].isna() & frame["actual_arrival"].isna())].reset_index(drop=True)
    frame["actual_arrival"] = frame["actual_arrival"].fillna(frame["scheduled_arrival"])
    n_arr = frame["actual_arrival"].isna().sum()
    frame = frame[frame["actual_arrival"].notna()].reset_index(drop=True)
    run.record("stop_trips", "drop/impute", "terminal arrival times repaired", n_ts + n_arr)

    frame["arr_delay_min"] = pd.to_numeric(frame["arr_delay_min"], errors="coerce").clip(-10, 160).round(1)
    med_arr = frame.groupby("trip_id")["arr_delay_min"].transform("median")
    n_missing = frame["arr_delay_min"].isna().sum()
    frame["arr_delay_min"] = frame["arr_delay_min"].fillna(med_arr).fillna(0).round(1)
    run.record("stop_trips", "impute", "arrival delay from trip median", n_missing)

    frame["boarded"] = pd.to_numeric(frame["boarded"], errors="coerce").fillna(0).clip(0, 300)
    frame["alighted"] = pd.to_numeric(frame["alighted"], errors="coerce").fillna(0).clip(0, 300)

    # rebuild the occupant trajectory from the boarding/alighting stream
    frame = frame.sort_values(["trip_id", "seq"]).reset_index(drop=True)
    onboard = frame.groupby("trip_id")["boarded"].cumsum() - frame.groupby("trip_id")["alighted"].cumsum()
    frame["onboard_after"] = onboard.clip(0, 500).astype(int)
    cap = frame["trip_id"].map(reference["trips"].set_index("trip_id")["capacity"]).fillna(80).clip(1)
    frame["occupancy_after"] = (frame["onboard_after"] / cap).round(3)
    run.record("stop_trips", "rebuild", "occupancy trajectory rebuilt", int(len(frame)))

    return frame


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------

def clean_tickets(raw, reference, run: CleaningRun) -> pd.DataFrame:
    frame = raw.copy()
    total = len(frame)

    before = len(frame)
    frame = frame.drop_duplicates(subset=["ticket_id"]).reset_index(drop=True)
    run.record("tickets", "dedupe", "drop duplicate ticket_id", before - len(frame))

    valid_trips = set(reference["trips"]["trip_id"])
    unknown_t = ~frame["trip_id"].isin(valid_trips)
    run.record("tickets", "drop", "drop tickets on unknown trip", int(unknown_t.sum()))
    frame = frame[~unknown_t].reset_index(drop=True)

    # the trip is the source of truth for the route a ticket belongs to
    route_of_trip = reference["trips"].set_index("trip_id")["route_id"]
    n_route = int((frame["route_id"] != frame["trip_id"].map(route_of_trip).fillna(frame["route_id"])).sum())
    frame["route_id"] = frame["trip_id"].map(route_of_trip).fillna(frame["route_id"])
    run.record("tickets", "repair", "route_id re-aligned to trip", n_route)

    _to_datetimes(frame, ["boarding_datetime", "alighting_datetime"])
    rows_before = len(frame)
    frame = frame[frame["boarding_datetime"].notna() & frame["alighting_datetime"].notna()].reset_index(drop=True)
    run.record("tickets", "drop", "drop tickets with invalid datetimes", rows_before - len(frame))

    rows_before = len(frame)
    frame = frame[frame["alighting_datetime"] > frame["boarding_datetime"]].reset_index(drop=True)
    run.record("tickets", "drop", "drop chronologically impossible tickets", rows_before - len(frame))

    dist = pd.to_numeric(frame["distance_km"], errors="coerce")
    n_dist = dist.isna().sum()
    dist_med = frame.groupby("route_id")["distance_km"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").median())
    frame["distance_km"] = dist.fillna(dist_med).clip(0.1, 100).round(2)
    run.record("tickets", "impute", "distance from route median", n_dist)

    pass_discount = frame["pass_type"].map(settings.FARE_DISCOUNT).fillna(1.0)
    fare = pd.to_numeric(frame["fare"], errors="coerce")
    n_fare = fare.isna().sum()
    recomputed = (settings.FARE_BASE + settings.FARE_PER_KM * frame["distance_km"]) * pass_discount
    frame["fare"] = fare.where(fare.notna(), recomputed).clip(0.1, 1000).round(1)
    run.record("tickets", "impute", "fare recomputed from distance", n_fare)

    frame["travel_min"] = pd.to_numeric(frame["travel_min"], errors="coerce").clip(0.1, 300).round(1)

    n_pass = frame["passenger_id"].isna().sum()
    frame = frame[frame["passenger_id"].notna()].reset_index(drop=True)
    run.record("tickets", "drop", "drop tickets without passenger", n_pass)

    frame["pass_type"] = frame["pass_type"].replace(CATEGORY_FIXES)
    frame["pass_type"] = frame["pass_type"].where(
        frame["pass_type"].isin({"single", "day", "weekly", "monthly"}), "single")
    return frame


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def clean_all(reference: dict | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Clean the whole raw layer and persist results + a cleaning report."""
    run = CleaningRun()
    raw = {}
    for name in ["routes", "stops", "route_stops", "vehicles", "vehicle_pools",
                 "passengers", "calendar", "events", "trips", "stop_trips", "tickets", "schedules"]:
        path = DATA_RAW / f"{name}.parquet"
        if path.exists():
            raw[name] = read_dataset(path)

    reference = reference or {}
    if not reference:
        for name in ["routes", "stops", "vehicles", "vehicle_pools"]:
            path = DATA_GENERATED / f"{name}.parquet"
            if path.exists():
                reference[name] = read_dataset(path)

    cleaned_ref = clean_reference_tables(raw, run)
    refs = {"routes": cleaned_ref["routes"], "stops": cleaned_ref["stops"],
            "vehicles": cleaned_ref["vehicles"],
            "vehicle_pools": reference.get("vehicle_pools", pd.DataFrame())}

    trips = clean_trips(raw["trips"], refs, run)
    refs["vehicle_pools"] = reference.get("vehicle_pools", pd.DataFrame())
    refs["trips"] = trips
    stop_trips = clean_stop_trips(raw["stop_trips"], refs, run)
    tickets = clean_tickets(raw["tickets"], refs, run)

    # reconcile trip-level occupancy with the rebuilt stop-level trajectory
    occ = stop_trips.groupby("trip_id")["occupancy_after"].agg(["mean", "max"])
    trips = trips.set_index("trip_id")
    trips["occupancy_avg_pct"] = occ["mean"].reindex(trips.index).mul(100).round(1).fillna(0)
    trips["occupancy_max_pct"] = occ["max"].reindex(trips.index).mul(100).round(1).fillna(0)
    trips = trips.reset_index()
    run.record("trips", "reconcile", "occupancy reconciled to stop-level trajectory", int(len(occ)))

    schedules = trips[["trip_id", "route_id", "direction", "date", "scheduled_departure",
                       "actual_departure", "departure_delay_min", "scheduled_headway_min",
                       "actual_headway_min"]].copy()

    cleaned = dict(cleaned_ref)
    cleaned.update({"trips": trips, "stop_trips": stop_trips, "tickets": tickets,
                    "schedules": schedules})

    DATA_CLEANED.mkdir(parents=True, exist_ok=True)
    for name, frame in cleaned.items():
        write_dataset(frame, DATA_CLEANED / f"{name}.parquet")

    report = run.frame()
    summary = _cleaning_summary(report, raw, cleaned)
    report_path = DATA_METADATA / "cleaning_report.csv"
    report.to_csv(report_path, index=False)
    (DATA_METADATA / "cleaning_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("cleaning done: %d operations, %d rows handled",
                len(report), int(report["rows_affected"].sum()))
    return cleaned, report


def _cleaning_summary(report, raw, cleaned) -> dict:
    kept = {name: len(cleaned[name]) for name in cleaned}
    before = {name: int(len(raw[name])) for name in raw if name in kept}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quality": {
            "datasets_cleaned": len(kept),
            "operations": int(len(report)),
            "total_handled_rows": int(report["rows_affected"].sum()),
        },
        "rows_before": before,
        "rows_after": kept,
    }


if __name__ == "__main__":
    cleaned, report = clean_all()
    print(report.to_string(index=False))
    print("\ntotal handled by dataset:")
    print(report.groupby("dataset")["rows_affected"].sum().to_string())