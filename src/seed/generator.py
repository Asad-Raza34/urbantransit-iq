"""Synthetic data generation for UrbanTransit IQ.

:func:`generate` builds every dataset to the configured scale and writes two
layers to disk:

* ``data/generated`` - the clean ground-truth frames;
* ``data/raw``       - the same records with controlled quality problems.

Both layers are Parquet. Deterministic seeds keep the output reproducible.
"""

import time

import pandas as pd

from src.audit import record
from src.config import settings
from src.log import get_logger
from src.paths import DATA_GENERATED, DATA_RAW
from src.seed.context import build_calendar
from src.seed.network import build_routes, build_stops
from src.seed.passengers import build_passengers
from src.seed.quality import introduce_problems, manifest
from src.seed.trips import build_route
from src.seed.vehicles import build_vehicles
from src.storage import write_dataset

logger = get_logger(__name__)


def generate(seed: int = None, scale: dict = None, months: int = None,
             to_disk: bool = True) -> dict:
    """Generate the full synthetic dataset.

    Returns a dict mapping dataset name -> dataframe and, when ``to_disk`` is
    true, also persists the generated (clean) and raw (corrupted) copies.
    """
    t0 = time.perf_counter()
    seed = seed or settings.GEN_SEED
    scale = scale or dict(settings.GEN_SCALE)
    months = months or settings.GEN_MONTHS

    days = _days_for_months(months)

    stops = build_stops(scale["stops"], seed)
    routes, route_stops = build_routes(scale["routes"], stops, seed)
    vehicles, pools = build_vehicles(scale["vehicles"], routes, seed + 3)
    passengers = build_passengers(scale["passengers"], stops, seed + 5)
    calendar = build_calendar(days=days, seed=seed + 7)
    events = _build_events(calendar, seed + 9)

    trips_all, stop_all, tickets_all = [], [], []
    trip_start = 0
    for _, route in routes.iterrows():
        trips, stop_trips, tickets, trip_start = build_route(
            route, route_stops, stops, vehicles, pools, passengers,
            calendar, events, seed, trip_start,
        )
        if trips.empty:
            continue
        trips_all.append(trips)
        stop_all.append(stop_trips)
        tickets_all.append(tickets)
        if len(trips_all) % 20 == 0:
            logger.info("generated routes: %d (trips so far %d)", len(trips_all), trip_start)

    trips = pd.concat(trips_all, ignore_index=True)
    stop_trips = pd.concat(stop_all, ignore_index=True)
    tickets = pd.concat(tickets_all, ignore_index=True)
    tickets["ticket_id"] = [f"TK-{i + 1:09d}" for i in range(len(tickets))]
    tickets = tickets[["ticket_id", "trip_id", "route_id", "boarding_stop_id",
                        "alighting_stop_id", "passenger_id", "pass_type",
                        "boarding_datetime", "alighting_datetime",
                        "distance_km", "travel_min", "fare"]]
    schedules = _schedule_view(trips)

    frames = {
        "routes": routes,
        "stops": stops,
        "route_stops": route_stops,
        "vehicles": vehicles,
        "vehicle_pools": pools,
        "passengers": passengers,
        "calendar": calendar,
        "events": events,
        "trips": trips,
        "stop_trips": stop_trips,
        "tickets": tickets,
        "schedules": schedules,
    }

    logger.info(
        "generated: %d trips, %d stop-level observations, %d tickets, %d passengers",
        len(trips), len(stop_trips), len(tickets), len(passengers),
    )

    if to_disk:
        _write_layers(frames, seed)

    record("dataset_generation", "seed", details=_summary(frames, seed),
           duration_ms=(time.perf_counter() - t0) * 1000)
    return frames


def _write_layers(frames, seed):
    raw_frames = introduce_problems(frames, seed + 11)
    for name, frame in frames.items():
        write_dataset(frame, DATA_GENERATED / f"{name}.parquet")
    for name, frame in raw_frames.items():
        write_dataset(frame, DATA_RAW / f"{name}.parquet")
    write_dataset(manifest(), DATA_RAW / "quality_manifest.csv")
    logger.info("wrote clean + raw layers for %d datasets", len(frames))


def _summary(frames, seed) -> str:
    counts = {k: len(v) for k, v in frames.items() if isinstance(v, pd.DataFrame)}
    return f"seed={seed} rows={counts}"


def _days_for_months(months: int) -> int:
    """Cumulative day count for the first ``months`` months of 2024."""
    month_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return sum(month_days[:months])


def _schedule_view(trips) -> pd.DataFrame:
    return trips[["trip_id", "route_id", "date", "direction", "hour", "is_peak",
                  "scheduled_departure", "scheduled_headway_min",
                  "actual_departure", "actual_headway_min"]].rename(
        columns={"actual_departure": "departure_observed"}
    )


def _build_events(calendar, seed):
    from src.seed.context import build_events

    return build_events(calendar, seed)