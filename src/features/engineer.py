"""Feature engineering — deterministic, *target-free* ML inputs.

Three frames are produced from the integrated (processed) layer:

* ``trip_features``   — one row per trip for delay / on-time prediction;
* ``stop_features``   - one row per stop-visit for stop-level delay analytics;
* ``demand_features`` - route x date x time-band aggregates for demand
                        forecasting.

Design rules (mirrored by the feature manifest):

* **Target-free**: no delay / occupancy / on-time *outcome* field is read to
  build a feature, so the frames can feed supervised learners without leaking
  the label. Everything is context known at scheduled-departure time: calendar
  fields, weather snapshot, route/vehicle capability.
* **Deterministic**: same integrated frames in, identical features out. The
  manifest records the exact columns and code maps used for each frame.
* **Cyclical**: hour / weekday / month are encoded as ``(sin, cos)`` pairs so
  periodic quantities remain distance-aware for tree and linear models alike.
* **Categorical codes**: every labelled column keeps its verbatim value *and*
  a stable integer ``<column>_code`` for code-sparse learners (sklearn, Spark
  MLlib).

Outputs are written to ``data/features`` and mirrored to the logical
``features`` HDFS area; a ``feature_manifest.json`` documents every frame.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import src.paths as paths
from src.log import get_logger
from src.storage import mirror_to_hdfs, read_dataset, write_dataset

logger = get_logger(__name__)

DATA_FEATURES = paths.DATA_FEATURES
DATA_PROCESSED = paths.DATA_PROCESSED
DATA_METADATA = paths.DATA_METADATA

SEASON_CODE = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
WEATHER_CODE = {"clear": 0, "rain": 1, "snow": 2, "storm": 3}
TIME_BAND_CODE = {"night": 0, "morning peak": 1, "midday": 2, "evening peak": 3}
DAY_TYPE_CODE = {"weekday": 0, "weekend": 1, "holiday": 2}
ROUTE_CATEGORY_CODE = {"core": 0, "feeder": 1, "express": 2}
VEHICLE_TYPE_CODE = {"standard": 0, "articulated": 1, "mini": 2}
ZONE_CODE = {"residential": 0, "commercial": 1, "industrial": 2,
             "mixed": 3, "university": 4}

CODE_MAPS = {
    "season": SEASON_CODE, "weather": WEATHER_CODE,
    "time_band": TIME_BAND_CODE, "day_type": DAY_TYPE_CODE,
    "route_category": ROUTE_CATEGORY_CODE, "vehicle_type": VEHICLE_TYPE_CODE,
    "zone": ZONE_CODE,
}


def _cyclical(series: pd.Series, period: int, name: str) -> pd.DataFrame:
    """Encode a bounded periodic quantity as a (sin, cos) pair."""
    theta = 2.0 * np.pi * series.astype(float) / period
    return pd.DataFrame(
        {f"{name}_sin": np.sin(theta), f"{name}_cos": np.cos(theta)},
        index=series.index,
    )


def _cat_code(frame: pd.DataFrame, column: str, code_map: dict) -> None:
    frame[column + "_code"] = frame[column].map(code_map)


def _read_tables(full: bool = True) -> dict[str, pd.DataFrame]:
    suffix = "" if full else "_sample"
    return {
        name: read_dataset(DATA_PROCESSED / f"{name}{suffix}.parquet")
        for name in ("trip_facts", "stop_facts", "passenger_flow")
    }


def _cyclical_join(frame: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    return pd.concat([frame, _cyclical(frame[column], period, column)], axis=1)


def engineer(full: bool = True) -> dict[str, pd.DataFrame]:
    """Build the feature sets and return them for persistence.

    ``full=False`` runs against the ~1% sample partitions, handy for iterating
    during development without re-processing the 7.4M-row stop layer.
    """
    data = _read_tables(full=full)
    trips = data["trip_facts"]
    stop_facts = data["stop_facts"]
    flow = data["passenger_flow"]
    suffix = "" if full else "_sample"
    logger.info("feature engineering on sample=%s", not full)

    # ---- trip features ----------------------------------------------------- #
    trip_feats = trips.copy()
    del trip_feats["on_time"]
    dep = pd.to_datetime(trip_feats["scheduled_departure"])
    trip_feats["hour"] = dep.dt.hour
    trip_feats["weekday"] = dep.dt.weekday
    trip_feats["month"] = dep.dt.month
    trip_feats = _cyclical_join(trip_feats, "hour", 24)
    trip_feats = _cyclical_join(trip_feats, "weekday", 7)
    trip_feats = _cyclical_join(trip_feats, "month", 12)
    trip_feats["is_weekend_int"] = trip_feats["is_weekend"].astype(int)
    trip_feats["is_peak_int"] = trip_feats["is_peak"].astype(int)
    for column, code_map in CODE_MAPS.items():
        if column in trip_feats.columns:
            _cat_code(trip_feats, column, code_map)
    trip_features = trip_feats[[
        "trip_id", "route_id", "vehicle_id", "date", "day_type",
        "day_type_code", "direction", "is_peak", "is_peak_int", "hour",
        "hour_sin", "hour_cos", "weekday", "weekday_sin", "weekday_cos",
        "month", "month_sin", "month_cos", "season", "season_code", "weather",
        "weather_code", "time_band", "time_band_code", "route_category",
        "route_category_code", "vehicle_type", "vehicle_type_code",
        "distance_km", "scheduled_travel_min", "scheduled_headway_min",
        "route_length_km", "capacity_x", "capacity_y", "is_weekend",
        "is_weekend_int", "scheduled_departure",
    ]]

    # ---- stop features ----------------------------------------------------- #
    stop_feats = stop_facts.copy()
    seq_max = stop_feats.groupby("trip_id")["seq"].transform("max")
    stop_feats["position_pct"] = (100.0 * stop_feats["seq"] / seq_max).round(2)
    stop_feats["is_first"] = stop_feats["seq"].eq(1).astype(int)
    stop_feats["is_last"] = stop_feats["seq"].eq(seq_max).astype(int)
    stop_feats["hour"] = pd.to_datetime(
        stop_feats["scheduled_arrival"]).dt.hour
    stop_feats["weekday"] = pd.to_datetime(
        stop_feats["scheduled_arrival"]).dt.weekday
    stop_feats = _cyclical_join(stop_feats, "hour", 24)
    stop_feats = _cyclical_join(stop_feats, "weekday", 7)
    stop_feats["is_weekend_int"] = stop_feats["is_weekend"].astype(int)
    for column, code_map in CODE_MAPS.items():
        if column in stop_feats.columns:
            _cat_code(stop_feats, column, code_map)
    stop_features = stop_feats[[
        "trip_id", "route_id", "stop_id", "seq", "position_pct", "is_first",
        "is_last", "stop_name", "zone", "zone_code", "date", "day_type",
        "day_type_code", "weather", "weather_code", "hour", "hour_sin",
        "hour_cos", "weekday", "weekday_sin", "weekday_cos", "month",
        "season", "season_code", "time_band", "time_band_code",
        "route_category", "route_category_code", "is_weekend",
        "is_weekend_int", "boarded", "alighted", "onboard_after",
        "scheduled_arrival",
    ]]

    # ---- demand features --------------------------------------------------- #
    demand = flow.copy()
    demand["date"] = pd.to_datetime(demand["date"])
    demand = demand.groupby(
        ["route_id", "date", "time_band", "day_type", "weather", "season"],
        as_index=False,
    ).agg(
        boardings=("ticket_id", "count"),
        avg_fare=("fare", "mean"),
        passenger_km=("distance_km", "sum"),
        avg_distance_km=("distance_km", "mean"),
        avg_travel_min=("travel_min", "mean"),
    )
    demand["weekday"] = demand["date"].dt.weekday
    demand["month"] = demand["date"].dt.month
    demand = demand.sort_values(
        ["route_id", "time_band", "date"]).reset_index(drop=True)
    lag_key = ["route_id", "time_band"]
    demand["prev_week_boardings"] = (
        demand.groupby(lag_key)["boardings"].shift(7))
    demand["rolling_7d_boardings"] = (
        demand.groupby(lag_key)["boardings"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()))
    demand = _cyclical_join(demand, "weekday", 7)
    demand = _cyclical_join(demand, "month", 12)
    for column, code_map in CODE_MAPS.items():
        if column in demand.columns:
            _cat_code(demand, column, code_map)
    demand_features = demand[[
        "route_id", "date", "weekday", "weekday_sin", "weekday_cos", "month",
        "month_sin", "month_cos", "time_band", "time_band_code", "day_type",
        "day_type_code", "weather", "weather_code", "season", "season_code",
        "boardings", "avg_fare", "passenger_km", "avg_distance_km",
        "avg_travel_min", "prev_week_boardings", "rolling_7d_boardings",
    ]]

    return {
        "trip_features": trip_features,
        "stop_features": stop_features,
        "demand_features": demand_features,
    }


def persist(features: dict[str, pd.DataFrame], full: bool = True) -> None:
    """Write the feature frames, mirror to logical HDFS, and record the
    manifest documenting every column and code map used."""
    suffix = "" if full else "_sample"
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "full_scale": full,
        "code_maps": {name: code_map for name, code_map in CODE_MAPS.items()},
    }
    for name, frame in features.items():
        path = DATA_FEATURES / f"{name}{suffix}.parquet"
        write_dataset(frame, path)
        mirror_to_hdfs(frame, "features", f"{name}{suffix}", "parquet")
        manifest[name] = {
            "rows": int(len(frame)), "columns": int(frame.shape[1]),
            "cols": list(frame.columns),
        }
        logger.info("wrote %s (%d rows)", path.name, len(frame))
    manifest_path = DATA_METADATA / f"feature_manifest{suffix}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    features = engineer(full=True)
    persist(features, full=True)
    for name, frame in features.items():
        print(f"{name:18s} rows={len(frame):10d} cols={frame.shape[1]}")
