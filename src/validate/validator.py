"""Rule-based validation of a data layer.

The validator runs a fixed set of human-readable checks against every dataset in
a layer (raw, generated, cleaned, ...) and produces a tabular issue report plus
a JSON manifest. The checks are deliberately simple and explicit so the results
are easy to explain and to reuse in the data-quality dashboard.

Business rules (delay bounds, crowding threshold, ...) live in
``config/settings``; the checks reference them instead of re-declaring magic
numbers here.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from src.log import get_logger
from src.storage import read_dataset
from src.paths import DATA_RAW, DATA_GENERATED, DATA_CLEANED, DATA_METADATA

logger = get_logger(__name__)


@dataclass
class Layer:
    """A data layer: metadata plus the set of files that belong to it."""
    name: str
    directory: Path
    datasets: list[str] = field(default_factory=list)


def raw_layer() -> Layer:
    return Layer("raw", DATA_RAW, [
        "routes", "stops", "route_stops", "vehicles", "vehicle_pools",
        "passengers", "calendar", "events", "trips", "stop_trips", "tickets",
        "schedules",
    ])


def generated_layer() -> Layer:
    return Layer("generated", DATA_GENERATED, raw_layer().datasets)


def cleaned_layer() -> Layer:
    return Layer("cleaned", DATA_CLEANED, raw_layer().datasets)


# ---------------------------------------------------------------------------
# Issue collection helpers
# ---------------------------------------------------------------------------

ISSUE_COLUMNS = ["dataset", "column", "rule", "severity", "bad_rows", "total_rows", "sample"]


def _issues():
    return []


def _add(rows, dataset, column, rule, severity, bad, total, sample):
    if bad > 0:
        rows.append({
            "dataset": dataset, "column": column, "rule": rule, "severity": severity,
            "bad_rows": int(bad), "total_rows": int(total),
            "sample": sample[:120],
        })


def _null_count(frame, column):
    return int(frame[column].isna().sum())


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_nulls(rows, name, frame, columns, severity="warning"):
    for col in columns:
        if col in frame.columns:
            n = _null_count(frame, col)
            sample = frame.loc[frame[col].isna(), col].head(3).astype(str).tolist() if n else []
            _add(rows, name, col, "missing values", severity, n, len(frame), sample)


def _check_duplicates(rows, name, frame, keys, severity="warning"):
    if not keys or not all(k in frame.columns for k in keys):
        return
    dups = frame.duplicated(subset=keys)
    n = int(dups.sum())
    sample = frame.loc[dups, keys].head(3).values.tolist()
    _add(rows, name, "+".join(keys), "duplicate records", severity, n, len(frame), sample)


def _check_id_format(rows, name, frame, column, pattern, severity="error"):
    if column not in frame.columns:
        return
    originals = frame[column]
    vals = originals.astype(str)
    bad = ~vals.str.match(pattern) & originals.notna()
    n = int(bad.sum())
    sample = vals[bad].head(3).tolist()
    _add(rows, name, column, "malformed id", severity, n, len(frame), sample)


def _check_range(rows, name, frame, column, lo, hi, severity="error", allow_null=True,
                 label="out of range"):
    if column not in frame.columns:
        return
    col = pd.to_numeric(frame[column], errors="coerce")
    bad = col < lo
    bad |= col > hi
    if not allow_null:
        bad |= col.isna()
    n = int(bad.sum())
    sample = col[bad].dropna().head(3).round(2).tolist()
    _add(rows, name, column, label, severity, n, len(frame), sample)


def _check_categories(rows, name, frame, column, allowed, severity="warning"):
    if column not in frame.columns:
        return
    allowed = set(allowed)
    bad = ~frame[column].astype(str).isin(allowed)
    bad &= frame[column].notna()
    n = int(bad.sum())
    sample = frame.loc[bad, column].astype(str).head(3).tolist()
    _add(rows, name, column, "unknown category value", severity, n, len(frame), sample)


def _check_timestamp(rows, name, frame, column, lo, hi, severity="warning"):
    if column not in frame.columns:
        return
    col = pd.to_datetime(frame[column], errors="coerce")
    bad = col.isna()
    bad |= (col < lo) | (col > hi)
    n = int(bad.sum())
    sample = frame.loc[bad, column].astype(str).head(3).tolist()
    _add(rows, name, column, "invalid/out-of-range timestamp", severity, n, len(frame), sample)


def _check_timestamp_order(rows, name, frame, col_before, col_after, severity="error"):
    if not {col_before, col_after} <= set(frame.columns):
        return
    b = pd.to_datetime(frame[col_before], errors="coerce")
    a = pd.to_datetime(frame[col_after], errors="coerce")
    bad = (a < b) & b.notna() & a.notna()
    n = int(bad.sum())
    sample = frame.loc[bad, [col_before, col_after]].astype(str).head(3).values.tolist()
    _add(rows, name, f"{col_before} > {col_after}", "chronology violation", severity, n, len(frame), sample)


def _check_reference(rows, name, frame, column, reference, severity="error",
                     label="unknown reference value"):
    if column not in frame.columns:
        return
    ref = set(reference.astype(str))
    vals = frame[column].astype(str)
    bad = ~vals.isin(ref) & frame[column].notna()
    n = int(bad.sum())
    sample = frame.loc[bad, column].astype(str).head(3).tolist()
    _add(rows, name, column, label, severity, n, len(frame), sample)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def validate_layer(layer: Layer, reference: Layer | None = None) -> pd.DataFrame:
    """Run every applicable check against the datasets of ``layer``.

    ``reference`` (usually the generated ground truth) enables referential
    checks such as "trips.route_id exists in routes"; when it is not provided
    those checks are skipped so the validator can also run on a hidden dataset.
    """
    rows = []
    t0 = datetime.now(timezone.utc)

    frames = {}
    for name in layer.datasets:
        path = layer.directory / f"{name}.parquet"
        if not path.exists():
            rows.append({
                "dataset": name, "column": "-", "rule": "missing dataset",
                "severity": "error", "bad_rows": 1, "total_rows": 0, "sample": str(path),
            })
            continue
        try:
            frames[name] = read_dataset(path)
        except Exception as exc:  # noqa: BLE001 - report don't crash
            rows.append({
                "dataset": name, "column": "-", "rule": f"unreadable dataset ({exc})",
                "severity": "error", "bad_rows": 1, "total_rows": 0, "sample": str(path),
            })

    refs = {}
    if reference is not None:
        for name in layer.datasets:
            path = reference.directory / f"{name}.parquet"
            if path.exists():
                refs[name] = read_dataset(path)

    _validate_trips(rows, frames, refs)
    _validate_stop_trips(rows, frames, refs)
    _validate_tickets(rows, frames, refs)
    _validate_reference_tables(rows, frames)
    _validate_passengers(rows, frames)

    report = pd.DataFrame(rows, columns=ISSUE_COLUMNS)

    manifest_path = DATA_METADATA / f"validation_{layer.name}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "layer": layer.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_ms": int((datetime.now(timezone.utc) - t0).total_seconds() * 1000),
        "datasets": {name: {"rows": len(f) if not f.empty else 0} for name, f in frames.items()},
        "issue_groups": _summary(report),
        "report_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("validation[%s]: %d issues found in %d datasets",
                layer.name, len(report), len(frames))
    return report


def _summary(report: pd.DataFrame) -> dict:
    if report.empty:
        return {"total": 0, "by_severity": {}}
    return {
        "total": int(len(report)),
        "by_severity": report["severity"].value_counts().to_dict(),
    }


def _validate_reference_tables(rows, frames):
    if "routes" in frames:
        _check_nulls(rows, "routes", frames["routes"], ["route_id", "category"], "error")
        _check_duplicates(rows, "routes", frames["routes"], ["route_id"])
        _check_range(rows, "routes", frames["routes"], "length_km", 0.5, 80, allow_null=True)

    if "stops" in frames:
        _check_nulls(rows, "stops", frames["stops"], ["stop_id", "zone"], "error")
        _check_duplicates(rows, "stops", frames["stops"], ["stop_id"])
        _check_range(rows, "stops", frames["stops"], "lat", -90, 90, allow_null=False)
        _check_range(rows, "stops", frames["stops"], "lon", -180, 180, allow_null=False)

    if "route_stops" in frames:
        _check_duplicates(rows, "route_stops", frames["route_stops"], ["route_id", "seq"])
        _check_range(rows, "route_stops", frames["route_stops"], "segment_km", 0.0, 20,
                     allow_null=False)

    if "vehicles" in frames:
        _check_duplicates(rows, "vehicles", frames["vehicles"], ["vehicle_id"])
        _check_range(rows, "vehicles", frames["vehicles"], "capacity", 1, 400,
                     allow_null=False, label="impossible capacity")
        _check_categories(rows, "vehicles", frames["vehicles"], "vehicle_type",
                          {"standard", "articulated", "mini"})

    if "calendar" in frames:
        _check_duplicates(rows, "calendar", frames["calendar"], ["date"])
        _check_categories(rows, "calendar", frames["calendar"], "weather",
                          {"clear", "rain", "snow", "storm"})

    if "events" in frames:
        _check_categories(rows, "events", frames["events"], "kind",
                          {"sports", "concert", "festival", "accident", "construction"})
        _check_range(rows, "events", frames["events"], "demand_multiplier", 0.5, 3.0)

    if "schedules" in frames:
        _check_duplicates(rows, "schedules", frames["schedules"],
                          ["route_id", "direction", "scheduled_departure"])


def _validate_passengers(rows, frames):
    if "passengers" not in frames:
        return
    frame = frames["passengers"]
    _check_nulls(rows, "passengers", frame, ["passenger_id", "home_zone", "pass_type"], "error")
    _check_duplicates(rows, "passengers", frame, ["passenger_id"])
    _check_range(rows, "passengers", frame, "birth_year", 1940, 2012, allow_null=True)
    _check_range(rows, "passengers", frame, "annual_trips", 1, 400, allow_null=True)
    _check_categories(rows, "passengers", frame, "pass_type",
                      {"single", "day", "weekly", "monthly"})
    _check_categories(rows, "passengers", frame, "home_zone",
                      set(settings.ZONE_WEIGHTS) if hasattr(settings, "ZONE_WEIGHTS") else
                      {"residential", "mixed", "commercial", "university", "industrial"})


def _validate_trips(rows, frames, refs):
    if "trips" not in frames:
        return
    frame = frames["trips"]
    _check_duplicates(rows, "trips", frame, ["trip_id"], "error")
    _check_id_format(rows, "trips", frame, "trip_id", r"T-\d+")
    _check_id_format(rows, "trips", frame, "route_id", r"R-\d+")
    _check_range(rows, "trips", frame, "departure_delay_min",
                 settings.DELAY_ON_TIME_MAX - 20, 120, label="delay out of bounds")
    _check_range(rows, "trips", frame, "occupancy_max_pct", 0, 200, allow_null=True)
    _check_range(rows, "trips", frame, "occupancy_avg_pct", 0, 200, allow_null=True)
    _check_range(rows, "trips", frame, "boardings", 0, 400, allow_null=True)
    _check_range(rows, "trips", frame, "alightings", 0, 400, allow_null=True)
    _check_range(rows, "trips", frame, "capacity", 1, 400, allow_null=False)
    _check_categories(rows, "trips", frame, "day_type", {"weekday", "weekend", "holiday"})
    _check_categories(rows, "trips", frame, "weather", {"clear", "rain", "snow", "storm"})
    _check_timestamp(rows, "trips", frame, "scheduled_departure",
                     pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-02 00:00"))
    _check_timestamp(rows, "trips", frame, "actual_departure",
                     pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-02 00:00"))

    if "routes" in refs:
        _check_reference(rows, "trips", frame, "route_id", refs["routes"]["route_id"])
    if "vehicles" in refs:
        _check_reference(rows, "trips", frame, "vehicle_id", refs["vehicles"]["vehicle_id"])
    if "calendar" in refs:
        _check_reference(rows, "trips", frame, "date", pd.to_datetime(refs["calendar"]["date"]).dt.strftime("%Y-%m-%d"))


def _validate_stop_trips(rows, frames, refs):
    if "stop_trips" not in frames:
        return
    frame = frames["stop_trips"]
    _check_duplicates(rows, "stop_trips", frame, ["trip_id", "seq"], "error")
    _check_range(rows, "stop_trips", frame, "arr_delay_min", -10, 160, allow_null=True)
    _check_range(rows, "stop_trips", frame, "boarded", 0, 300, allow_null=True)
    _check_range(rows, "stop_trips", frame, "alighted", 0, 300, allow_null=True)
    _check_range(rows, "stop_trips", frame, "onboard_after", 0, 500, allow_null=False,
                 label="impossible onboard count")
    _check_range(rows, "stop_trips", frame, "seq", 1, 60, allow_null=False)
    _check_timestamp(rows, "stop_trips", frame, "scheduled_arrival",
                     pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-02 00:00"))
    _check_actual_monotonic(rows, "stop_trips", frame)

    if "trips" in refs:
        valid = set(refs["trips"]["trip_id"])
        bad = ~frame["trip_id"].isin(valid)
        _add(rows, "stop_trips", "trip_id", "unknown trip reference", "error",
             int(bad.sum()), len(frame), frame.loc[bad, "trip_id"].head(3).tolist())


def _check_actual_monotonic(rows, name, frame, severity="error"):
    """A vehicle can never "arrive" at a downstream stop before a previous one."""
    if "actual_arrival" not in frame.columns or "seq" not in frame.columns:
        return
    tmp = frame.sort_values(["trip_id", "seq"]).copy()
    a = pd.to_datetime(tmp["actual_arrival"], errors="coerce")
    prev = a.groupby(tmp["trip_id"]).shift(1)
    bad = (a < prev - pd.Timedelta("30s")) & prev.notna() & a.notna()
    n = int(bad.sum())
    sample = tmp.loc[bad, ["trip_id", "seq", "actual_arrival"]].astype(str).head(3).values.tolist()
    _add(rows, name, "actual_arrival", "arrival time moves backwards", severity, n, len(frame), sample)


def _validate_tickets(rows, frames, refs):
    if "tickets" not in frames:
        return
    frame = frames["tickets"]
    _check_duplicates(rows, "tickets", frame, ["ticket_id"] if "ticket_id" in frame.columns else ["trip_id", "boarding_datetime", "boarding_stop_id", "passenger_id"],
                      "error")
    _check_range(rows, "tickets", frame, "distance_km", 0.1, 100, allow_null=True)
    _check_range(rows, "tickets", frame, "travel_min", 0.1, 300, allow_null=True)
    _check_range(rows, "tickets", frame, "fare", 0.1, 1000, allow_null=True, label="invalid fare")
    _check_categories(rows, "tickets", frame, "pass_type",
                      {"single", "day", "weekly", "monthly"})
    _check_timestamp(rows, "tickets", frame, "boarding_datetime",
                     pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-02 00:00"))
    _check_timestamp(rows, "tickets", frame, "alighting_datetime",
                     pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-02 00:00"))
    _check_timestamp_order(rows, "tickets", frame, "boarding_datetime", "alighting_datetime")

    if "passengers" in refs:
        _check_reference(rows, "tickets", frame, "passenger_id", refs["passengers"]["passenger_id"])
    if "trips" in refs:
        _check_reference(rows, "tickets", frame, "trip_id", refs["trips"]["trip_id"])
    if "routes" in refs:
        _check_reference(rows, "tickets", frame, "route_id", refs["routes"]["route_id"])

    _check_boarding_zone(rows, frame, frames, refs)


def _check_boarding_zone(rows, frame, frames, refs):
    """boarding_stop_id/alighting_stop_id must resolve to a known stop."""
    if "stops" not in frames and "stops" not in refs:
        return
    stops = frames["stops"] if "stops" in frames else refs["stops"]
    valid = set(stops["stop_id"].astype(str))
    for col in ("boarding_stop_id", "alighting_stop_id"):
        if col not in frame.columns:
            continue
        bad = ~frame[col].astype(str).isin(valid) & frame[col].notna()
        _add(rows, "tickets", col, "unknown stop reference", "error",
             int(bad.sum()), len(frame), frame.loc[bad, col].head(3).tolist())


if __name__ == "__main__":
    report = validate_layer(raw_layer(), reference=generated_layer())
    print(report.to_string(index=False))
    print()
    print(report["rule"].value_counts().to_string())