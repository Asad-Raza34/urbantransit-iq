"""Intentional data-quality problems.

The SRS requires the raw data to contain realistic quality issues so the
cleaning layer has genuine work to do. This module introduces a *controlled*
set of problems into otherwise-clean generated records; the clean originals
are kept in ``data/generated`` as ground truth and the corrupted copies become
``data/raw``. The exact problems and their rates live here so they can be
reviewed and reproduced.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.log import get_logger

logger = get_logger(__name__)

FRAME_MISSING = {
    "trips": ["departure_delay_min", "occupancy_max_pct", "boardings"],
    "stop_trips": ["arr_delay_min", "boarded"],
    "tickets": ["passenger_id", "fare", "alighting_stop_id"],
    "vehicles": ["capacity"],
    "passengers": ["birth_year"],
}
FRAME_INVALID = {
    "trips": ["occupancy_max_pct"],
    "vehicles": ["capacity"],
    "stop_trips": ["arr_delay_min", "onboard_after"],
    "tickets": ["distance_km", "travel_min"],
}
FRAME_TIMESTAMPS = {
    "trips": ["scheduled_departure", "actual_departure"],
    "tickets": ["boarding_datetime", "alighting_datetime"],
}
FRAME_IDS = {
    "trips": "route_id",
    "stop_trips": "stop_id",
    "tickets": "route_id",
}
FRAME_CATEGORIES = {"trips": ["weather", "day_type"], "stop_trips": []}
FRAME_DUPLICATES = {
    "trips": 0.004,
    "stop_trips": 0.002,
    "tickets": 0.004,
    "passengers": 0.001,
}


def introduce_problems(frames: dict[str, pd.DataFrame], seed: int) -> dict[str, pd.DataFrame]:
    """Return corrupted copies of the clean frames (the raw layer).

    ``frames`` maps dataset name -> clean dataframe. The returned dict mirrors
    the same keys with quality problems applied and shares nothing mutated.
    """
    rng = np.random.default_rng(seed)
    raw = {}
    for name, frame in frames.items():
        out = frame.copy()
        if name in FRAME_DUPLICATES:
            out = _duplicate_rows(out, FRAME_DUPLICATES[name], rng)
        if name in FRAME_MISSING:
            out = _missing_values(out, FRAME_MISSING[name], rng)
        if name in FRAME_INVALID:
            out = _invalid_values(out, FRAME_INVALID[name], rng)
        if name in FRAME_TIMESTAMPS:
            out = _invalid_timestamps(out, FRAME_TIMESTAMPS[name], rng)
        if name in FRAME_IDS:
            out = _inconsistent_ids(out, FRAME_IDS[name], rng)
        if name in FRAME_CATEGORIES:
            out = _inconsistent_categories(out, FRAME_CATEGORIES[name], rng)
        if name == "tickets":
            out = _wrong_route_links(out, frames["trips"], rng)
        raw[name] = out
        logger.info("quality problems injected into %s (%d rows)", name, len(out))
    return raw


def _missing_values(frame, columns, rng, rate=0.004):
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            continue
        n = int(len(out) * rate)
        idx = rng.choice(len(out), size=n, replace=False)
        out.loc[idx, col] = np.nan
    return out


def _invalid_values(frame, columns, rng, rate=0.003):
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_bool_dtype(out[col]):
            continue
        n = int(len(out) * rate)
        if n == 0:
            continue
        idx = rng.choice(len(out), size=n, replace=False)
        if pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].astype("int64")
            out.loc[idx, col] = rng.integers(-80, -1, size=n)
        else:
            out.loc[idx, col] = rng.choice([-500.0, 9999.0, -1.0], size=n)
    return out


def _invalid_timestamps(frame, columns, rng, rate=0.0008):
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            continue
        n = int(len(out) * rate)
        if n == 0:
            continue
        # unparseable / out-of-range timestamps surface as nulls in the raw
        # layer (datetime columns must stay typed for parquet); the validator
        # counts them as invalid timestamps and cleaning repairs or drops them
        idx = rng.choice(len(out), size=n, replace=False)
        out.loc[idx, col] = pd.NaT
    return out


def _duplicate_rows(frame, rate, rng):
    n = max(int(len(frame) * rate), 1)
    idx = rng.choice(len(frame), size=n, replace=False)
    dups = frame.iloc[idx]
    return pd.concat([frame, dups], ignore_index=True)


def _inconsistent_ids(frame, col, rng, rate=0.002):
    """Replace a few valid ids with formatting variants the cleaners must fix."""
    out = frame.copy()
    vals = out[col].astype(str)
    n = max(int(len(vals) * rate), 1)
    idx = rng.choice(len(vals), size=n, replace=False)
    styled = [v.replace("-", "").lower() if "-" in v else v.lower() for v in vals.iloc[idx]]
    out.loc[idx, col] = styled
    return out


def _inconsistent_categories(frame, columns, rng, rate=0.002):
    """Misspell a few categorical values so cleaning normalises them."""
    out = frame.copy()
    typo_map = {
        "rain": "rainy", "clear": "clearr", "storm": "thunderstorm",
        "weekend": "wknd", "holiday": "holliday", "snow": "snowy",
    }
    for col in columns:
        if col not in out.columns:
            continue
        vals = out[col].astype(str)
        n = max(int(len(vals) * rate), 1)
        cand = [i for i, v in enumerate(vals) if v in typo_map]
        chosen = rng.choice(cand if cand else [0], size=min(n, len(cand) or 1), replace=False) if cand else []
        for i in chosen:
            out.loc[i, col] = typo_map[str(vals.iloc[i])]
    return out


def _wrong_route_links(tickets, trips, rng, rate=0.001):
    """A few tickets pointing at route ids that do not exist / are stale."""
    out = tickets.copy()
    n = max(int(len(out) * rate), 1)
    idx = rng.choice(len(out), size=n, replace=False)
    all_routes = sorted(trips["route_id"].unique())
    bad = [f"R-{random.randint(900, 999)}" for _ in range(n)]
    out.loc[idx, "route_id"] = bad
    return out


def manifest() -> pd.DataFrame:
    """Human-readable summary of the problems this module introduces."""
    rows = []
    for name, cols in FRAME_MISSING.items():
        rows.append({"dataset": name, "problem": "missing values", "columns": ", ".join(cols)})
    for name, cols in FRAME_INVALID.items():
        rows.append({"dataset": name, "problem": "impossible numeric values", "columns": ", ".join(cols)})
    for name, cols in FRAME_TIMESTAMPS.items():
        rows.append({"dataset": name, "problem": "invalid timestamps", "columns": ", ".join(cols)})
    for name, col in FRAME_IDS.items():
        rows.append({"dataset": name, "problem": "inconsistent ids", "columns": col})
    for name, rate in FRAME_DUPLICATES.items():
        rows.append({"dataset": name, "problem": f"duplicate rows ({rate:.3f})", "columns": "-"})
    rows.append({"dataset": "tickets", "problem": "dangling route ids", "columns": "route_id"})
    return pd.DataFrame(rows)