import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.storage import read_dataset
import numpy as np

raw = read_dataset(Path(PROJECT_ROOT) / "data" / "raw" / "stop_trips.parquet")
print("len", len(raw), "cols", list(raw.columns))
print("null actual_arrival", raw["actual_arrival"].isna().sum())
print("null arr_delay", raw["arr_delay_min"].isna().sum())

gen = read_dataset(Path(PROJECT_ROOT) / "data" / "generated" / "stop_trips.parquet")
sub = gen.sort_values(["trip_id", "seq"]).copy()
a = pd.to_datetime(sub["actual_arrival"], errors="coerce")
prev = a.groupby(sub["trip_id"]).shift(1)
d0 = (a - prev).dt.total_seconds() / 60.0
bad0 = sub[d0 < 0].head(5)[["trip_id", "seq", "actual_arrival", "arr_delay_min"]]
print("generated negative diffs (no tolerance):", int((d0 < 0).sum()))
print(bad0.to_string())

# within 30s
bad = sub[(d0 < -0.5) & prev.notna()]
print("generated backwards >30s:", len(bad), bad["trip_id"].nunique(), "trips")
print(bad.head(10).to_string())

# raw layer
r = raw.sort_values(["trip_id", "seq"]).copy()
ra = pd.to_datetime(r["actual_arrival"], errors="coerce")
rprev = ra.groupby(r["trip_id"]).shift(1)
rd = (ra - rprev).dt.total_seconds() / 60.0
badr = r[(rd < -0.5) & rprev.notna()].head(8)[["trip_id", "seq", "actual_arrival", "arr_delay_min"]]
print("RAW backwards >30s:", ((rd < -0.5) & rprev.notna()).sum())
print(badr.to_string())