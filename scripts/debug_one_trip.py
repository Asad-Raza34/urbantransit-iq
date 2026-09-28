import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
from src.storage import read_dataset

gen = read_dataset(Path(PROJECT_ROOT) / "data" / "generated" / "stop_trips.parquet")
for tid in ["T-00009422", "T-00108761"]:
    sub = gen[gen["trip_id"] == tid].sort_values("seq")
    print("==", tid)
    print(sub[["seq", "stop_id", "arr_delay_min", "scheduled_arrival", "actual_arrival"]].to_string(index=False))

trips = read_dataset(Path(PROJECT_ROOT) / "data" / "generated" / "trips.parquet")
print(trips[trips["trip_id"].isin(["T-00009422", "T-00108761"])][["trip_id","route_id","direction","departure_delay_min","scheduled_departure","actual_departure"]].to_string(index=False))