import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

root = Path("data/generated")
t = pd.read_parquet(root / "trips.parquet")
cat = pd.read_parquet(root / "routes.parquet")[["route_id", "category"]]
tm = t.merge(cat, on="route_id")

print("trips per category:")
print(tm.groupby("category")["trip_id"].count())

print("\nboardings / occupancy by category (all trips):")
print(tm.groupby("category").agg(
    trips=("trip_id", "count"),
    boardings_mean=("boardings", "mean"),
    occ_max_mean=("occupancy_max_pct", "mean"),
    occ_max_q90=("occupancy_max_pct", lambda s: s.quantile(0.90)),
    occ_max_q99=("occupancy_max_pct", lambda s: s.quantile(0.99)),
    occ_avg_mean=("occupancy_avg_pct", "mean"),
    cap_mean=("capacity", "mean"),
).round(2))

peak = tm[(tm["is_peak"]) & (tm["day_type"] == "weekday")]
print("\nweekday PEAK by category:")
print(peak.groupby("category").agg(
    n=("trip_id", "count"),
    boardings=("boardings", "mean"),
    occ_max=("occupancy_max_pct", "mean"),
    occ_max_q99=("occupancy_max_pct", lambda s: s.quantile(0.99)),
).round(1))

off = tm[(~tm["is_peak"]) & (tm["day_type"] == "weekday")]
print("\nweekday OFFPEAK boardings mean:")
print(off.groupby("category")["boardings"].mean().round(1))

tickets = pd.read_parquet(root / "tickets.parquet")
tt = tickets.merge(tm[["trip_id", "category"]], on="trip_id")
print("\ntickets per category:")
print(tt.groupby("category").size())

print("\ntrips per day by category (weekday):")
wk = tm[tm["day_type"] == "weekday"].groupby(["category", "date"]).size()
print(wk.groupby("category").mean().round(1))