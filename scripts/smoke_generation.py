"""Small-scale generator smoke test (dev tool, not part of the pipeline)."""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.seed.generator import generate  # noqa: E402

if __name__ == "__main__":
    t0 = time.perf_counter()
    frames = generate(
        seed=2024,
        scale={"routes": 8, "stops": 60, "vehicles": 12, "passengers": 3000},
        months=2,
        to_disk=False,
    )
    for name, frame in frames.items():
        print(f"{name:15s} rows={len(frame):8d} cols={frame.shape[1]}")
    trips = frames["trips"]
    print("trips sample:", trips.head(3).to_string())
    print("delay stats:", trips["departure_delay_min"].describe().round(2).to_dict())
    print("boardings total:", int(frames["tickets"]["trip_id"].nunique()), "trips w/ tickets")
    print("tickets total:", len(frames["tickets"]))
    print("occupancy max pct:", trips["occupancy_max_pct"].max())
    print("headway missing:", trips["scheduled_headway_min"].isna().sum())
    st = frames["stop_trips"]
    print("stop rows sanity ok =", (st["onboard_after"] >= 0).all() and (st["arr_delay_min"].notna()).all())
    print("duplicate trip ids:", trips["trip_id"].duplicated().sum())
    print(f"elapsed {time.perf_counter() - t0:.1f}s")