"""Fleet generation - vehicles with types, capacities and per-route pools."""

import numpy as np
import pandas as pd

from src.log import get_logger

logger = get_logger(__name__)

TYPE_SPECS = {
    "standard": {"capacity": 26, "share": 0.55},
    "articulated": {"capacity": 40, "share": 0.20},
    "mini": {"capacity": 17, "share": 0.25},
}


def build_vehicles(n_vehicles: int, routes: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create the fleet and a per-route vehicle pool (2-4 vehicles per route)."""
    rng = np.random.default_rng(seed)
    types = rng.choice(
        list(TYPE_SPECS), size=n_vehicles, p=[s["share"] for s in TYPE_SPECS.values()]
    )
    capacities = np.array([TYPE_SPECS[t]["capacity"] for t in types])
    age = rng.integers(1, 14, size=n_vehicles).astype(int)

    vehicles = pd.DataFrame(
        {
            "vehicle_id": [f"V-{i + 1:03d}" for i in range(n_vehicles)],
            "vehicle_type": types,
            "capacity": capacities,
            "age_years": age,
            "home_depot": rng.choice(["North", "South", "East", "West"], size=n_vehicles),
        }
    )

    pool = []
    route_ids = routes["route_id"].tolist()
    for idx, route_id in enumerate(route_ids):
        n_pool = int(rng.integers(2, 5))
        start = (idx * n_pool) % n_vehicles
        for offset in range(n_pool):
            vid = vehicles.iloc[(start + offset) % n_vehicles]["vehicle_id"]
            pool.append({"route_id": route_id, "vehicle_id": vid})
    pools = pd.DataFrame(pool).drop_duplicates()

    logger.info("synthetic fleet: %d vehicles, %d route-pool links", len(vehicles), len(pools))
    return vehicles, pools