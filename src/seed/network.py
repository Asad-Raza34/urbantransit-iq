"""Synthetic transit network - routes, stops and the route-stop geometry.

Spatial positions are synthetic but geographically plausible: stops are laid
out around a city centre so routes radiate outwards, which keeps distances
and travel times believable and gives the map page something real to show.
"""

import numpy as np
import pandas as pd

from src.log import get_logger

logger = get_logger(__name__)

CITY_CENTRE = (33.6844, 73.0479)


def build_stops(n_stops: int, seed: int) -> pd.DataFrame:
    """Generate ``n_stops`` stations scattered around the city centre."""
    rng = np.random.default_rng(seed)
    centre = np.array(CITY_CENTRE)
    radius = rng.uniform(1.0, 9.0, n_stops)
    angle = rng.uniform(0, 2 * np.pi, n_stops)
    lat = np.round(centre[0] + radius * np.cos(angle) * 0.0085, 5)
    lon = np.round(centre[1] + radius * np.sin(angle) * 0.0105, 5)

    zones = np.array(["residential", "commercial", "industrial", "mixed", "university"])
    zone = rng.choice(zones, size=n_stops, p=[0.3, 0.28, 0.1, 0.2, 0.12])
    is_terminal = (radius > 7.6) | (radius < 1.4)

    frame = pd.DataFrame(
        {
            "stop_id": [f"S-{i + 1:03d}" for i in range(n_stops)],
            "stop_name": [f"Stop {i + 1}" for i in range(n_stops)],
            "lat": lat,
            "lon": lon,
            "zone": zone,
            "is_terminal": is_terminal,
        }
    )
    logger.info("synthetic stops: %d", len(frame))
    return frame


def build_routes(n_routes: int, stops: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build routes plus the ordered route-stop geometry with real distances.

    Returns
    -------
    routes : DataFrame
        One row per route with category, terminals, length and stop count.
    route_stops : DataFrame
        One row per (route, stop) with sequence number, segment length from
        the previous stop and cumulative distance from the route origin.
    """
    rng = np.random.default_rng(seed)
    terminals = stops[stops["is_terminal"]].reset_index(drop=True)
    if len(terminals) < 4:
        terminals = stops.head(4)

    route_rows, link_rows = [], []
    for i in range(n_routes):
        route_id = f"R-{i + 1:03d}"
        category = "core" if i % 3 == 0 else ("feeder" if i % 3 == 1 else "express")
        a, b = rng.choice(len(terminals), size=2, replace=False)
        start, end = terminals.iloc[a], terminals.iloc[b]
        length_km = float(np.hypot(end.lat - start.lat, end.lon - start.lon) * 111.0)
        length_km = max(length_km, 5.0)
        n_stops = int(np.clip(round(length_km / 1.6) + 5, 6, 16))

        sequence = _corridor_stops(start, end, n_stops, stops)
        cum, prev = 0.0, None
        for seq, (stop_id, coords) in enumerate(sequence, start=1):
            if prev is None:
                seg = 0.0
            else:
                seg = float(np.hypot(coords[0] - prev[0], coords[1] - prev[1]) * 111.0)
            cum += seg
            link_rows.append(
                {
                    "route_id": route_id,
                    "stop_id": stop_id,
                    "seq": seq,
                    "segment_km": round(seg, 3),
                    "cumulative_km": round(cum, 3),
                }
            )
            prev = coords

        route_rows.append(
            {
                "route_id": route_id,
                "route_name": f"Route {i + 1}",
                "category": category,
                "from_stop": sequence[0][0],
                "to_stop": sequence[-1][0],
                "length_km": round(cum, 3),
                "stops_count": len(sequence),
            }
        )

    routes = pd.DataFrame(route_rows)
    route_stops = pd.DataFrame(link_rows)
    logger.info("synthetic routes: %d, route-stop links: %d", len(routes), len(route_stops))
    return routes, route_stops


def _corridor_stops(start: pd.Series, end: pd.Series, n_stops: int,
                    stops: pd.DataFrame) -> list[tuple[str, tuple[float, float]]]:
    """Pick ``n_stops`` real stops lying roughly along the start->end line."""
    t = np.linspace(0.0, 1.0, n_stops)
    corridor = pd.DataFrame(
        {
            "lat": start["lat"] + (end["lat"] - start["lat"]) * t,
            "lon": start["lon"] + (end["lon"] - start["lon"]) * t,
        }
    )
    chosen, used = [], set()
    for _, point in corridor.iterrows():
        dist = np.hypot(stops["lat"] - point["lat"], stops["lon"] - point["lon"])
        for sid in dist.sort_values().index:
            if sid not in used:
                used.add(sid)
                chosen.append((stops.at[sid, "stop_id"], (stops.at[sid, "lat"], stops.at[sid, "lon"])))
                break
    return chosen