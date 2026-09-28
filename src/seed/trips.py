"""Trip, stop-level and ticket generation for a single route.

The important property of this module is that every output dataset is
internally consistent:

* each trip has a scheduled and an actual departure;
* boarding demand reacts to time-of-day, day type, season, weather and
  special events;
* boardings are spread over the route's stops;
* for every boarding an alighting stop is drawn from a downstream gravity
  model, which fixes both the on-board occupancy trajectory and the OD pairs
  that become the ticket records - nothing is double counted.

The entry point is :func:`build_route`; it is deterministic for a given seed,
so the whole generated dataset is reproducible.
"""

import numpy as np
import pandas as pd

from config import settings
from src.log import get_logger

logger = get_logger(__name__)

SERVICE_START_MIN = 330          # 05:30
SERVICE_END_MIN = 1427           # 23:47
PEAK_WINDOWS = [(420, 560), (960, 1140)]   # 07:00-09:20 and 16:00-19:00

CATEGORY = {
    "core": {"weekday": 28, "weekend": 22, "holiday": 18, "avg_boardings": 55.0, "speed_kmh": 18.0, "latent_delay": 1.2},
    "feeder": {"weekday": 10, "weekend": 7, "holiday": 6, "avg_boardings": 18.0, "speed_kmh": 20.0, "latent_delay": 0.6},
    "express": {"weekday": 13, "weekend": 10, "holiday": 8, "avg_boardings": 24.0, "speed_kmh": 24.0, "latent_delay": 0.8},
}
ZONE_WEIGHT = {"residential": 0.34, "mixed": 0.26, "commercial": 0.14, "university": 0.19, "industrial": 0.07}
MONTH_FACTOR = {1: 1.08, 2: 1.06, 3: 1.0, 4: 0.98, 5: 0.97, 6: 0.94, 7: 0.92, 8: 0.93, 9: 0.98, 10: 1.0, 11: 1.04, 12: 1.08}
WEATHER_BOARDING = {"rain": 1.15, "snow": 1.05, "storm": 1.20, "clear": 1.0}
# Weather acts through three separate physical channels: an additive delay that
# every trip of the operating day absorbs, a congestion amplifier (a saturated
# network slows down far more in rain or snow), and a per-segment running-time
# penalty used for the arrival-delay trajectory.
WEATHER_DELAY = {"rain": 2.5, "snow": 16.0, "storm": 28.0, "clear": 0.0}
WEATHER_CONGESTION = {"rain": 1.6, "snow": 2.6, "storm": 3.4, "clear": 1.0}
WEATHER_RUN_SEG = {"rain": 0.35, "snow": 1.4, "storm": 2.2, "clear": 0.0}
WEATHER_TRAVEL = {"rain": 1.06, "snow": 1.18, "storm": 1.30, "clear": 1.0}
FARE_DISCOUNT = {"monthly": 0.35, "weekly": 0.50, "day": 0.75, "single": 1.0}

# ---------------------------------------------------------------------------
# Delay structure
# ---------------------------------------------------------------------------
# A departure delay is the sum of effects that operations either plan for, can
# observe on the day, or can learn from earlier departures - plus genuine
# residual randomness:
#
#   baseline     small timetable bias per route category
#   congestion   time-band load on the route, amplified by weather
#   chronic      structural per-route lateness (dwells, signals, demand)
#   weather      additive penalty of the operating day's weather
#   disruption   route-day regime (roadworks, blocked lane, clearance backlog)
#                that raises every later trip of that route-day
#   vehicle/systemic/dow   per-vehicle reliability and day-level tendencies
#   incidents    two independent processes (routine and rare serious)
#
# Nothing here is deterministic: the residual term stays non-zero and the two
# incident processes keep the outcome uncertain even when every persistent
# factor is known.
CONGESTION_BASE = 1.8              # minutes of peak congestion on the worst fare stage
ROUTE_CONGESTION_CV = 0.35         # per-route congestion multiplier spread
ROUTE_CHRONIC_SD = 1.2             # per-route structural lateness (minutes)
BAND_CONGESTION = {"morning peak": 1.0, "evening peak": 0.9, "midday": 0.55, "night": 0.30}
WEEKEND_CONGESTION = 0.78          # weekend load relative to a weekday
HOLIDAY_CONGESTION = 0.85
ROUTE_WEATHER_SENSITIVITY_SD = 0.20  # per-route weather multiplier variation
VEHICLE_RELIABILITY_SD = 1.0       # per-vehicle delay tendency (minutes)
DAILY_SYSTEMIC_SD = 1.0            # per-date network-wide effect (minutes)
DAY_OF_WEEK_EFFECT = {"mon": 0.8, "tue": 0.0, "wed": 0.0, "thu": 0.0, "fri": 0.5, "sat": -0.3, "sun": -0.5}
DISRUPTION_PROB = 0.15             # share of route-days with an active regime
DISRUPTION_LEVELS = (5.0, 12.0, 22.0, 35.0)  # regime magnitudes (minutes added to the route-day)
DISRUPTION_WEIGHTS = (0.35, 0.30, 0.20, 0.15)
RESIDUAL_SD = 0.7                  # unexplained per-trip variation (minutes)
MINOR_INCIDENT_PROB = 0.015        # routine incidents (signal failure, blocked bay) (reduced)
MINOR_INCIDENT_RANGE = (2.0, 10.0)
MAJOR_INCIDENT_PROB = 0.0008       # rare serious incidents (collision, road closure) (reduced)
MAJOR_INCIDENT_RANGE = (20.0, 45.0)
EVENT_DELAY_NOISE_SD = 0.30        # realised event impact varies around the plan


def build_route(route, route_stops, stops, vehicles, pools, passengers,
                calendar, events, seed: int, trip_start: int):
    """Produce ``(trips, stop_trips, tickets, next_trip_start)`` for one route."""
    spec = CATEGORY[route["category"]]
    rng = np.random.default_rng(seed + int(route["route_id"].split("-")[1]) * 7)

    geo, geo_rev = _geometry(route["route_id"], route_stops, stops)
    S = len(geo["stop_id"])

    pool_vids = pools[pools["route_id"] == route["route_id"]]["vehicle_id"].tolist() or \
        [vehicles["vehicle_id"].iloc[0]]
    caps = {vid: int(vehicles.set_index("vehicle_id").at[vid, "capacity"]) for vid in pool_vids}
    pool_caps = np.array([caps[vid] for vid in pool_vids])

    # -----------------------------------------------------------------
    # Persistent structure.  These are *not* written to the dataset: they are
    # the hidden state of the simulated city.  A model can only recover them
    # from earlier departures (route/vehicle history), which is exactly how a
    # real control room estimates them.
    # -----------------------------------------------------------------
    route_rng = np.random.default_rng(seed + int(route["route_id"].split("-")[1]) * 11)
    route_congestion_index = float(np.exp(route_rng.normal(0.0, ROUTE_CONGESTION_CV)))
    route_chronic_offset = float(route_rng.normal(0.0, ROUTE_CHRONIC_SD))
    route_weather_sensitivity = 1.0 + route_rng.normal(0.0, ROUTE_WEATHER_SENSITIVITY_SD)

    # Route-day disruption regime: drawn once per route and shared by every
    # trip of that route on that date (both directions), so the regime is
    # visible in the trips that already departed earlier the same day.
    day_levels = route_rng.choice(DISRUPTION_LEVELS, size=len(calendar), p=DISRUPTION_WEIGHTS)
    day_active = route_rng.random(len(calendar)) < DISRUPTION_PROB
    disruption_by_date = {str(d): float(lv if act else 0.0)
                          for d, lv, act in zip(calendar["date"], day_levels, day_active)}

    vehicle_reliability = {}
    for vid in pool_vids:
        v_rng = np.random.default_rng(seed + int(vid.split("-")[1]) * 13)
        vehicle_reliability[vid] = v_rng.normal(0, VEHICLE_RELIABILITY_SD)

    # Daily systemic effects: shared by the whole network on a given date.
    daily_systemic = {}
    for day in calendar.itertuples():
        d_rng = np.random.default_rng(seed + 997 + int(pd.Timestamp(day.date).dayofyear) * 17)
        daily_systemic[str(day.date)] = d_rng.normal(0, DAILY_SYSTEMIC_SD)

    # -----------------------------------------------------------------
    # Timetable for the whole year for this route.
    # -----------------------------------------------------------------
    minutes, dates = [], []
    for day in calendar.itertuples():
        n = spec["weekend"] if day.is_weekend else (spec["holiday"] if day.is_holiday else spec["weekday"])
        m = _sample_minutes(rng, n)
        minutes.extend(m)
        dates.extend([str(day.date)] * len(m))
    n_trips = len(minutes)
    dates = np.array(dates, dtype="U10")
    sched_min = np.round(np.array(minutes, dtype=float)).astype("int64")
    direction = (np.arange(n_trips) % 2).astype("i1")
    vehicle_ids = np.array([pool_vids[i % len(pool_vids)] for i in range(n_trips)], dtype=object)
    capacity = np.array([pool_caps[i % len(pool_vids)] for i in range(n_trips)], dtype=int)

    event_map = _event_map(events, geo["zone"])
    cal_index = {str(d): i for i, d in enumerate(calendar["date"])}

    blocks = {}
    for d in (0, 1):
        sel = np.flatnonzero(direction == d)
        blocks[d] = _direction_block(
            d, sel, dates, sched_min, calendar, cal_index, event_map,
            geo if d == 0 else geo_rev, capacity[sel], vehicle_ids[sel], spec, rng,
            route_congestion_index, route_chronic_offset, route_weather_sensitivity,
            vehicle_reliability, daily_systemic, disruption_by_date
        )

    data = _concatenate_blocks(blocks[0], blocks[1])
    order = np.argsort(data["sched_dep"])
    data = _reorder(data, order)

    trip_ids = np.array([f"T-{i + 1:08d}" for i in range(trip_start, trip_start + n_trips)])

    trips = _build_trips_frame(data, trip_ids, route, n_trips)
    stop_trips = _build_stop_trips_frame(data, trip_ids, route, geo, geo_rev)
    tickets = _build_tickets(data, trip_ids, route["route_id"], geo, geo_rev, stops, passengers, rng)

    return trips, stop_trips, tickets, trip_start + n_trips


# ---------------------------------------------------------------------------
# Geometry / context
# ---------------------------------------------------------------------------

def _geometry(route_id, route_stops, stops):
    rs = route_stops[route_stops["route_id"] == route_id].sort_values("seq")
    seg = rs["segment_km"].to_numpy()
    cum = rs["cumulative_km"].to_numpy()
    sids = rs["stop_id"].to_numpy()
    zone = stops.set_index("stop_id").reindex(sids)["zone"].to_numpy()
    weights = np.array([ZONE_WEIGHT.get(z, 0.2) for z in zone])
    weights = weights / weights.sum()

    fwd = {"stop_id": sids, "segment": seg, "cum_dist": cum, "zone": zone, "weights": weights}

    rev_seg = np.concatenate([[0.0], seg[1:][::-1]])
    bwd = {"stop_id": sids[::-1].copy(), "segment": rev_seg,
           "cum_dist": np.cumsum(rev_seg), "zone": zone[::-1].copy(),
           "weights": weights[::-1].copy()}
    return fwd, bwd


def _event_map(events, zones):
    """date -> (demand multiplier, extra delay) for events touching route zones."""
    out = {}
    if events is None or events.empty:
        return out
    for row in events.itertuples():
        if row.zone in list(zones):
            out[str(row.date)] = (float(row.demand_multiplier), float(row.extra_delay_min))
    return out


def _context_arrays(dates, sched_min, calendar, cal_index, event_map):
    idx = np.array([cal_index[d] for d in dates], dtype=int)
    rows = calendar.iloc[idx]
    day_type = np.where(rows["is_weekend"].to_numpy(), "weekend",
                        np.where(rows["is_holiday"].to_numpy(), "holiday", "weekday"))
    weather = rows["weather"].to_numpy()
    month = rows["month"].to_numpy()
    is_peak = _is_peak(sched_min)
    hour = (sched_min / 60).astype(int)

    ev_mult = np.ones(len(dates))
    ev_delay = np.zeros(len(dates))
    for i, d in enumerate(dates):
        hit = event_map.get(d)
        if hit:
            ev_mult[i], ev_delay[i] = hit
    return day_type, weather, month, is_peak, hour, ev_mult, ev_delay


# ---------------------------------------------------------------------------
# Direction block
# ---------------------------------------------------------------------------

def _direction_block(direction, sel, dates, sched_min, calendar, cal_index, event_map,
                     geo, capacity, vehicle_ids, spec, rng,
                     route_congestion_index, route_chronic_offset, route_weather_sensitivity,
                     vehicle_reliability, daily_systemic, disruption_by_date):
    """All per-trip matrices for one direction of this route."""
    n = len(sel)
    S = len(geo["stop_id"])
    if n == 0:
        return _empty_block(S)

    sm = sched_min[sel]
    ds = dates[sel]
    day_type, weather, month, peak, hour, ev_mult, ev_delay = _context_arrays(ds, sm, calendar, cal_index, event_map)

    base = pd.to_datetime(ds).to_numpy()                      # datetime64[ns]
    sched_dep = base + sm.astype("timedelta64[m]")            # datetime64[ns]

    # ------------------------------------------------------------------
    # Departure delay.  See the "Delay structure" block at the top of the
    # module for the reasoning behind each term.
    # ------------------------------------------------------------------
    band = _time_band(hour)
    band_eff = np.array([BAND_CONGESTION[b] for b in band])
    day_load = np.where(day_type == "weekend", WEEKEND_CONGESTION,
                        np.where(day_type == "holiday", HOLIDAY_CONGESTION, 1.0))
    congestion = (CONGESTION_BASE * route_congestion_index * band_eff
                  * np.vectorize(WEATHER_CONGESTION.get)(weather) * day_load)
    latent = np.full(n, spec["latent_delay"] + route_chronic_offset)
    weather_eff = np.vectorize(WEATHER_DELAY.get)(weather) * route_weather_sensitivity
    disruption = np.array([disruption_by_date.get(d, 0.0) for d in ds])
    vehicle_eff = np.array([vehicle_reliability.get(vid, 0.0) for vid in vehicle_ids])
    daily_eff = np.array([daily_systemic.get(d, 0.0) for d in ds])
    dow = pd.to_datetime(ds).dayofweek  # 0=Mon, 6=Sun
    dow_eff = np.array([DAY_OF_WEEK_EFFECT.get({0:"mon",1:"tue",2:"wed",3:"thu",4:"fri",5:"sat",6:"sun"}[d], 0.0) for d in dow])
    # a planned event's realised impact varies around its plan
    event_eff = ev_delay * (1.0 + rng.normal(0.0, EVENT_DELAY_NOISE_SD, n))
    minor_incident = (rng.random(n) < MINOR_INCIDENT_PROB) * rng.uniform(*MINOR_INCIDENT_RANGE, n)
    major_incident = (rng.random(n) < MAJOR_INCIDENT_PROB) * rng.uniform(*MAJOR_INCIDENT_RANGE, n)
    dep_delay = np.clip(latent + congestion + weather_eff + disruption + event_eff
                        + vehicle_eff + daily_eff + dow_eff
                        + minor_incident + major_incident
                        + rng.normal(0, RESIDUAL_SD, n), -2, 95)

    # travel-time geometry (minutes)
    travel_speed = np.vectorize(WEATHER_TRAVEL.get)(weather)
    seg_t = np.broadcast_to(geo["segment"][1:] / spec["speed_kmh"] * 60, (n, S - 1)).copy()
    seg_t = seg_t * (np.where(peak, 1.28, 1.0) * travel_speed)[:, None]
    seg_t = np.concatenate([np.zeros((n, 1)), seg_t], axis=1)
    dwell_t = np.maximum(0.25 + 0.02 * np.arange(S - 1), 0.0)
    dwell_cum = np.concatenate([[0.0], np.cumsum(dwell_t)])
    cum_min = np.cumsum(seg_t, axis=1) + dwell_cum[None, :]
    cum_dist = np.broadcast_to(geo["cum_dist"][None, :], (n, S)).copy()

    # boardings from demand drivers
    activity = _activity(hour)
    daily = np.where(day_type == "weekend", 0.62, np.where(day_type == "holiday", 0.72, 1.0))
    lam = spec["avg_boardings"] * activity * daily \
        * np.vectorize(MONTH_FACTOR.get)(month) \
        * np.vectorize(WEATHER_BOARDING.get)(weather) * ev_mult
    outward = np.clip(np.arange(1, S + 1) / S - 0.4, 0, 1)
    w = geo["weights"][None, :] * (1 + outward[None, :] * (hour == 8)[:, None] * 0.6)
    w = w / w.sum(axis=1, keepdims=True)
    lam_mat = np.clip(lam, 0.01, None)[:, None] * w
    B = rng.poisson(lam_mat).astype(int)

    # OD matrix via stick-breaking over the downstream gravity
    grav = _gravity(geo["cum_dist"])
    D = np.zeros((n, S, S), dtype=int)
    for j in range(S - 1):
        rem = B[:, j].copy()
        for k in range(j + 1, S):
            p = grav[j, k] / (grav[j, j:].sum() + 1e-9)
            dk = rng.binomial(rem, np.clip(p, 0, 1))
            D[:, j, k] = dk
            rem -= dk
        D[:, j, S - 1] += rem  # everyone alights by the end of the line
    A = D.sum(axis=1)

    # occupancy trajectory (integer count and its ratio against the vehicle capacity)
    onboard_mat = np.zeros((n, S), dtype=int)
    occ = np.zeros((n, S), dtype=float)
    onboard = np.zeros(n, dtype=int)
    for k in range(S):
        onboard = onboard - A[:, k] + B[:, k]
        onboard_mat[:, k] = onboard
        occ[:, k] = onboard / np.maximum(capacity, 1)

    # arrival delays: a vehicle may recover a little delay at each stop (shorter
    # dwells) but never more than the shortest segment takes, otherwise the
    # downstream arrival time could precede the upstream one
    run_seg = np.vectorize(WEATHER_RUN_SEG.get)(weather)
    arr_delay = np.zeros((n, S))
    arr_delay[:, 0] = dep_delay
    for k in range(1, S):
        dwell_extra = np.clip((B[:, k] - 8) * 0.12, 0, 3.5)
        arr_delay[:, k] = arr_delay[:, k - 1] + dwell_extra + run_seg \
            + rng.normal(0, 0.5, n)
        arr_delay[:, k] = np.minimum(arr_delay[:, k], 140)
        arr_delay[:, k] = np.maximum(arr_delay[:, k], arr_delay[:, k - 1] - 0.1)

    return {
        "sched_dep": sched_dep, "sched_min": sm, "date": ds,
        "direction": np.full(n, direction, dtype="i1"), "day_type": day_type, "weather": weather,
        "is_peak": peak, "hour": hour, "capacity": capacity, "vehicle_id": vehicle_ids,
        "dep_delay": dep_delay, "cum_min": cum_min, "cum_dist": cum_dist,
        "sched_travel": cum_min[:, -1].copy(), "B": B, "A": A, "D": D, "onboard": onboard_mat, "occ": occ,
        "arr_delay": arr_delay,
    }


def _empty_block(S):
    return {"sched_dep": np.array([], dtype="datetime64[ns]"), "sched_min": np.array([], dtype="int64"),
            "date": np.array([], dtype="U10"), "direction": np.array([], dtype="i1"),
            "day_type": np.array([], dtype=object), "weather": np.array([], dtype=object),
            "is_peak": np.array([], dtype=bool), "hour": np.array([], dtype="int64"),
            "capacity": np.array([], dtype="int64"), "vehicle_id": np.array([], dtype=object),
            "dep_delay": np.array([], dtype=float), "cum_min": np.zeros((0, S)),
            "cum_dist": np.zeros((0, S)), "sched_travel": np.array([], dtype=float),
            "B": np.zeros((0, S), dtype=int), "A": np.zeros((0, S), dtype=int),
            "D": np.zeros((0, S, S), dtype=int), "onboard": np.zeros((0, S), dtype=int),
            "occ": np.zeros((0, S)),
            "arr_delay": np.zeros((0, S)),
    }


def _concatenate_blocks(b0, b1):
    out = {}
    for key in b0:
        a, c = b0[key], b1[key]
        if a.ndim == 0:
            out[key] = a
        elif a.ndim == 3:
            out[key] = np.concatenate([a, c], axis=0)
        else:
            out[key] = np.concatenate([a, c])
    return out


def _reorder(data, order):
    return {key: (val[order] if val.ndim > 0 and val.shape[0] == len(order) else val)
            for key, val in data.items()}


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _sample_minutes(rng, n):
    if n <= 0:
        return np.array([], dtype=int)
    which = rng.choice(3, size=n, p=[0.34, 0.34, 0.32])
    out = np.empty(n, dtype=int)
    out[which == 0] = rng.integers(*PEAK_WINDOWS[0], size=int((which == 0).sum()))
    out[which == 1] = rng.integers(*PEAK_WINDOWS[1], size=int((which == 1).sum()))
    off = np.arange(SERVICE_START_MIN, SERVICE_END_MIN)
    off = off[~np.isin(off, np.concatenate([np.arange(a, b) for a, b in PEAK_WINDOWS]))]
    out[which == 2] = rng.choice(off, size=int((which == 2).sum()))
    return np.sort(out)


def _is_peak(minutes):
    out = np.zeros_like(minutes, dtype=bool)
    for lo, hi in PEAK_WINDOWS:
        out |= (minutes >= lo) & (minutes < hi)
    return out


def _time_band(hours):
    """Service band of each trip.

    Mirrors ``src.integrate.integrator._time_band`` so the generated demand
    band and the analysts' ``time_band`` column always agree.
    """
    return np.select([hours < 5, hours < 10, hours < 16, hours < 20],
                     ["night", "morning peak", "midday", "evening peak"],
                     default="night")


def _activity(hour):
    rise = np.exp(-((hour - 8.0) / 1.6) ** 2)
    evening = np.exp(-((hour - 17.5) / 1.8) ** 2)
    midday = np.exp(-((hour - 13.0) / 4.0) ** 2)
    return np.clip((rise + evening * 0.9 + midday * 0.35) / 1.9, 0.12, 1.0)


def _gravity(cum_dist):
    """Downstream gravity matrix: passengers ride out far enough for the bus
    to fill up along its trunk (a steeper decay would empty the bus too early
    and make every trip look under-occupied)."""
    S = len(cum_dist)
    grav = np.zeros((S, S))
    for j in range(S - 1):
        for k in range(j + 1, S):
            d = float(cum_dist[k] - cum_dist[j])
            grav[j, k] = max(np.exp(-0.05 * d) / (1 + 0.15 * d), 1e-3)
    return grav


# ---------------------------------------------------------------------------
# Output frame builders
# ---------------------------------------------------------------------------

def _build_trips_frame(data, trip_ids, route, n):
    sched_dep = pd.to_datetime(data["sched_dep"])
    dep_delay = data["dep_delay"]
    arr_delay = data["arr_delay"]

    frame = pd.DataFrame(
        {
            "trip_id": trip_ids,
            "route_id": route["route_id"],
            "vehicle_id": data["vehicle_id"].astype(str),
            "date": data["date"],
            "day_type": data["day_type"],
            "direction": data["direction"],
            "is_peak": data["is_peak"],
            "hour": data["hour"],
            "scheduled_departure": sched_dep,
            "actual_departure": pd.to_datetime(
                sched_dep.to_numpy() + (dep_delay * 60).astype("int64").astype("timedelta64[s]")
            ),
            "departure_delay_min": dep_delay.round(1),
            "weather": data["weather"],
            "distance_km": np.full(n, float(route["length_km"])).round(3),
            "scheduled_travel_min": data["sched_travel"].round(1),
            "actual_travel_min": (data["sched_travel"] + arr_delay[:, -1] - dep_delay).round(1),
            "boardings": data["B"].sum(axis=1),
            "alightings": data["A"].sum(axis=1),
            "capacity": data["capacity"],
            "occupancy_avg_pct": (data["occ"].mean(axis=1) * 100).round(1),
            "occupancy_max_pct": (data["occ"].max(axis=1) * 100).round(1),
            "on_time": dep_delay <= settings.DELAY_ON_TIME_MAX,
        }
    )
    # NOTE: the per-route / per-vehicle / per-day effects that drive the delay
    # above (route chronic offset, congestion index, weather sensitivity,
    # route-day disruption regime, vehicle reliability, daily systemic factor,
    # day-of-week effect) are the *hidden state* of the simulated city.  They
    # are additive/near-additive components of ``departure_delay_min``, so
    # persisting them would hand a model the answer (target leakage).  A real
    # control room can only recover them from earlier departures of the same
    # route / vehicle, which is exactly what the delay classifier's history
    # features do.  Do not add them back to the output frame.
    frame["scheduled_headway_min"] = _headway(frame, "scheduled_departure")
    frame["actual_headway_min"] = _headway(frame, "actual_departure")
    return frame


def _build_stop_trips_frame(data, trip_ids, route, geo, geo_rev):
    n, S = data["B"].shape
    fwd_ids = np.array(geo["stop_id"])
    bwd_ids = np.array(geo_rev["stop_id"])
    dir_all = data["direction"]

    stop_id_seq = np.empty((n, S), dtype="U10")
    stop_id_seq[dir_all == 0] = fwd_ids[None, :]
    stop_id_seq[dir_all == 1] = bwd_ids[None, :]

    base = pd.to_datetime(data["sched_dep"]).to_numpy()[:, None]     # (n,1)
    sched_sec = (data["cum_min"] * 60).astype("int64").astype("timedelta64[s]")
    actual_sec = ((data["cum_min"] + data["arr_delay"]) * 60).astype("int64").astype("timedelta64[s]")
    sched_arr = base + sched_sec
    actual_arr = base + actual_sec

    aboard = data["B"].ravel(); alit = data["A"].ravel()
    return pd.DataFrame(
        {
            "trip_id": np.repeat(trip_ids, S),
            "route_id": route["route_id"],
            "stop_id": stop_id_seq.ravel(),
            "seq": np.tile(np.arange(1, S + 1), n),
            "boarded": aboard,
            "alighted": alit,
            "onboard_after": data["onboard"].ravel(),
            "arr_delay_min": data["arr_delay"].ravel().round(1),
            "dwell_sec": np.round(25 + data["onboard"].ravel() * 2.5, 1),
            "scheduled_arrival": pd.to_datetime(sched_arr.ravel()),
            "actual_arrival": pd.to_datetime(actual_arr.ravel()),
            "occupancy_after": data["occ"].ravel().round(3),
        }
    )


def _headway(frame, col):
    """Minutes to the previous departure within the same (date, direction)."""
    return (frame.groupby(["date", "direction"])[col]
            .diff().dt.total_seconds() / 60.0).round(1)


def _build_tickets(data, trip_ids, route_id, geo, geo_rev, stops, passengers, rng):
    """Build ticket rows from the OD count tensor ``D``."""
    n, S = data["B"].shape
    cols = ["ticket_id", "trip_id", "route_id", "boarding_stop_id", "alighting_stop_id",
            "passenger_id", "pass_type", "boarding_datetime", "alighting_datetime",
            "distance_km", "travel_min", "fare"]
    if n == 0:
        return pd.DataFrame(columns=cols)

    t_idx, j_idx, k_idx = [], [], []
    for j in range(S - 1):
        for k in range(j + 1, S):
            counts = data["D"][:, j, k]
            tot = int(counts.sum())
            if tot == 0:
                continue
            t_idx.append(np.repeat(np.arange(n), counts))
            j_idx.append(np.full(tot, j, dtype=int))
            k_idx.append(np.full(tot, k, dtype=int))
    t = np.concatenate(t_idx)
    j = np.concatenate(j_idx)
    k = np.concatenate(k_idx)

    fwd_ids = np.array(geo["stop_id"])
    bwd_ids = np.array(geo_rev["stop_id"])
    stop_id_seq = np.empty((n, S), dtype="U10")
    stop_id_seq[data["direction"] == 0] = fwd_ids[None, :]
    stop_id_seq[data["direction"] == 1] = bwd_ids[None, :]

    cum_diff = data["cum_dist"][t, k] - data["cum_dist"][t, j]
    travel_min = (data["cum_min"][t, k] + data["arr_delay"][t, k]
                  - data["cum_min"][t, j] - data["arr_delay"][t, j])

    base = pd.to_datetime(data["sched_dep"]).to_numpy()[t]
    boarding_dt = base + pd.to_timedelta(data["cum_min"][t, j] + data["arr_delay"][t, j], unit="m")
    alighting_dt = base + pd.to_timedelta(data["cum_min"][t, k] + data["arr_delay"][t, k], unit="m")

    payer, pass_type = _sample_passengers(passengers, stops, stop_id_seq[t, j], rng)
    discount = np.array([FARE_DISCOUNT[p] for p in pass_type])

    # ticket ids embed the (globally unique) trip id so they cannot collide
    # across routes or regeneration runs
    ordinal = pd.Series(trip_ids[t]).groupby(trip_ids[t]).cumcount().add(1).to_numpy()
    ticket_id = np.array([f"{tid}_{n:04d}" for tid, n in zip(trip_ids[t], ordinal)])

    return pd.DataFrame(
        {
            "ticket_id": ticket_id,
            "trip_id": np.take(trip_ids, t),
            "route_id": route_id,
            "boarding_stop_id": stop_id_seq[t, j],
            "alighting_stop_id": stop_id_seq[t, k],
            "passenger_id": payer,
            "pass_type": pass_type,
            "boarding_datetime": pd.to_datetime(boarding_dt),
            "alighting_datetime": pd.to_datetime(alighting_dt),
            "distance_km": np.round(cum_diff, 2),
            "travel_min": np.round(np.maximum(travel_min, 0.5), 1),
            "fare": np.round((18 + 4.2 * cum_diff) * discount, 1),
        }
    )


def _sample_passengers(passengers, stops, boarding_stop, rng):
    """Pick a passenger per ticket, favouring the boarding stop's zone."""
    pidx = passengers.set_index("passenger_id")
    home = pidx["home_zone"].to_numpy()
    pids = pidx.index.to_numpy()
    zone_of_stop = stops.set_index("stop_id")["zone"].reindex(boarding_stop).to_numpy()

    zones = list(pidx["home_zone"].unique())
    zone_pools = {z: pids[home == z] for z in zones}
    zone_code = {z: i for i, z in enumerate(zones)}
    pool_sizes = np.array([len(zone_pools[z]) for z in zones])
    zc = np.array([zone_code[z] for z in zone_of_stop])

    pick = rng.integers(0, np.maximum(pool_sizes[zc], 1))
    assigned = np.empty(len(zc), dtype=object)
    for z, c in zone_code.items():
        mask = zc == c
        assigned[mask] = zone_pools[z][pick[mask]]

    cross = rng.random(len(zc)) < 0.12
    assigned[cross] = pids[rng.integers(0, len(pids), size=int(cross.sum()))]

    ordered = assigned.astype(str)
    pass_type = pidx["pass_type"].reindex(ordered).to_numpy()
    return ordered, pass_type