"""Demand analytics: daily series, peak detection, anomalies, event impact.

* ``daily_demand``   - passengers per route per day (the forecasting input);
* ``peak_patterns``  - per-route time-band demand share + detected peak bands;
* ``anomalies``      - z-score deviations of daily demand from each route's
                       own baseline (settings.ANOMALY_ZSCORE);
* ``event_impact``   - measured effect of special events on city-wide demand
                       and average delay vs a 7-day baseline;
* ``od_zone_flow``   - zone-to-zone passenger and revenue summary.
"""

import numpy as np
import pandas as pd

from config import settings
from src.analytics.core import read_processed, read_clean, save
from src.log import get_logger

logger = get_logger(__name__)

TIME_BANDS = ["night", "morning peak", "midday", "evening peak"]


def daily_demand() -> pd.DataFrame:
    flow = read_processed("passenger_flow")
    g = flow.groupby(["route_id", "date"], as_index=False).agg(
        passengers=("ticket_id", "count"),
        revenue=("fare", "sum"),
        avg_travel_min=("travel_min", "mean"),
    )
    g["date"] = pd.to_datetime(g["date"])
    g["is_weekend"] = g["date"].dt.weekday >= 5
    g["month"] = g["date"].dt.month
    g = g.sort_values(["route_id", "date"]).reset_index(drop=True)
    return g[["route_id", "date", "passengers", "revenue",
              "avg_travel_min", "is_weekend", "month"]]


def peak_patterns() -> pd.DataFrame:
    flow = read_processed("passenger_flow")
    tb = flow.groupby(["route_id", "time_band"], as_index=False)["ticket_id"] \
        .count().rename(columns={"ticket_id": "passengers"})
    tb = tb[tb["time_band"].isin(TIME_BANDS)]
    share = tb.groupby("route_id")["passengers"].transform("sum")
    tb["band_share"] = (tb["passengers"] / share.clip(1)).round(4)

    def top_bands(g: pd.DataFrame) -> pd.Series:
        top = g.sort_values("band_share", ascending=False).head(2)["time_band"].tolist()
        return pd.Series({"primary_peak": top[0], "secondary_peak": top[1] if len(top) > 1 else ""})

    routes = tb.groupby("route_id").apply(top_bands).reset_index()
    tb = tb.merge(routes, on="route_id", how="left")

    hourly = flow.copy()
    hourly["hour"] = pd.to_datetime(hourly["boarding_datetime"]).dt.hour
    hour_dist = hourly.groupby("hour")["ticket_id"].count().reset_index() \
        .rename(columns={"ticket_id": "passengers"})
    am = hour_dist.sort_values("passengers", ascending=False).head(3)["hour"].tolist()
    summary = pd.DataFrame({
        "key": ["peak_hours_city_wide", "time_bands"],
        "value": [f"{sorted(am)}", "4"],
    })
    return tb, summary


def anomalies() -> pd.DataFrame:
    dd = daily_demand()
    mean = dd.groupby("route_id")["passengers"].transform("mean")
    std = dd.groupby("route_id")["passengers"].transform("std").replace(0, np.nan)
    z = (dd["passengers"] - mean) / std
    out = dd[z.abs() > settings.ANOMALY_ZSCORE].copy()
    out["z_score"] = z[z.abs() > settings.ANOMALY_ZSCORE].round(2).values
    out["anomaly_type"] = np.where(
        z[z.abs() > settings.ANOMALY_ZSCORE] > 0, "high demand", "low demand")
    out["is_weekend"] = out["date"].dt.weekday >= 5
    return out.sort_values(["route_id", "date"]).reset_index(drop=True)


def event_impact() -> pd.DataFrame:
    events = read_clean("events")
    dd = daily_demand()
    daily = dd.groupby("date")["passengers"].sum().sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max())).fillna(0)
    delay = read_processed("trip_facts")[["date", "departure_delay_min"]]
    delay_daily = delay.groupby("date")["departure_delay_min"].mean() \
        .reindex(daily.index).fillna(delay["departure_delay_min"].mean())

    rows = []
    for ev in events.itertuples():
        d = pd.Timestamp(ev.date)
        if d not in daily.index:
            continue
        base_idx = daily.index[(daily.index >= d - pd.Timedelta(days=7)) & (daily.index < d)]
        if len(base_idx) == 0:
            continue
        base_demand = float(daily.loc[base_idx].mean())
        base_delay = float(delay_daily.loc[base_idx].mean())
        demand_delta = 0.0 if base_demand <= 0 else (daily.loc[d] / base_demand - 1.0)
        rows.append({
            "event_id": ev.event_id,
            "date": d.strftime("%Y-%m-%d"),
            "type": ev.type,
            "zone": ev.zone,
            "expected_multiplier": float(ev.demand_multiplier),
            "actual_demand": int(daily.loc[d]),
            "baseline_demand": round(base_demand, 1),
            "demand_delta_pct": round(demand_delta * 100.0, 1),
            "actual_avg_delay": round(float(delay_daily.loc[d]), 2),
            "baseline_avg_delay": round(base_delay, 2),
            "delay_delta_pct": round(
                (float(delay_daily.loc[d]) / base_delay - 1.0) * 100.0, 1) if base_delay else 0.0,
        })
    return pd.DataFrame(rows)


def od_zone_flow() -> pd.DataFrame:
    flow = read_processed("passenger_flow")
    od = flow.groupby(["boarding_zone", "alighting_zone"], as_index=False).agg(
        passengers=("ticket_id", "count"),
        avg_fare=("fare", "mean"),
        revenue=("fare", "sum"),
    )
    return od.sort_values("passengers", ascending=False).reset_index(drop=True)


def build() -> dict[str, int]:
    dd = save("daily_demand", daily_demand())
    tb, summary = peak_patterns()
    save("peak_patterns", tb)
    save("peak_summary", summary, index=False)
    save("anomalies", anomalies())
    save("event_impact", event_impact())
    save("od_zone_flow", od_zone_flow())
    return {
        "daily_demand": len(dd),
        "peak_patterns": len(tb),
        "event_impact": len(pd.read_parquet(_path("event_impact"))),
    }


def _path(name: str):
    from pathlib import Path
    return Path("data/analytics") / f"{name}.parquet"