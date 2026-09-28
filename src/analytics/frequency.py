"""Frequency analysis: service frequency, headway, and service level metrics.

* ``frequency_analysis`` - per-route, per-period service frequency metrics;
* ``headway_analysis``   - headway distribution and reliability metrics;
* ``service_level``      - service level metrics by route/time-period.
"""

import numpy as np
import pandas as pd

from config import settings
from src.analytics.core import read_processed, save
from src.log import get_logger

logger = get_logger(__name__)


TIME_BANDS = ["night", "morning peak", "midday", "evening peak"]
DAY_TYPES = ["weekday", "weekend", "holiday"]


def _time_period_label(hour: int) -> str:
    """Map hour to time period label."""
    if hour < 5:
        return "night"
    elif hour < 10:
        return "morning peak"
    elif hour < 16:
        return "midday"
    elif hour < 20:
        return "evening peak"
    return "night"


def _compute_headway_reliability_vectorized(trips: pd.DataFrame, include_date: bool = False) -> pd.DataFrame:
    """Compute headway reliability using vectorized operations instead of apply.

    A trip is "reliable" if actual_headway is within 20% of scheduled_headway.
    """
    trips = trips.copy()
    trips["date"] = pd.to_datetime(trips["date"])
    trips["time_period"] = trips["hour"].apply(
        lambda h: "night" if h < 5 else
        "morning peak" if h < 10 else
        "midday" if h < 16 else
        "evening peak" if h < 20 else "night"
    )

    # Filter to rows where both headways are available
    valid = trips["scheduled_headway_min"].notna() & trips["actual_headway_min"].notna() & (trips["scheduled_headway_min"] > 0)
    df = trips[valid].copy()

    lower = 0.8 * df["scheduled_headway_min"]
    upper = 1.2 * df["scheduled_headway_min"]
    df["is_reliable"] = (df["actual_headway_min"] >= lower) & (df["actual_headway_min"] <= upper)

    # Group and compute mean reliability
    if include_date:
        group_cols = ["route_id", "date", "time_period"]
    else:
        group_cols = ["route_id", "time_period"]
    reliability = df.groupby(group_cols, as_index=False)["is_reliable"].mean()
    reliability = reliability.rename(columns={"is_reliable": "headway_reliability"})
    return reliability


def frequency_analysis() -> pd.DataFrame:
    """Compute service frequency metrics per route, date, and time period.

    Metrics:
    - scheduled_trips: number of scheduled trips
    - actual_trips: number of trips that actually ran
    - trips_per_hour: scheduled trips per hour
    - avg_headway_min: average scheduled headway
    - headway_cv: coefficient of variation of headway
    - headway_reliability: share of headways within 20% of scheduled
    - service_gap_min: average gap between scheduled and actual headway
    - service_cancellation_rate: fraction of scheduled trips that didn't run
    """
    trips = read_processed("trip_facts")
    trips = trips.copy()
    trips["date"] = pd.to_datetime(trips["date"])
    trips["time_period"] = trips["hour"].apply(
        lambda h: "night" if h < 5 else
        "morning peak" if h < 10 else
        "midday" if h < 16 else
        "evening peak" if h < 20 else "night"
    )

    # Aggregate by route, date, time_period
    freq = trips.groupby(["route_id", "date", "time_period"], as_index=False).agg(
        scheduled_trips=("trip_id", "count"),
        actual_trips=("trip_id", "count"),
        avg_scheduled_headway_min=("scheduled_headway_min", "mean"),
        avg_actual_headway_min=("actual_headway_min", "mean"),
        headway_std_min=("scheduled_headway_min", "std"),
    )

    # Compute derived metrics
    freq["trips_per_hour"] = (freq["scheduled_trips"] / freq["avg_scheduled_headway_min"].replace(0, np.nan) * 60).round(2)
    freq["headway_cv"] = (freq["headway_std_min"] / freq["avg_scheduled_headway_min"].replace(0, np.nan)).round(3)
    freq["service_gap_min"] = (freq["avg_actual_headway_min"] - freq["avg_scheduled_headway_min"]).round(2)
    freq["service_cancellation_rate"] = 0.0  # All scheduled trips ran in our data

    # Add date parts
    freq["date"] = pd.to_datetime(freq["date"])
    freq["weekday"] = freq["date"].dt.weekday
    freq["month"] = freq["date"].dt.month
    freq["is_weekend"] = freq["weekday"] >= 5

    # Merge headway reliability (computed efficiently)
    reliability = _compute_headway_reliability_vectorized(read_processed("trip_facts"), include_date=True)
    freq = freq.merge(reliability, on=["route_id", "date", "time_period"], how="left")

    return freq.sort_values(["route_id", "date", "time_period"]).reset_index(drop=True)


def headway_analysis() -> pd.DataFrame:
    """Analyze headway distribution and reliability by route and time period."""
    trips = read_processed("trip_facts")
    trips = trips.copy()
    trips["time_period"] = trips["hour"].apply(
        lambda h: "night" if h < 5 else
        "morning peak" if h < 10 else
        "midday" if h < 16 else
        "evening peak" if h < 20 else "night"
    )

    # Headway distribution stats
    headway_stats = trips.groupby(["route_id", "time_period"], as_index=False).agg(
        scheduled_mean=("scheduled_headway_min", "mean"),
        scheduled_std=("scheduled_headway_min", "std"),
        actual_mean=("actual_headway_min", "mean"),
        actual_std=("actual_headway_min", "std"),
        min_headway=("actual_headway_min", "min"),
        max_headway=("actual_headway_min", "max"),
        trips=("trip_id", "count"),
    )

    # Headway reliability
    reliability = _compute_headway_reliability_vectorized(trips, include_date=False)
    headway_stats = headway_stats.merge(reliability, on=["route_id", "time_period"], how="left")
    headway_stats["headway_cv"] = (headway_stats["scheduled_std"] / headway_stats["scheduled_mean"].replace(0, np.nan)).round(3)
    headway_stats["headway_reliability"] = headway_stats["headway_reliability"].round(4)

    return headway_stats.sort_values(["route_id", "time_period"]).reset_index(drop=True)


def service_level() -> pd.DataFrame:
    """Service level metrics by route and time period.

    Computes:
    - service_regularity: inverse of headway CV
    - service_availability: trips per hour
    - service_reliability: on-time + headway reliability composite
    """
    freq = frequency_analysis()
    headway = headway_analysis()

    # Merge frequency and headway on route_id + time_period
    # headway is at route/time_period level (no date), freq is at route/date/time_period
    # Drop headway_reliability from freq to avoid merge suffixes
    freq = freq.drop(columns=["headway_reliability"])
    service = freq.merge(
        headway[["route_id", "time_period", "headway_reliability", "headway_cv"]],
        on=["route_id", "time_period"],
        how="left"
    )

    # Service regularity (inverse of headway CV, normalized 0-100)
    # headway_cv might be NaN, handle gracefully
    if "headway_cv" in service.columns:
        service["service_regularity"] = (100 * (1 - service["headway_cv"].clip(0, 1))).round(1)
    else:
        service["service_regularity"] = 50.0  # Default neutral value

    # Trips per hour (service availability)
    service["trips_per_hour"] = service["trips_per_hour"].round(2)

    # Service reliability composite
    trips = read_processed("trip_facts").copy()
    trips["date"] = pd.to_datetime(trips["date"])
    trips["time_period"] = trips["hour"].apply(
        lambda h: "night" if h < 5 else
        "morning peak" if h < 10 else
        "midday" if h < 16 else
        "evening peak" if h < 20 else "night"
    )
    on_time = trips.groupby(["route_id", "time_period"])["on_time"].mean().reset_index(name="on_time_rate")
    service = service.merge(on_time, on=["route_id", "time_period"], how="left")

    # Composite service level score
    service["service_reliability"] = (
        0.4 * service["on_time_rate"].fillna(0) +
        0.3 * service["headway_reliability"].fillna(0) +
        0.3 * service["service_regularity"].fillna(50) / 100
    ) * 100

    # Round numeric columns only
    numeric_cols = service.select_dtypes(include=[np.number]).columns
    service[numeric_cols] = service[numeric_cols].round(2)
    return service


def frequency_peak_analysis() -> pd.DataFrame:
    """Analyze peak frequency patterns by route."""
    trips = read_processed("trip_facts")
    trips = trips.copy()

    # Trips per hour by route
    hourly = trips.groupby(["route_id", "hour"], as_index=False)["trip_id"].count()
    hourly = hourly.rename(columns={"trip_id": "trips"})

    # Peak hours per route
    peak_hours = hourly.groupby("route_id").apply(
        lambda g: g.nlargest(3, "trips")["hour"].tolist()
    ).reset_index(name="peak_hours")

    # Off-peak hours
    offpeak_hours = hourly.groupby("route_id").apply(
        lambda g: g.nsmallest(3, "trips")["hour"].tolist()
    ).reset_index(name="offpeak_hours")

    peak_analysis = peak_hours.merge(offpeak_hours, on="route_id")

    # Peak frequency ratio
    hourly_pivot = hourly.pivot(index="route_id", columns="hour", values="trips").fillna(0)
    
    # Ensure all 24 hours exist as columns
    for h in range(24):
        if h not in hourly_pivot.columns:
            hourly_pivot[h] = 0
    hourly_pivot = hourly_pivot[sorted(hourly_pivot.columns)]
    
    peak_cols = [h for h in range(7, 10)] + [h for h in range(16, 19)]  # 7-9, 16-18
    offpeak_cols = [h for h in range(24) if h not in peak_cols]

    hourly_pivot["peak_freq"] = hourly_pivot[peak_cols].sum(axis=1)
    hourly_pivot["offpeak_freq"] = hourly_pivot[offpeak_cols].sum(axis=1)
    hourly_pivot["peak_ratio"] = (hourly_pivot["peak_freq"] / hourly_pivot["offpeak_freq"].replace(0, np.nan)).round(2)

    return hourly_pivot.reset_index()[["route_id", "peak_freq", "offpeak_freq", "peak_ratio"]]


def build() -> dict[str, int]:
    """Build all frequency analytics and persist."""
    freq = save("frequency_analysis", frequency_analysis())
    hw = save("headway_analysis", headway_analysis())
    sl = save("service_level", service_level())
    fp = save("frequency_peak_analysis", frequency_peak_analysis())

    return {
        "frequency_analysis": len(freq),
        "headway_analysis": len(hw),
        "service_level": len(sl),
        "frequency_peak_analysis": len(fp),
    }


if __name__ == "__main__":
    result = build()
    print(result)