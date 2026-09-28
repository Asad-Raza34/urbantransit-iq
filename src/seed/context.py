"""Temporal and situational context - calendar, weather, special events.

These contextual tables influence demand and delays elsewhere, so the raw
data contains realistic seasonality and event-driven variations rather than
flat random numbers.
"""

import numpy as np
import pandas as pd

from src.log import get_logger

logger = get_logger(__name__)

SEASONS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}
HOLIDAYS = {
    (1, 1): "New Year",
    (3, 23): "Pakistan Day",
    (5, 1): "Labour Day",
    (8, 14): "Independence Day",
    (12, 25): "Quaid-e-Azam Day",
}


def build_calendar(start: str = "2024-01-01", days: int = 366, seed: int = 11) -> pd.DataFrame:
    """Daily calendar with weather conditions and holiday markers."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=days, freq="D")
    month = dates.month

    weekday = dates.dayofweek < 5
    holiday = np.array([(d.month, d.day) in HOLIDAYS for d in dates])

    season = np.select(
        [np.isin(month, SEASONS["winter"]),
         np.isin(month, SEASONS["spring"]),
         np.isin(month, SEASONS["summer"])],
        ["winter", "spring", "summer"],
        default="autumn",
    )

    # Rain probability per season; rainy days are slower and busier.
    rain_p = np.select(
        [season == "winter", season == "spring", season == "summer"],
        [0.30, 0.22, 0.12],
        default=0.18,
    )
    is_rain = rng.random(days) < rain_p
    is_snow = (season == "winter") & (rng.random(days) < 0.05)
    is_storm = (rng.random(days) < 0.015) & (~is_snow)
    weather = np.select(
        [is_storm, is_snow, is_rain], ["storm", "snow", "rain"], default="clear"
    )

    base = 13 + 9 * np.sin(2 * np.pi * (month - 3) / 12)  # seasonal temp curve
    temp = (base + rng.normal(0, 3, days)).round(1)
    precip = np.where(is_rain, rng.uniform(2, 30, days).round(1), 0.0)

    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "day_name": dates.day_name(),
            "is_weekend": ~weekday,
            "is_holiday": holiday,
            "holiday_name": [HOLIDAYS.get((d.month, d.day), "") for d in dates],
            "month": month,
            "week_of_year": dates.isocalendar().week.astype(int),
            "season": season,
            "weather": weather,
            "temp_c": temp,
            "precip_mm": precip,
        }
    )
    logger.info("calendar: %d days (2024 leap year)", len(frame))
    return frame


def build_events(calendar: pd.DataFrame, seed: int = 21) -> pd.DataFrame:
    """One-off events: sports, concerts, festivals and road incidents.

    Each event boosts demand on routes touching its zone and/or adds delays.
    """
    rng = np.random.default_rng(seed)
    event_types = {
        "sports": {"demand": 1.35, "delay": 2.0},
        "concert": {"demand": 1.45, "delay": 3.0},
        "festival": {"demand": 1.30, "delay": 2.0},
        "accident": {"demand": 1.05, "delay": 12.0},
        "construction": {"demand": 1.0, "delay": 8.0},
    }

    n_events = int(len(calendar) * 0.12)
    dates = calendar["date"].to_numpy()
    idx = rng.integers(0, len(dates), n_events)
    kind = rng.choice(list(event_types), size=n_events, p=[0.2, 0.15, 0.25, 0.25, 0.15])
    zones = rng.choice(["residential", "commercial", "industrial", "mixed", "university"],
                       size=n_events)

    rows = []
    for i, (d, k, z) in enumerate(zip(idx, kind, zones), start=1):
        spec = event_types[k]
        rows.append(
            {
                "event_id": f"EVT-{i:03d}",
                "date": dates[d],
                "name": f"{k.title()} Event {i}",
                "type": k,
                "zone": z,
                "demand_multiplier": spec["demand"],
                "extra_delay_min": spec["delay"],
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    logger.info("special events: %d", len(frame))
    return frame