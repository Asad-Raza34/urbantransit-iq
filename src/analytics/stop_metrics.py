"""Stop-level performance and bottleneck detection.

Reads the 7.4M-row ``stop_facts`` layer once and collapses it to per-stop and
per-(stop, time-band) summary frames, plus an evidence-based bottleneck score:

    bottleneck_score = 100 * (0.4*delay  + 0.3*crowding + 0.3*volume)

where each factor is the stop's value normalized to a 0..1 scale. A stop is
flagged as a bottleneck when the score clears the configured percentile cut.
"""

import numpy as np
import pandas as pd

from config import settings
from src.analytics.core import clip100, read_processed, save
from src.log import get_logger

logger = get_logger(__name__)

BOTTLENECK_PERCENTILE = 0.6


def compute() -> tuple[pd.DataFrame, pd.DataFrame]:
    stops = read_processed("stop_facts")

    g = stops.groupby("stop_id")
    agg = pd.DataFrame(
        {
            "stop_id": g["stop_id"].count().index,
            "visits": g["stop_id"].count().values,
            "boardings": g["boarded"].sum().values,
            "alightings": g["alighted"].sum().values,
            "avg_arr_delay_min": g["arr_delay_min"].mean().values.round(2),
            "p95_arr_delay_min": g["arr_delay_min"].quantile(0.95).values.round(2),
            "on_time_share": g["arr_delay_min"].apply(
                lambda s: (s <= settings.DELAY_ON_TIME_MAX).mean()).values.round(4),
            "crowding_share": g["occupancy_after"].apply(
                lambda s: (s >= settings.CROWDING_HIGH).mean()).values.round(4),
            "avg_dwell_sec": g["dwell_sec"].mean().values.round(1),
            "peak_onboard_avg": g["onboard_after"].mean().values.round(1),
        }
    )

    stop_info = stops[["stop_id", "stop_name", "zone"]].drop_duplicates("stop_id")
    agg = agg.merge(stop_info, on="stop_id", how="left")

    peak = stops.groupby(["stop_id", "hour"])["boarded"].sum().groupby("stop_id").idxmax()
    max_board = stops.groupby(["stop_id", "hour"])["boarded"].sum().groupby("stop_id").max()
    agg["peak_hour"] = peak.map(lambda x: x[1] if isinstance(x, tuple) else x).astype(int).values
    agg["max_hourly_boardings"] = max_board.values

    delay_norm = np.minimum(agg["p95_arr_delay_min"] / 20.0, 1.0)
    vol_norm = agg["boardings"] / max(agg["boardings"].max(), 1)
    crowd_norm = agg["crowding_share"]
    score = 100.0 * (0.4 * delay_norm + 0.3 * crowd_norm + 0.3 * vol_norm)
    agg["bottleneck_score"] = clip100(round(score, 1))
    cut = agg["bottleneck_score"].quantile(BOTTLENECK_PERCENTILE)
    agg["is_bottleneck"] = agg["bottleneck_score"] >= cut
    agg["is_terminal"] = agg["visits"].le(agg["visits"].quantile(0.05))

    order = ["stop_id", "stop_name", "zone", "visits", "boardings", "alightings",
             "avg_arr_delay_min", "p95_arr_delay_min", "on_time_share",
             "crowding_share", "avg_dwell_sec", "peak_hour",
             "max_hourly_boardings", "bottleneck_score", "is_bottleneck",
             "is_terminal"]
    agg = agg[order].sort_values("bottleneck_score", ascending=False).reset_index(drop=True)

    tb = stops.groupby(["stop_id", "time_band"], as_index=False).agg(
        visits=("stop_id", "count"),
        boardings=("boarded", "sum"),
        alightings=("alighted", "sum"),
        avg_arr_delay_min=("arr_delay_min", "mean"),
        crowding_share=("occupancy_after", lambda s: (s >= settings.CROWDING_HIGH).mean()),
    )
    tb = tb.merge(stop_info, on="stop_id", how="left")
    tb["crowding_share"] = tb["crowding_share"].round(4)
    tb["avg_arr_delay_min"] = tb["avg_arr_delay_min"].round(2)
    tb = tb.sort_values(["stop_id", "time_band"]).reset_index(drop=True)

    return agg, tb


def build() -> dict[str, pd.DataFrame]:
    stop_metrics, stop_timeband = compute()
    save("stop_metrics", stop_metrics)
    save("stop_timeband", stop_timeband)
    return {"stop_metrics": len(stop_metrics), "stop_timeband": len(stop_timeband)}