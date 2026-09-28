"""Passenger segmentation.

Groups the 50k travellers into behaviour segments using the rules in
``settings.SEGMENT_RULES`` (frequent-trip and peak-hour thresholds). The full
``segmented_passengers`` frame powers the passenger analytics page; the
compact ``segment_summary`` power dashboards and the recommendation engine's
demand-side evidence.
"""

import pandas as pd

from config import settings
from src.analytics.core import read_processed, save
from src.log import get_logger

logger = get_logger(__name__)


def _segment_label(trips: int, peak_share: float) -> str:
    if trips >= settings.SEGMENT_RULES["frequent_trips_min"]:
        if peak_share >= settings.SEGMENT_RULES["peak_share_min"]:
            return "commuter"
        return "frequent"
    if trips >= 20:
        return "regular"
    if trips >= 4:
        return "occasional"
    return "rare"


def segmentation() -> pd.DataFrame:
    flow = read_processed("passenger_flow")
    flow["is_peak_ride"] = flow["time_band"].isin(["morning peak", "evening peak"])
    peak = flow.groupby("passenger_id")["is_peak_ride"].mean()

    g = flow.groupby("passenger_id").agg(
        trips=("ticket_id", "count"),
        avg_distance_km=("distance_km", "mean"),
        avg_travel_min=("travel_min", "mean"),
        total_spend=("fare", "sum"),
        active_days=("date", "nunique"),
        primary_pass_type=("pass_type", lambda s: s.mode().iloc[0] if len(s) else "single"),
    )
    g["peak_share"] = peak.reindex(g.index).fillna(0).round(3)
    g["segment"] = [ _segment_label(int(t), float(p)) for t, p
                     in zip(g["trips"], g["peak_share"]) ]
    g = g.reset_index()
    return g[[
        "passenger_id", "segment", "trips", "peak_share", "avg_distance_km",
        "avg_travel_min", "total_spend", "active_days", "primary_pass_type",
    ]].sort_values("trips", ascending=False).reset_index(drop=True)


def segment_summary(seg: pd.DataFrame) -> pd.DataFrame:
    order = ["commuter", "frequent", "regular", "occasional", "rare"]
    out = seg.groupby("segment").agg(
        passengers=("passenger_id", "count"),
        avg_trips=("trips", "mean"),
        avg_peak_share=("peak_share", "mean"),
        avg_distance_km=("avg_distance_km", "mean"),
        avg_spend=("total_spend", "mean"),
    )
    out["avg_trips"] = out["avg_trips"].round(1)
    out["avg_peak_share"] = out["avg_peak_share"].round(3)
    out["avg_distance_km"] = out["avg_distance_km"].round(1)
    out["avg_spend"] = out["avg_spend"].round(1)
    out = out.reindex([s for s in order if s in out.index]).reset_index()
    out["share_of_passengers"] = (out["passengers"] / out["passengers"].sum()).round(4)
    return out


def build() -> dict[str, int]:
    seg = save("segmented_passengers", segmentation())
    summ = save("segment_summary", segment_summary(seg))
    return {"segmented_passengers": len(seg), "segment_summary": len(summ)}