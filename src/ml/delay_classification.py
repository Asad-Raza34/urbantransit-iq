"""Delay severity / delay status classification for UrbanTransit IQ.

The classifier predicts the **operational delay category of a trip** from
information that is known before the trip departs.  The primary target is not
invented here: it is the platform's own operational definition
(``config.settings`` consumed by ``src.integrate.integrator._delay_severity``),
so an ML prediction means exactly the same thing as the analytics dashboards and
the alert rules:

======================  =========================================
on time                 departure delay <= ``DELAY_ON_TIME_MAX``
moderate                ``DELAY_ON_TIME_MAX`` < delay <= ``DELAY_MODERATE_MAX``
severe                  ``DELAY_MODERATE_MAX`` < delay <= ``DELAY_CRITICAL_MAX``
critical                delay > ``DELAY_CRITICAL_MAX``
======================  =========================================

A second, binary task ("Delay Status": on time vs delayed) and the earlier
balanced-threshold 4-class variant are available through the same machinery; see
:data:`TARGETS`.

Methodology guarantees enforced by this module (and covered by tests):

* **No target leakage.**  Every feature is either static (row A), strictly
  historical (row B, computed only from trips that departed earlier) or
  prediction-time schedule information (row C).  ``FEATURE_SPECS`` carries the
  provenance of each feature and ``FORBIDDEN_COLUMNS`` lists the trip-outcome
  columns that may never be used.
* **No test-set tuning.**  Selection is staged: ``--stage selection`` writes
  fold/validation metrics only, ``--stage final`` has to consume that frozen
  decision before it may touch the December test period once.
* **Every reported number is computed** by :func:`classification_metrics` from a
  single ``(y_true, y_pred)`` pair, and the markdown report is rendered from the
  persisted metrics JSON (see ``src/ml/delay_classification_report.py``).

Command line::

    python -m src.ml.delay_classification --stage selection
    python -m src.ml.delay_classification --stage final
    python -m src.ml.delay_classification --stage report
"""

from __future__ import annotations

import json
import platform
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import settings
from src.analytics.core import read_processed
from src.log import get_logger
from src.ml.demand import save_parquet
from src.paths import DATA_GENERATED, DATA_RAW, MODELS_PYTHON, REPORTS_ROOT

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 20240101
TRIP_FACTS_NAME = "trip_facts"

# Validation design: the last 28 days before the test window is the validation
# window; three expanding folds walk backwards from there.
VAL_DAYS = 28
N_FOLDS = 3
# Fold fits use the most recent rows of each fold's training window so the whole
# selection stage stays tractable.  The frozen model is always refit on every
# row available before the test window.
FOLD_TRAIN_CAP = 120_000
# how many top candidates get the extra expanding folds as a robustness check
ROBUSTNESS_TOP_N = 3


# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetSpec:
    """A classification target: ordered labels plus the delay cut points."""

    name: str
    labels: tuple[str, ...]
    thresholds: tuple[float, ...]
    definition: str
    source: str
    primary: bool = False

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    @property
    def positive_label(self) -> str:
        """Label used as the positive class for binary targets."""
        return self.labels[-1]

    def describe_thresholds(self) -> list[str]:
        out = []
        lower = None
        for i, label in enumerate(self.labels):
            upper = self.thresholds[i] if i < len(self.thresholds) else None
            if lower is None:
                out.append(f"{label}: delay <= {upper:g} min")
            elif upper is None:
                out.append(f"{label}: delay > {lower:g} min")
            else:
                out.append(f"{label}: {lower:g} < delay <= {upper:g} min")
            lower = upper
        return out

    def label_delay(self, delay: pd.Series) -> pd.Series:
        """Map delay minutes onto labels using the spec's cut points."""
        idx = np.searchsorted(np.asarray(self.thresholds, dtype=float),
                              delay.to_numpy(dtype=float), side="left")
        return pd.Series([self.labels[int(i)] for i in idx], index=delay.index, name="target")


# The platform's operational severity buckets (config.settings + integrator).
TARGET_SEVERITY = TargetSpec(
    name="severity",
    labels=("on time", "moderate", "severe", "critical"),
    thresholds=(settings.DELAY_ON_TIME_MAX, settings.DELAY_MODERATE_MAX,
                settings.DELAY_CRITICAL_MAX),
    definition="operational delay severity used by the analytics pipeline and alert rules",
    source="config.settings.DELAY_ON_TIME_MAX/_MODERATE_MAX/_CRITICAL_MAX (same cut points as src.integrate.integrator._delay_severity)",
    primary=True,
)

# Binary operational task: the SRS-facing on-time performance question.
TARGET_STATUS = TargetSpec(
    name="delay_status",
    labels=("on time", "delayed"),
    thresholds=(settings.DELAY_ON_TIME_MAX,),
    definition="on-time performance: is the trip delayed beyond the operational tolerance?",
    source="config.settings.DELAY_ON_TIME_MAX (the same threshold the reliability KPI uses)",
)

# Earlier balanced-threshold variant, kept for comparison only.  Its labels do
# NOT match the analytics pipeline ("moderate" means something else there).
TARGET_BALANCED = TargetSpec(
    name="severity_balanced",
    labels=("on time", "minor", "moderate", "severe"),
    thresholds=(4.0, 7.0, 12.0),
    definition="balanced-threshold variant introduced previously for class balance (labels collide with the analytics severity names)",
    source="historical project definition, superseded",
)

TARGETS: dict[str, TargetSpec] = {
    TARGET_SEVERITY.name: TARGET_SEVERITY,
    TARGET_STATUS.name: TARGET_STATUS,
    TARGET_BALANCED.name: TARGET_BALANCED,
}

PRIMARY_TARGET = TARGET_SEVERITY.name

# Backwards-compatible module level names (dashboard/tests import these).
CLASS_NAMES: list[str] = list(TARGET_SEVERITY.labels)
CLASS_ORDER: dict[str, int] = {label: i for i, label in enumerate(TARGET_SEVERITY.labels)}
MODEL_TYPES: list[str] = ["hist_gradient_boosting", "extra_trees", "random_forest",
                          "logistic_regression", "gradient_boosting", "ordinal_cumulative"]


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------
# availability: "A" static known before the trip, "B" historical known before the
# trip (strictly earlier departures), "C" same-trip prediction-time information.

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    availability: str
    source: str
    calculation: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # -- temporal / schedule -------------------------------------------------
    FeatureSpec("hour", "temporal", "A", "scheduled_departure",
                "hour of the scheduled departure"),
    FeatureSpec("minute_of_day", "temporal", "A", "scheduled_departure",
                "minutes since midnight of the scheduled departure"),
    FeatureSpec("weekday", "temporal", "A", "scheduled_departure", "0=Monday..6=Sunday"),
    FeatureSpec("month", "temporal", "A", "scheduled_departure", "calendar month 1..12"),
    FeatureSpec("is_peak_int", "temporal", "A", "scheduled_departure",
                "generator peak window 07:00-09:20 / 16:00-19:00"),
    FeatureSpec("is_weekend_int", "temporal", "A", "calendar", "weekend flag"),
    FeatureSpec("is_holiday_int", "temporal", "A", "calendar", "public holiday flag"),
    FeatureSpec("days_since_holiday", "temporal", "A", "calendar",
                "days since the most recent public holiday"),
    FeatureSpec("day_type_code", "temporal", "A", "calendar",
                "weekday=0, weekend=1, holiday=2"),
    FeatureSpec("season_code", "temporal", "A", "calendar",
                "winter=0, spring=1, summer=2, autumn=3"),
    FeatureSpec("time_band_code", "temporal", "A", "scheduled_departure",
                "night=0, morning peak=1, midday=2, evening peak=3"),
    FeatureSpec("hour_sin", "temporal", "A", "scheduled_departure", "sin(2*pi*hour/24)"),
    FeatureSpec("hour_cos", "temporal", "A", "scheduled_departure", "cos(2*pi*hour/24)"),
    FeatureSpec("minute_of_day_sin", "temporal", "A", "scheduled_departure",
                "sin(2*pi*minute_of_day/1440)"),
    FeatureSpec("minute_of_day_cos", "temporal", "A", "scheduled_departure",
                "cos(2*pi*minute_of_day/1440)"),
    FeatureSpec("weekday_sin", "temporal", "A", "scheduled_departure", "sin(2*pi*weekday/7)"),
    FeatureSpec("weekday_cos", "temporal", "A", "scheduled_departure", "cos(2*pi*weekday/7)"),
    FeatureSpec("month_sin", "temporal", "A", "scheduled_departure", "sin(2*pi*month/12)"),
    FeatureSpec("month_cos", "temporal", "A", "scheduled_departure", "cos(2*pi*month/12)"),
    # -- route / trip plan ---------------------------------------------------
    FeatureSpec("route_category_code", "route", "A", "routes.category",
                "core=0, feeder=1, express=2"),
    FeatureSpec("route_length_km", "route", "A", "routes.length_km", "route length"),
    FeatureSpec("distance_km", "route", "A", "routes.length_km",
                "trip distance (equals route length on this network)"),
    FeatureSpec("scheduled_travel_min", "schedule", "A", "timetable",
                "planned end-to-end running time incl. dwell"),
    FeatureSpec("scheduled_headway_min", "schedule", "A", "timetable",
                "minutes to the previous scheduled departure on the same date/direction"),
    FeatureSpec("headway_pressure", "schedule", "A", "timetable",
                "route mean scheduled headway / this trip's scheduled headway"),
    # -- vehicle -------------------------------------------------------------
    FeatureSpec("vehicle_type_code", "vehicle", "A", "vehicles.type",
                "standard=0, articulated=1, mini=2"),
    FeatureSpec("vehicle_capacity", "vehicle", "A", "vehicles.capacity",
                "seated+standing capacity of the assigned vehicle"),
    # NOTE: the generator's hidden state (route chronic offset, per-route
    # congestion/weather sensitivity, route-day disruption regime, vehicle
    # reliability, daily systemic factor, day-of-week effect) is deliberately
    # NOT a feature.  Those quantities are additive/near-additive components of
    # ``departure_delay_min``; exposing them would hand the model the answer
    # (target leakage) instead of making it infer them from earlier departures.
    # The ``history_*`` features below recover the same signal honestly.
    # -- weather (observed for the operating day) ----------------------------
    FeatureSpec("weather_code", "weather", "A", "calendar.weather",
                "clear=0, rain=1, snow=2, storm=3"),
    FeatureSpec("precip_mm", "weather", "A", "calendar.precip_mm", "daily precipitation"),
    FeatureSpec("temp_c", "weather", "A", "calendar.temp_c", "daily mean temperature"),
    FeatureSpec("snow_int", "weather", "A", "calendar.weather", "weather == snow"),
    FeatureSpec("storm_int", "weather", "A", "calendar.weather", "weather == storm"),
    FeatureSpec("severe_weather_int", "weather", "A", "calendar",
                "weather in {snow, storm} or precip > 10 or temp < 0"),
    # -- planned special events ---------------------------------------------
    FeatureSpec("has_event", "event", "A", "events",
                "an event touches one of the route's stop zones on this date"),
    FeatureSpec("event_extra_delay", "event", "A", "events.extra_delay_min",
                "generator delay contribution of that event for this route"),
    FeatureSpec("event_demand_mult", "event", "A", "events.demand_multiplier",
                "generator demand multiplier of that event"),
    # -- historical: same route, same operating day (real-time feed) ---------
    FeatureSpec("route_date_prior_trip_count", "history_same_day", "B",
                "earlier trips today on this route",
                "number of trips of this route that already departed today"),
    FeatureSpec("route_date_prior_delay_mean", "history_same_day", "B",
                "earlier trips today on this route",
                "mean departure delay of those trips"),
    FeatureSpec("route_date_prior_delay_std", "history_same_day", "B",
                "earlier trips today on this route",
                "standard deviation of those delays"),
    FeatureSpec("route_date_prior_delay_max", "history_same_day", "B",
                "earlier trips today on this route",
                "worst departure delay observed today on this route"),
    FeatureSpec("route_date_peak_prior_delay_mean", "history_same_day", "B",
                "earlier trips today on this route with the same peak flag",
                "mean delay of already-departed trips in the same peak/off-peak regime"),
    FeatureSpec("vehicle_date_prior_delay_mean", "history_same_day", "B",
                "earlier trips today of the same vehicle",
                "mean delay of the vehicle's already-completed trips today"),
    FeatureSpec("network_prior_delay_mean", "history_same_day", "B",
                "all trips that already departed earlier today",
                "expanding mean delay over the whole network earlier the same day "
                "(the control room's view of how bad today is), current trip excluded"),
    FeatureSpec("network_prior_trip_count", "history_same_day", "B",
                "all trips that already departed earlier today",
                "how far into the operating day this trip departs"),
    FeatureSpec("vehicle_prior_delay_mean", "history_vehicle", "B",
                "all earlier departures of this vehicle",
                "expanding mean delay of that vehicle's past trips, current trip excluded"),
    # -- historical: route expanding statistics ------------------------------
    FeatureSpec("route_prior_delay_mean", "history_route", "B",
                "all earlier trips of this route",
                "expanding mean of past departure delays, current trip excluded"),
    FeatureSpec("route_prior_delay_std", "history_route", "B",
                "all earlier trips of this route",
                "expanding standard deviation, current trip excluded"),
    FeatureSpec("route_prior_delay_p90", "history_route", "B",
                "all earlier trips of this route",
                "mean + 1.2816 * std of past delays (normal-quantile proxy for p90)"),
    FeatureSpec("route_prior_delay_rolling7", "history_route", "B",
                "the 7 trips before this one on this route",
                "mean delay of the previous up-to-7 trips on the route"),
    FeatureSpec("route_prior_delay_rolling28", "history_route", "B",
                "the 28 trips before this one on this route",
                "mean delay of the previous up-to-28 trips on the route"),
    FeatureSpec("route_timeband_prior_delay_mean", "history_route", "B",
                "earlier trips of this route in the same time band",
                "expanding mean per route x time band, current trip excluded"),
    FeatureSpec("route_weather_prior_delay_mean", "history_route", "B",
                "earlier trips of this route in the same weather class",
                "expanding mean per route x weather, current trip excluded"),
    FeatureSpec("route_prior_on_time_rate", "history_route", "B",
                "all earlier trips of this route",
                "share of past trips within the on-time tolerance (lagged label aggregate)"),
    FeatureSpec("route_prior_moderate_plus_rate", "history_route", "B",
                "all earlier trips of this route",
                "share of past trips beyond the moderate tolerance (lagged label aggregate)"),
)

FEATURE_NAMES: list[str] = [spec.name for spec in FEATURE_SPECS]

# Trip-outcome columns that are produced by (or after) the trip itself.  Using
# any of them as a feature would be target leakage or future information.
FORBIDDEN_COLUMNS: tuple[str, ...] = (
    "departure_delay_min",       # the target itself
    "delay_severity",            # the target label
    "on_time",                   # target-derived label
    "actual_departure",          # target-derived
    "actual_travel_min",         # recorded after the trip
    "actual_headway_min",        # = delay_i - delay_{i-1}: algebraic leakage
    "boardings", "alightings",   # accumulated after departure
    "occupancy_avg_pct", "occupancy_max_pct",  # accumulated after departure
    "arr_delay_min", "actual_arrival",
    # generator hidden state: additive components of departure_delay_min that a
    # model must recover from history rather than read off the row
    "route_latent_offset", "route_peak_sensitivity", "route_weather_sensitivity",
    "disruption_delay", "vehicle_reliability", "daily_systemic_delay",
    "dow_delay_effect",
)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_KEEP = (
    "trip_id", "route_id", "vehicle_id", "date", "day_type", "direction",
    "is_peak", "hour", "scheduled_departure", "weather", "distance_km",
    "scheduled_travel_min", "capacity_x", "scheduled_headway_min",
    "route_category", "route_length_km", "vehicle_type", "weekday", "month",
    "season", "is_weekend", "time_band", "departure_delay_min",
)

DAY_TYPE_CODE = {"weekday": 0, "weekend": 1, "holiday": 2}
WEATHER_CODE = {"clear": 0, "rain": 1, "snow": 2, "storm": 3}
SEASON_CODE = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
TIME_BAND_CODE = {"night": 0, "morning peak": 1, "midday": 2, "evening peak": 3}
ROUTE_CATEGORY_CODE = {"core": 0, "feeder": 1, "express": 2}
VEHICLE_TYPE_CODE = {"standard": 0, "articulated": 1, "mini": 2}


def _trip_facts_path():
    from src.paths import DATA_PROCESSED
    return DATA_PROCESSED / f"{TRIP_FACTS_NAME}.parquet"


def _fingerprint() -> str:
    """Cache key: the trip_facts file identity plus the inputs it depends on."""
    parts = []
    for path in (_trip_facts_path(), DATA_GENERATED / "calendar.parquet",
                 DATA_GENERATED / "events.parquet", DATA_RAW / "route_stops.parquet",
                 DATA_RAW / "stops.parquet"):
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}")
        except OSError:
            parts.append(f"{path.name}:missing")
    return "|".join(parts)


def _prior_stats(frame: pd.DataFrame, keys: Sequence[str], col: str,
                 prefix: str) -> None:
    """Add expanding mean/std of ``col`` per ``keys``, *excluding* the current row.

    Implemented with cumulative sums so it stays vectorised on 500k+ rows.  The
    current row is always subtracted, which is the guarantee that makes these
    features leakage-free.
    """
    grouped = frame.groupby(list(keys), sort=False)[col]
    count_before = grouped.cumcount()
    denom = count_before.replace(0, np.nan)
    total = grouped.cumsum()
    current = frame[col]
    mean_before = (total - current) / denom

    frame["__sq"] = current * current
    total_sq = frame.groupby(list(keys), sort=False)["__sq"].cumsum()
    var_before = ((total_sq - frame["__sq"]) / denom) - mean_before ** 2
    frame.drop(columns=["__sq"], inplace=True)

    frame[f"{prefix}_mean"] = mean_before
    frame[f"{prefix}_std"] = np.sqrt(np.clip(var_before, 0.0, None))
    frame[f"{prefix}_count"] = count_before


def _prior_stats_by_time(frame: pd.DataFrame, keys: Sequence[str], col: str,
                         prefix: str) -> None:
    """Expanding statistics per ``keys`` computed in *chronological* order.

    :func:`_prior_stats` walks the frame in its current order, which is
    route-major.  For keys that span routes (the operating day as a whole, or a
    vehicle) that order is not chronological, so a trip could otherwise see
    departures that happened *later* the same day.  Here the frame is sorted by
    ``(keys, date, scheduled_departure)`` first and the statistics are mapped
    back onto the original row positions.
    """
    order_keys = list(keys) + [c for c in ("date", "scheduled_departure")
                               if c not in keys]
    work = frame[order_keys + [col]].copy()
    work["__pos"] = np.arange(len(frame))
    work = work.sort_values([*order_keys, "__pos"], kind="mergesort")
    _prior_stats(work, list(keys), col, prefix)
    pos = work["__pos"].to_numpy()
    for suffix in ("mean", "std", "count"):
        out = np.full(len(frame), np.nan)
        out[pos] = work[f"{prefix}_{suffix}"].to_numpy()
        frame[f"{prefix}_{suffix}"] = out


def _prior_max(frame: pd.DataFrame, keys: Sequence[str], col: str, name: str) -> None:
    grouped = frame.groupby(list(keys), sort=False)[col]
    running_max = grouped.cummax()
    frame[name] = running_max.groupby(
        [frame[k] for k in keys], sort=False).shift(1)


def _prior_rate(frame: pd.DataFrame, keys: Sequence[str], flag: pd.Series,
                name: str) -> None:
    """Expanding share of past trips with ``flag`` true, current row excluded."""
    tmp = "__flag"
    frame[tmp] = flag.astype(float)
    grouped = frame.groupby(list(keys), sort=False)[tmp]
    count_before = grouped.cumcount()
    total = grouped.cumsum()
    frame[name] = (total - frame[tmp]) / count_before.replace(0, np.nan)
    frame.drop(columns=[tmp], inplace=True)


def _rolling_prior_mean(frame: pd.DataFrame, keys: Sequence[str], col: str,
                        window: int, name: str) -> None:
    shifted = frame.groupby(list(keys), sort=False)[col].shift(1)
    rolled = (shifted.groupby([frame[k] for k in keys], sort=False)
              .rolling(window, min_periods=1).mean())
    frame[name] = rolled.reset_index(level=0, drop=True).sort_index()


def _event_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Route-zone matched special-event effects (mirrors the generator).

    The generator applies an event's ``extra_delay_min`` to a route only when
    the event's zone is one of the route's stop zones, and keeps the last such
    event of the day.  Reproducing that mapping exactly is what makes
    ``event_extra_delay`` a genuine prediction-time feature instead of a
    date-level average.
    """
    events = pd.read_parquet(DATA_GENERATED / "events.parquet")
    events["date"] = pd.to_datetime(events["date"])
    route_stops = pd.read_parquet(DATA_RAW / "route_stops.parquet")
    stops = pd.read_parquet(DATA_RAW / "stops.parquet")
    zone_of = stops.set_index("stop_id")["zone"]
    route_zones = {str(rid): set(zone_of.reindex(g["stop_id"]).dropna())
                   for rid, g in route_stops.groupby("route_id")}

    by_date: dict[pd.Timestamp, list[tuple[str, float, float]]] = {}
    for row in events.itertuples():
        by_date.setdefault(row.date, []).append(
            (row.zone, float(row.demand_multiplier), float(row.extra_delay_min)))

    rows: list[tuple[str, pd.Timestamp, float, float]] = []
    for route_id, zones in route_zones.items():
        for date, pairs in by_date.items():
            hit = None
            for zone, mult, extra in pairs:
                if zone in zones:
                    hit = (mult, extra)
            if hit is not None:
                rows.append((route_id, date, hit[0], hit[1]))

    if not rows:
        frame["event_demand_mult"] = 1.0
        frame["event_extra_delay"] = 0.0
        return frame

    effects = pd.DataFrame(rows, columns=["route_id", "date",
                                          "event_demand_mult", "event_extra_delay"])
    merged = frame.merge(effects, on=["route_id", "date"], how="left")
    merged["event_demand_mult"] = merged["event_demand_mult"].fillna(1.0)
    merged["event_extra_delay"] = merged["event_extra_delay"].fillna(0.0)
    merged["has_event"] = (merged["event_extra_delay"] > 0).astype(int)
    return merged


def _build_features_uncached() -> pd.DataFrame:
    """Build the trip-level feature frame (target column included, no imputation)."""
    raw = read_processed(TRIP_FACTS_NAME)
    missing = [c for c in _KEEP if c not in raw.columns]
    if missing:
        raise KeyError(f"trip_facts is missing expected columns: {missing}")
    df = raw[list(_KEEP)].copy()
    del raw

    df["date"] = pd.to_datetime(df["date"])
    df["scheduled_departure"] = pd.to_datetime(df["scheduled_departure"])
    # Chronological order *within* every group; scheduled_departure is known in
    # advance, so ordering by it cannot smuggle in outcome information.
    df = df.sort_values(["route_id", "date", "scheduled_departure"]).reset_index(drop=True)

    cal = pd.read_parquet(DATA_GENERATED / "calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"])
    df = df.merge(cal[["date", "temp_c", "precip_mm", "is_holiday"]], on="date", how="left")

    # ---- static encodings -------------------------------------------------
    df["day_type_code"] = df["day_type"].map(DAY_TYPE_CODE).fillna(0).astype(int)
    df["weather_code"] = df["weather"].map(WEATHER_CODE).fillna(0).astype(int)
    df["season_code"] = df["season"].map(SEASON_CODE).fillna(0).astype(int)
    df["time_band_code"] = df["time_band"].map(TIME_BAND_CODE).fillna(0).astype(int)
    df["route_category_code"] = df["route_category"].map(ROUTE_CATEGORY_CODE).fillna(0).astype(int)
    df["vehicle_type_code"] = df["vehicle_type"].map(VEHICLE_TYPE_CODE).fillna(0).astype(int)
    df["is_peak_int"] = df["is_peak"].astype(int)
    df["is_weekend_int"] = df["is_weekend"].astype(int)
    df["is_holiday_int"] = df["is_holiday"].fillna(False).astype(int)
    df["vehicle_capacity"] = df["capacity_x"].astype(int)

    dep = df["scheduled_departure"]
    df["minute_of_day"] = (dep.dt.hour * 60 + dep.dt.minute).astype(int)
    df["hour"] = dep.dt.hour.astype(int)
    df["weekday"] = dep.dt.weekday.astype(int)
    df["month"] = dep.dt.month.astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_of_day_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["minute_of_day_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    holiday_dates = np.sort(df.loc[df["is_holiday_int"] == 1, "date"].unique())
    if len(holiday_dates):
        pos = np.searchsorted(holiday_dates, df["date"].to_numpy())
        previous = np.where(pos > 0, holiday_dates[np.clip(pos - 1, 0, None)],
                            holiday_dates[0] - np.timedelta64(365, "D"))
        df["days_since_holiday"] = ((df["date"].to_numpy() - previous)
                                    / np.timedelta64(1, "D")).astype(float)
        df.loc[pos == 0, "days_since_holiday"] = 365.0
    else:
        df["days_since_holiday"] = 365.0

    # ---- weather ----------------------------------------------------------
    df["precip_mm"] = df["precip_mm"].fillna(0.0).astype(float)
    df["temp_c"] = df["temp_c"].astype(float)
    df["snow_int"] = (df["weather"] == "snow").astype(int)
    df["storm_int"] = (df["weather"] == "storm").astype(int)
    df["severe_weather_int"] = ((df["snow_int"] == 1) | (df["storm_int"] == 1)
                                 | (df["precip_mm"] > 10)
                                 | (df["temp_c"] < 0)).astype(int)

    # ---- schedule pressure ------------------------------------------------
    route_headway = df.groupby("route_id", sort=False)["scheduled_headway_min"].transform("mean")
    df["headway_pressure"] = (route_headway
                              / df["scheduled_headway_min"].replace(0, np.nan)).fillna(1.0)

    # ---- events -----------------------------------------------------------
    df = _event_effects(df)

    # ---- historical features (strictly earlier departures) ----------------
    delay = df["departure_delay_min"]
    same_day = ["route_id", "date"]
    _prior_stats(df, same_day, "departure_delay_min", "route_date_prior_delay")
    _prior_max(df, same_day, "departure_delay_min", "route_date_prior_delay_max")
    df.rename(columns={"route_date_prior_delay_count": "route_date_prior_trip_count"},
              inplace=True)
    _prior_stats(df, ["route_id", "date", "is_peak"], "departure_delay_min",
                 "__route_date_peak_prior_delay")
    df.rename(columns={"__route_date_peak_prior_delay_mean":
                       "route_date_peak_prior_delay_mean"}, inplace=True)
    df.drop(columns=["__route_date_peak_prior_delay_std",
                     "__route_date_peak_prior_delay_count"], inplace=True)
    _prior_stats(df, ["vehicle_id", "date"], "departure_delay_min",
                 "__vehicle_date_prior_delay")
    df.rename(columns={"__vehicle_date_prior_delay_mean": "vehicle_date_prior_delay_mean"},
              inplace=True)
    df.drop(columns=["__vehicle_date_prior_delay_std",
                     "__vehicle_date_prior_delay_count"], inplace=True)

    # how bad the operating day already is network-wide, and how this vehicle
    # has been running: both are observable from the trip feed at prediction
    # time, and both must be built in true chronological order
    _prior_stats_by_time(df, ["date"], "departure_delay_min", "network_prior_delay")
    df.rename(columns={"network_prior_delay_count": "network_prior_trip_count"},
              inplace=True)
    df.drop(columns=["network_prior_delay_std"], inplace=True)
    _prior_stats_by_time(df, ["vehicle_id"], "departure_delay_min", "vehicle_prior_delay")
    df.drop(columns=["vehicle_prior_delay_std", "vehicle_prior_delay_count"],
            inplace=True)

    _prior_stats(df, ["route_id"], "departure_delay_min", "route_prior_delay")
    df["route_prior_delay_p90"] = (df["route_prior_delay_mean"]
                                   + 1.2816 * df["route_prior_delay_std"])
    _prior_stats(df, ["route_id", "time_band"], "departure_delay_min",
                 "__route_timeband_prior_delay")
    df.rename(columns={"__route_timeband_prior_delay_mean":
                       "route_timeband_prior_delay_mean"}, inplace=True)
    df.drop(columns=["__route_timeband_prior_delay_std",
                     "__route_timeband_prior_delay_count"], inplace=True)
    _prior_stats(df, ["route_id", "weather"], "departure_delay_min",
                 "__route_weather_prior_delay")
    df.rename(columns={"__route_weather_prior_delay_mean":
                       "route_weather_prior_delay_mean"}, inplace=True)
    df.drop(columns=["__route_weather_prior_delay_std",
                     "__route_weather_prior_delay_count"], inplace=True)
    _rolling_prior_mean(df, ["route_id"], "departure_delay_min", 7,
                        "route_prior_delay_rolling7")
    _rolling_prior_mean(df, ["route_id"], "departure_delay_min", 28,
                        "route_prior_delay_rolling28")

    # lagged label aggregates (strictly past trips only)
    _prior_rate(df, ["route_id"], delay <= settings.DELAY_ON_TIME_MAX,
                "route_prior_on_time_rate")
    _prior_rate(df, ["route_id"], delay > settings.DELAY_MODERATE_MAX,
                "route_prior_moderate_plus_rate")

    df["route_prior_delay_count"] = df.groupby("route_id", sort=False).cumcount()
    df = df.drop(columns=[c for c in df.columns if c.startswith("__")])
    return df


def build_features(force: bool = False) -> pd.DataFrame:
    """Feature frame for the whole dataset (cached per process)."""
    global _FEATURE_CACHE, _FEATURE_CACHE_KEY
    key = _fingerprint()
    if force or _FEATURE_CACHE is None or _FEATURE_CACHE_KEY != key:
        t0 = time.perf_counter()
        _FEATURE_CACHE = _build_features_uncached()
        _FEATURE_CACHE_KEY = key
        logger.info("built delay-classification features: %d rows x %d columns in %.1fs",
                    len(_FEATURE_CACHE), _FEATURE_CACHE.shape[1],
                    time.perf_counter() - t0)
    return _FEATURE_CACHE


_FEATURE_CACHE: pd.DataFrame | None = None
_FEATURE_CACHE_KEY: str | None = None


# Legacy alias kept for any caller of the previous implementation.
_load_classification_features = build_features


# ---------------------------------------------------------------------------
# Target attachment, splits and folds
# ---------------------------------------------------------------------------

def attach_target(df: pd.DataFrame, target: TargetSpec) -> pd.DataFrame:
    """Add the ``target`` label column derived from ``departure_delay_min``."""
    out = df.copy()
    out["target"] = target.label_delay(out["departure_delay_min"])
    return out


def split_train_val_test(df: pd.DataFrame, val_days: int = VAL_DAYS,
                         horizon_days: int | None = None,
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train | validation | test split (no shuffling anywhere)."""
    horizon_days = horizon_days or settings.FORECAST_HORIZON_DAYS
    max_date = df["date"].max().normalize()
    cutoff_test = max_date - pd.Timedelta(days=horizon_days)
    cutoff_val = cutoff_test - pd.Timedelta(days=val_days)
    train = df[df["date"] <= cutoff_val].copy()
    val = df[(df["date"] > cutoff_val) & (df["date"] <= cutoff_test)].copy()
    test = df[df["date"] > cutoff_test].copy()
    logger.info("split -> train %d (<= %s) | val %d (%s..%s) | test %d (%s..%s)",
                len(train), cutoff_val.date(), len(val),
                val["date"].min().date(), val["date"].max().date(), len(test),
                test["date"].min().date(), test["date"].max().date())
    return train, val, test


def cutoff_dates(val_days: int = VAL_DAYS, horizon_days: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(cutoff_val, cutoff_test)`` derived from the data itself."""
    horizon_days = horizon_days or settings.FORECAST_HORIZON_DAYS
    dates = pd.read_parquet(_trip_facts_path(), columns=["date"])["date"]
    max_date = pd.to_datetime(dates).max().normalize()
    del dates
    cutoff_test = max_date - pd.Timedelta(days=horizon_days)
    return cutoff_test - pd.Timedelta(days=val_days), cutoff_test


def expanding_folds(df: pd.DataFrame, n_folds: int = N_FOLDS,
                    val_days: int = VAL_DAYS) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window folds that never touch the test period.

    Fold ``i`` validates on the ``i``-th 28-day block counted backwards from the
    end of the development window; its training part is everything before that
    block.  The newest fold is exactly the validation window used by
    :func:`split_train_val_test`.
    """
    df = df.reset_index(drop=True)
    dev_end = df["date"].max().normalize()
    folds = []
    for i in range(n_folds):
        val_end = dev_end - pd.Timedelta(days=i * val_days)
        val_start = val_end - pd.Timedelta(days=val_days)
        train = df[df["date"] <= val_start]
        val = df[(df["date"] > val_start) & (df["date"] <= val_end)]
        if len(train) == 0 or len(val) == 0:
            continue
        if FOLD_TRAIN_CAP and len(train) > FOLD_TRAIN_CAP:
            train = train.tail(FOLD_TRAIN_CAP)
        folds.append((train.copy(), val.copy()))
    return folds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray | None = None,
                           labels: Sequence[str] | None = None) -> dict[str, Any]:
    """Full metric block computed from one ``(y_true, y_pred)`` pair.

    Everything (accuracy, macro/weighted F1, balanced accuracy, per-class
    precision/recall/F1/support and the confusion matrix) comes from the same
    two arrays, so supports, percentages and the matrix cannot disagree.
    """
    labels = list(labels or sorted(set(np.asarray(y_true, dtype=object)) |
                                   set(np.asarray(y_pred, dtype=object))))
    labels = [lbl for lbl in labels if lbl in set(np.asarray(y_true, dtype=object))]
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(len(y_true))

    per_class = {
        label: {
            "precision": round(float(report[label]["precision"]), 4),
            "recall": round(float(report[label]["recall"]), 4),
            "f1-score": round(float(report[label]["f1-score"]), 4),
            "support": int(cm[i].sum()),
            "share": round(float(cm[i].sum() / total), 4) if total else 0.0,
        }
        for i, label in enumerate(labels)
    }

    out: dict[str, Any] = {
        "n": total,
        "labels": labels,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_precision": round(float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "weighted_recall": round(float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_counts": {label: int(cm[i].sum()) for i, label in enumerate(labels)},
        "class_shares": {label: round(float(cm[i].sum() / total), 4) if total else 0.0
                         for i, label in enumerate(labels)},
    }
    if y_prob is not None and len(labels) == 2:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            positive = labels[1]
            binary = (y_true == positive).astype(int)
            scores = np.asarray(y_prob)[:, 1] if np.ndim(y_prob) == 2 else np.asarray(y_prob)
            out["roc_auc"] = round(float(roc_auc_score(binary, scores)), 4)
            out["pr_auc"] = round(float(average_precision_score(binary, scores)), 4)
        except Exception as exc:  # noqa: BLE001
            out["auc_error"] = str(exc)[:120]
    return out


def check_metric_consistency(metrics: dict[str, Any]) -> list[str]:
    """Return the list of internal inconsistencies found in a metrics block.

    Used by the test-suite and by the report writer, so a report can never
    publish numbers that do not agree with each other.
    """
    problems: list[str] = []
    labels = metrics["labels"]
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    total = metrics["n"]

    if cm.shape != (len(labels), len(labels)):
        problems.append(f"confusion matrix shape {cm.shape} != {len(labels)} labels")
        return problems
    if int(cm.sum()) != total:
        problems.append(f"confusion matrix sums to {int(cm.sum())} but n={total}")

    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    if abs(acc - metrics["accuracy"]) > 1e-4:
        problems.append(f"accuracy {metrics['accuracy']} != trace/n {acc:.4f}")

    f1s = []
    for i, label in enumerate(labels):
        support = int(cm[i].sum())
        if support != metrics["per_class"][label]["support"]:
            problems.append(f"{label}: support {metrics['per_class'][label]['support']} != row sum {support}")
        if abs(metrics["per_class"][label]["share"] - support / total) > 1e-4:
            problems.append(f"{label}: share does not match support/n")
        f1s.append(metrics["per_class"][label]["f1-score"])
    macro = float(np.mean(f1s))
    if abs(macro - metrics["macro_f1"]) > 1e-3:
        problems.append(f"macro F1 {metrics['macro_f1']} != mean per-class F1 {macro:.4f}")

    sums = metrics["class_shares"]
    if abs(sum(sums.values()) - 1.0) > 1e-3:
        problems.append(f"class shares sum to {sum(sums.values()):.4f}")
    return problems


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

WEIGHT_KINDS = ("none", "balanced", "sqrt")


def class_weight_mapping(y: pd.Series, kind: str, labels: Sequence[str]) -> dict | None:
    """Per-class weights for a weighting strategy.

    ``none``     - every trip counts once (the natural class distribution).
    ``balanced`` - sklearn's inverse-frequency weights (n / (k * n_c)).
    ``sqrt``     - square-root inverse frequency: a deliberately *moderate*
                   correction that keeps the majority class informative.
    """
    if kind == "none":
        return None
    values = np.asarray(y, dtype=object)
    counts = np.array([max(int((values == lbl).sum()), 1) for lbl in labels], dtype=float)
    n, k = len(values), len(labels)
    weights = n / (k * counts)
    if kind == "sqrt":
        weights = np.sqrt(weights)
    weights = weights / weights.mean()
    return {str(lbl): float(weights[i]) for i, lbl in enumerate(labels)}


def class_weight_vector(y: pd.Series, kind: str, labels: Sequence[str]) -> np.ndarray | None:
    """Per-row sample weights for a weighting strategy (``None`` for ``none``)."""
    mapping = class_weight_mapping(y, kind, labels)
    if mapping is None:
        return None
    values = np.asarray(y, dtype=object)
    return np.array([mapping[str(v)] for v in values], dtype=float)


def fit_estimator(model: Any, X, y, weights: np.ndarray | None) -> Any:
    """``fit`` that never passes ``sample_weight`` when there are no weights.

    ``Pipeline.fit`` rejects the keyword outright in scikit-learn 1.5 even when
    it is ``None``, so the call has to be split.
    """
    if weights is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=weights)
    return model


def weighted_estimator(model: Any, y: pd.Series, kind: str, labels: Sequence[str]
                       ) -> tuple[Any, np.ndarray | None]:
    """Apply a weighting strategy the way the estimator supports it.

    Estimators that expose ``class_weight`` (random forest, extra trees,
    histogram gradient boosting, logistic regression) get the mapping directly;
    the rest (plain gradient boosting, the ordinal wrapper) receive per-row
    sample weights.  Returns ``(model, sample_weight_or_None)``.
    """
    mapping = class_weight_mapping(y, kind, labels)
    if mapping is None:
        return model, None
    target = model.steps[-1][1] if isinstance(model, Pipeline) else model
    supports_native = isinstance(target, (LogisticRegression,
                                          RandomForestClassifier,
                                          ExtraTreesClassifier))
    if supports_native:
        try:
            target.set_params(class_weight=mapping)
            return model, None
        except Exception:  # noqa: BLE001  (estimator without a real setter)
            pass
    # HistGradientBoosting recodes string labels internally and would need the
    # dict keyed by the numeric codes - per-row sample weights are equivalent.
    return model, class_weight_vector(y, kind, labels)


class OrdinalCumulativeClassifier:
    """Ordinal decomposition for ordered classes (Phase 12).

    Fits ``K-1`` binary models on the cumulative question *"is the delay worse
    than class k?"*, then reconstructs class probabilities from the monotone
    differences of those cumulative probabilities.  This respects the ordering
    of the severity scale instead of treating the classes as unrelated.
    """

    def __init__(self, base_family: str = "hist_gradient_boosting",
                 base_params: dict | None = None, labels: Sequence[str] = ()):
        self.base_family = base_family
        self.base_params = dict(base_params or {})
        self.labels = list(labels)
        self.models_: list[Any] = []
        self.classes_ = np.array(self.labels, dtype=object)

    def fit(self, X, y, sample_weight=None):
        codes = np.array([self.labels.index(v) for v in np.asarray(y, dtype=object)])
        self.models_ = []
        for k in range(len(self.labels) - 1):
            binary = (codes > k).astype(int)
            if len(np.unique(binary)) < 2:
                self.models_.append(None)
                continue
            model = make_estimator(self.base_family, self.base_params)
            fit_estimator(model, X, binary, sample_weight)
            self.models_.append(model)
        return self

    def _cumulative(self, X) -> np.ndarray:
        cum = np.zeros((len(X), len(self.labels) - 1))
        for k, model in enumerate(self.models_):
            if model is None:
                cum[:, k] = 0.0
                continue
            proba = model.predict_proba(X)
            classes = list(model.classes_)
            idx = classes.index(1) if 1 in classes else len(classes) - 1
            cum[:, k] = proba[:, idx]
        # enforce monotone non-increasing cumulative probabilities
        return np.minimum.accumulate(cum, axis=1)

    def predict_proba(self, X) -> np.ndarray:
        cum = self._cumulative(X)
        probs = np.zeros((len(X), len(self.labels)))
        probs[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, len(self.labels) - 1):
            probs[:, k] = np.clip(cum[:, k - 1] - cum[:, k], 0.0, 1.0)
        probs[:, -1] = cum[:, -1]
        total = probs.sum(axis=1, keepdims=True)
        return probs / np.where(total > 0, total, 1.0)

    def predict(self, X) -> np.ndarray:
        idx = self.predict_proba(X).argmax(axis=1)
        return self.classes_[idx]


def make_estimator(family: str, params: dict) -> Any:
    """Build an unfitted estimator for a family name."""
    if family == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=RANDOM_SEED, **params)),
        ])
    if family == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **params)
    if family == "extra_trees":
        return ExtraTreesClassifier(random_state=RANDOM_SEED, n_jobs=-1, **params)
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=RANDOM_SEED, **params)
    if family == "gradient_boosting":
        return GradientBoostingClassifier(random_state=RANDOM_SEED, **params)
    if family == "xgboost":
        from xgboost import XGBClassifier  # noqa: PLC0415
        return XGBClassifier(random_state=RANDOM_SEED, n_jobs=-1, verbosity=0, **params)
    if family == "ordinal_cumulative":
        base = dict(params)
        base_family = base.pop("base_family", "hist_gradient_boosting")
        return OrdinalCumulativeClassifier(base_family=base_family, base_params=base)
    raise ValueError(f"unknown model family: {family}")


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401, PLC0415
        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class CandidateSpec:
    """One (family, configuration, weighting) combination to evaluate."""

    family: str
    config: str
    params: dict
    weight: str
    folds: tuple[int, ...] = (0, 1, 2)
    sweep_weight: bool = True
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.family}::{self.config}::{self.weight}"


BASE_CONFIGS: dict[str, dict[str, dict]] = {
    "logistic_regression": {
        "lr_l2_C1": dict(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000),
        "lr_l2_C0.1": dict(C=0.1, penalty="l2", solver="lbfgs", max_iter=1000),
    },
    "random_forest": {
        "rf_depth12_leaf5": dict(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                  max_features=0.5, min_samples_split=10),
        "rf_depth20_leaf3": dict(n_estimators=200, max_depth=20, min_samples_leaf=3,
                                  max_features=0.7, min_samples_split=6),
    },
    "extra_trees": {
        "et_depth12_leaf5": dict(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                  max_features=0.5, min_samples_split=10),
        "et_depth20_leaf3": dict(n_estimators=200, max_depth=20, min_samples_leaf=3,
                                  max_features=0.7, min_samples_split=6),
    },
    "hist_gradient_boosting": {
        "hgb_lr05_depth6": dict(max_depth=6, learning_rate=0.05, max_iter=300,
                                min_samples_leaf=20, l2_regularization=1.0),
        "hgb_lr08_depth8": dict(max_depth=8, learning_rate=0.08, max_iter=400,
                                min_samples_leaf=10, l2_regularization=0.1),
    },
    "gradient_boosting": {
        "gb_depth4_lr08_sub5": dict(n_estimators=150, max_depth=4, learning_rate=0.08,
                                    subsample=0.5),
    },
    "ordinal_cumulative": {
        "ordinal_hgb_lr05_depth6": dict(base_family="hist_gradient_boosting", max_depth=6,
                                        learning_rate=0.05, max_iter=300,
                                        min_samples_leaf=20, l2_regularization=1.0),
    },
}

XGB_CONFIG = {"xgb_depth6_lr05": dict(max_depth=6, learning_rate=0.05, n_estimators=300,
                                      subsample=0.8, colsample_bytree=0.8,
                                      min_child_weight=5, reg_lambda=1.0)}


def candidate_specs(target: TargetSpec, full_sweep: bool = True) -> list[CandidateSpec]:
    """The experiment matrix for a target."""
    specs: list[CandidateSpec] = []
    configs = {f: dict(c) for f, c in BASE_CONFIGS.items()}
    if xgboost_available():
        configs["xgboost"] = dict(XGB_CONFIG)
    if not full_sweep:
        # cost-bounded sweep used for the secondary targets
        configs = {f: dict(sorted(c.items())[:1]) for f, c in configs.items()
                   if f in ("hist_gradient_boosting", "extra_trees", "logistic_regression")}
        return [CandidateSpec(family=f, config=cname, params=params, weight="none")
                for f, c in configs.items() for cname, params in c.items()]

    # Every candidate is scored on the primary validation window (fold 0).  The
    # extra expanding folds are reserved for the top candidates, which keeps the
    # sweep inside a sane compute budget; the protocol is recorded in the
    # artefact so the report can state exactly what was evaluated where.
    weight_probe = ("hist_gradient_boosting", "extra_trees")
    for family, cfgs in configs.items():
        for config_name, params in cfgs.items():
            weights = WEIGHT_KINDS if (family in weight_probe and config_name == sorted(cfgs)[0]) \
                else ("none",)
            for weight in weights:
                specs.append(CandidateSpec(family=family, config=config_name, params=params,
                                           weight=weight, folds=(0,)))
    specs.append(CandidateSpec(family="hist_gradient_boosting",
                               config=sorted(configs["hist_gradient_boosting"])[1],
                               params=configs["hist_gradient_boosting"][sorted(configs["hist_gradient_boosting"])[1]],
                               weight="sqrt", folds=(0,),
                               notes="second boosting configuration under the moderate weighting"))
    return specs


# ---------------------------------------------------------------------------
# Staged experiment protocol
# ---------------------------------------------------------------------------

def _imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", keep_empty_features=True)


def _as_pandas_imputer() -> SimpleImputer:
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    try:
        imp.set_output(transform="pandas")
    except Exception:  # noqa: BLE001  (older sklearn)
        pass
    return imp


def _fit_candidate(candidate: CandidateSpec, train: pd.DataFrame, val: pd.DataFrame,
                   labels: Sequence[str], features: Sequence[str]
                   ) -> tuple[Any, Any, dict, float]:
    """Fit one candidate on ``train`` and score it on ``val``.

    The imputer is fitted on ``train`` only and returned together with the model
    so callers reuse exactly the same transformation.  Returns
    ``(model, imputer, metrics, fit_seconds)``.
    """
    imp = _as_pandas_imputer()
    X_train = imp.fit_transform(train[list(features)])
    X_val = imp.transform(val[list(features)])
    y_train = train["target"].to_numpy(dtype=object)
    y_val = val["target"].to_numpy(dtype=object)

    model = make_estimator(candidate.family, candidate.params)
    if isinstance(model, OrdinalCumulativeClassifier) and not model.labels:
        model.labels = list(labels)
        model.classes_ = np.array(labels, dtype=object)
    model, weights = weighted_estimator(model, pd.Series(y_train), candidate.weight, labels)
    t0 = time.perf_counter()
    fit_estimator(model, X_train, y_train, weights)
    elapsed = time.perf_counter() - t0
    metrics = classification_metrics(y_val, model.predict(X_val), labels=labels)
    metrics["fit_seconds"] = round(elapsed, 2)
    return model, imp, metrics, elapsed


def _rank(rows: list[dict]) -> list[dict]:
    """Selection order: validation macro F1, then 3-fold mean macro F1, then accuracy."""
    return sorted(rows, key=lambda r: (-(r["primary_window_macro_f1"] or 0),
                                       -r["mean_macro_f1"],
                                       -(r["primary_window_accuracy"] or 0)))


def experiments_path(target_name: str):
    return REPORTS_ROOT / f"delay_classification_experiments_{target_name}.json"


def frozen_path(target_name: str):
    return REPORTS_ROOT / f"delay_classification_frozen_{target_name}.json"


def run_selection(target_name: str = PRIMARY_TARGET, full_sweep: bool = True) -> dict:
    """Stage 1: evaluate candidates on expanding folds.  Never sees the test period."""
    spec = TARGETS[target_name]
    df = attach_target(build_features(), spec)
    train, val, test = split_train_val_test(df)
    dev = pd.concat([train, val], ignore_index=True)
    folds = expanding_folds(dev)

    specs = candidate_specs(spec, full_sweep=full_sweep)
    logger.info("selection sweep: %d candidate configurations x up to %d folds",
                len(specs), len(folds))

    # Resumable: one candidate at a time is appended to a partial artefact, so a
    # long sweep can be run in bounded slices without losing finished work.
    partial_path = REPORTS_ROOT / f"delay_classification_experiments_{spec.name}.partial.json"
    done: dict[str, dict] = {}
    if partial_path.exists():
        try:
            done = json.loads(partial_path.read_text(encoding="utf-8")).get("done", {})
            logger.info("resuming sweep: %d/%d candidates already evaluated",
                        len(done), len(specs))
        except (OSError, json.JSONDecodeError):
            done = {}

    rows: list[dict] = []
    for i, cand in enumerate(specs, 1):
        if cand.key in done:
            rows.append(done[cand.key])
            continue
        fold_metrics = []
        started = time.perf_counter()
        for fold_idx in cand.folds:
            if fold_idx >= len(folds):
                continue
            f_train, f_val = folds[fold_idx]
            _, _, metrics, _ = _fit_candidate(cand, f_train, f_val, spec.labels, FEATURE_NAMES)
            metrics["fold"] = fold_idx
            metrics["fold_val_from"] = str(f_val["date"].min().date())
            metrics["fold_val_to"] = str(f_val["date"].max().date())
            metrics["fold_train_rows"] = int(len(f_train))
            fold_metrics.append(metrics)
        primary = next((m for m in fold_metrics if m["fold"] == 0), fold_metrics[0] if fold_metrics else None)
        row = {
            "candidate": {"family": cand.family, "config": cand.config,
                          "params": cand.params, "weight": cand.weight,
                          "folds": list(cand.folds)},
            "primary_window_macro_f1": primary["macro_f1"] if primary else None,
            "primary_window_accuracy": primary["accuracy"] if primary else None,
            "primary_window_balanced_accuracy": primary["balanced_accuracy"] if primary else None,
            "primary_window_weighted_f1": primary["weighted_f1"] if primary else None,
            "mean_macro_f1": round(float(np.mean([m["macro_f1"] for m in fold_metrics])), 4),
            "mean_accuracy": round(float(np.mean([m["accuracy"] for m in fold_metrics])), 4),
            "fold_metrics": fold_metrics,
            "elapsed_seconds": round(time.perf_counter() - started, 1),
        }
        rows.append(row)
        done[cand.key] = row
        partial_path.write_text(json.dumps({"done": done}, indent=2, default=str),
                                encoding="utf-8")
        logger.info("[%d/%d] %s -> fold0 macroF1=%.4f acc=%.4f (3-fold mean %.4f) %.0fs",
                    i, len(specs), row["candidate"]["family"] + "::" + row["candidate"]["config"]
                    + "::" + row["candidate"]["weight"],
                    row["primary_window_macro_f1"] or 0, row["primary_window_accuracy"] or 0,
                    row["mean_macro_f1"], row["elapsed_seconds"])

    # Robustness pass: the top candidates by validation macro F1 are re-scored on
    # the two older expanding folds, which costs two more fits per candidate and
    # keeps the selection evidence from resting on a single window.
    ranked = _rank(rows)
    for row in ranked[:ROBUSTNESS_TOP_N]:
        cand = CandidateSpec(**row["candidate"])
        missing = [f for f in (0, 1, 2) if f not in cand.folds]
        if not missing or row.get("robustness_done"):
            continue
        for fold_idx in missing:
            if fold_idx >= len(folds):
                continue
            f_train, f_val = folds[fold_idx]
            _, _, metrics, _ = _fit_candidate(cand, f_train, f_val, spec.labels, FEATURE_NAMES)
            metrics["fold"] = fold_idx
            metrics["fold_val_from"] = str(f_val["date"].min().date())
            metrics["fold_val_to"] = str(f_val["date"].max().date())
            metrics["fold_train_rows"] = int(len(f_train))
            row["fold_metrics"].append(metrics)
            row["fold_metrics"].sort(key=lambda m: m["fold"])
            row["mean_macro_f1"] = round(float(np.mean([m["macro_f1"] for m in row["fold_metrics"]])), 4)
            row["mean_accuracy"] = round(float(np.mean([m["accuracy"] for m in row["fold_metrics"]])), 4)
        row["robustness_done"] = True
        done[cand.key] = row
        partial_path.write_text(json.dumps({"done": done}, indent=2, default=str), encoding="utf-8")
        logger.info("robustness folds for %s -> 3-fold mean macroF1=%.4f", cand.key, row["mean_macro_f1"])

    ranked = _rank(rows)
    winner = ranked[0]

    artifact = {
        "target": spec.name,
        "target_definition": spec.definition,
        "target_source": spec.source,
        "labels": list(spec.labels),
        "thresholds": list(spec.thresholds),
        "selection_metric": "macro F1 on the validation window (2024-11-06..2024-12-03); tie-break 3-fold mean macro F1, then accuracy",
        "test_period_touched": False,
        "folds": [{"index": i, "val_from": str(v["date"].min().date()),
                   "val_to": str(v["date"].max().date()),
                   "train_rows_capped": int(len(t)), "val_rows": int(len(v))}
                  for i, (t, v) in enumerate(folds)],
        "fold_train_cap": FOLD_TRAIN_CAP,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows_reserved": int(len(test)),
        "n_features": len(FEATURE_NAMES),
        "features": list(FEATURE_NAMES),
        "candidates": rows,
        "selected": winner,
        "environment": environment_info(df),
    }
    artifact["artifact_digest"] = artifact_digest(artifact)
    experiments_path(spec.name).parent.mkdir(parents=True, exist_ok=True)
    experiments_path(spec.name).write_text(
        json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    logger.info("wrote %s (selected %s::%s::%s)", experiments_path(spec.name),
                winner["candidate"]["family"], winner["candidate"]["config"],
                winner["candidate"]["weight"])
    return artifact


def artifact_digest(payload: dict) -> str:
    import hashlib
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def environment_info(df: pd.DataFrame) -> dict:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": "available" if xgboost_available() else "not installed",
        "random_seed": RANDOM_SEED,
        "trip_facts_rows": int(len(df)),
        "data_fingerprint": _fingerprint(),
    }


def load_experiments(target_name: str = PRIMARY_TARGET) -> dict:
    path = experiments_path(target_name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m src.ml.delay_classification --stage selection"
            f" --target {target_name}` first")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Decision-rule optimisation (validation only)
# ---------------------------------------------------------------------------

def aligned_probabilities(model: Any, X, labels: Sequence[str]) -> np.ndarray | None:
    """``predict_proba`` re-ordered to the target's label order.

    Estimators store their own class order in ``classes_`` (HistGradientBoosting
    sorts string labels lexicographically, for example), so the columns can
    never be assumed to match the target's ordering - they are always mapped
    explicitly here.  Getting this wrong would silently shuffle probabilities
    between classes.
    """
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = np.asarray(model.predict_proba(X))
    except Exception:  # noqa: BLE001
        return None
    classes = [str(c) for c in getattr(model, "classes_", [])]
    if len(classes) != proba.shape[1]:
        if proba.shape[1] != len(labels):
            return None
        classes = [str(lbl) for lbl in labels]
    out = np.zeros((proba.shape[0], len(labels)))
    matched = False
    for j, cls in enumerate(classes):
        if cls in labels:
            out[:, labels.index(cls)] = proba[:, j]
            matched = True
    if not matched and proba.shape[1] == len(labels):
        return proba  # positional fallback for exotic estimators
    return out


def _macro_f1(labels: Sequence[str], y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def optimise_class_multipliers(proba: np.ndarray, y_true: np.ndarray,
                               labels: Sequence[str],
                               grid: Sequence[float] | None = None,
                               rounds: int = 3) -> dict:
    """Coordinate ascent on per-class probability multipliers (validation only).

    Multiplying the class probabilities by a per-class factor is a soft,
    monotone way of moving the decision boundary - for a binary target it is
    exactly a threshold change.  The search runs on the **validation** split and
    the resulting multipliers are frozen before the test period is scored.
    """
    grid = list(grid or np.round(np.arange(0.5, 2.01, 0.1), 2))
    multipliers = np.ones(len(labels), dtype=float)
    baseline = _macro_f1(labels, y_true, np.array(labels)[proba.argmax(axis=1)])
    best_score = baseline
    for _ in range(rounds):
        improved = False
        for c in range(len(labels)):
            current = multipliers[c]
            local_best, local_value = best_score, current
            for value in grid:
                trial = multipliers.copy()
                trial[c] = value
                pred = np.array(labels)[(proba * trial).argmax(axis=1)]
                score = _macro_f1(labels, y_true, pred)
                if score > local_best + 1e-6:
                    local_best, local_value = score, value
            if local_value != current:
                multipliers[c] = local_value
                best_score = local_best
                improved = True
        if not improved:
            break
    return {
        "multipliers": {lbl: round(float(multipliers[i]), 3) for i, lbl in enumerate(labels)},
        "validation_macro_f1_without": round(baseline, 4),
        "validation_macro_f1_with": round(best_score, 4),
        "search_grid": [float(g) for g in grid],
        "method": "coordinate ascent on per-class probability multipliers, fitted on the validation window only",
    }


def apply_multipliers(proba: np.ndarray, labels: Sequence[str],
                      multipliers: dict[str, float]) -> np.ndarray:
    factors = np.array([float(multipliers.get(lbl, 1.0)) for lbl in labels])
    return np.array(labels)[(np.asarray(proba) * factors).argmax(axis=1)]


# ---------------------------------------------------------------------------
# Stage 2: freeze and evaluate the untouched test period once
# ---------------------------------------------------------------------------

def model_path(target_name: str):
    return MODELS_PYTHON / f"delay_classification_{target_name}_frozen.joblib"


def metrics_path(target_name: str):
    return MODELS_PYTHON / f"delay_classification_{target_name}_metrics.json"


def predictions_path(target_name: str):
    return MODELS_PYTHON / f"delay_classification_{target_name}_predictions.parquet"


def run_final(target_name: str = PRIMARY_TARGET, force: bool = False,
              permutation_sample: int = 5000) -> dict:
    """Stage 2: refit the frozen configuration, then touch the test period once."""
    import joblib

    spec = TARGETS[target_name]
    experiments = load_experiments(target_name)
    if experiments.get("test_period_touched"):
        raise RuntimeError("the experiments artefact claims to have seen the test period")
    candidate = CandidateSpec(**{k: v for k, v in experiments["selected"]["candidate"].items()
                                if k in CandidateSpec.__dataclass_fields__})

    previous = None
    if frozen_path(target_name).exists():
        try:
            previous = json.loads(frozen_path(target_name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
    if previous and previous.get("test_evaluated") and not force:
        raise RuntimeError(
            f"{frozen_path(target_name)} already holds a test evaluation. The test period is "
            "evaluated once; pass force=True only to reproduce the same frozen configuration.")

    df = attach_target(build_features(), spec)
    train, val, test = split_train_val_test(df)

    # 1) validation run: fit on the whole pre-validation period (uncapped)
    val_model, val_imputer, validation, val_seconds = _fit_candidate(
        candidate, train, val, spec.labels, FEATURE_NAMES)
    X_val = val_imputer.transform(val[FEATURE_NAMES])
    y_val = val["target"].to_numpy(dtype=object)
    val_proba = aligned_probabilities(val_model, X_val, spec.labels)
    validation["fit_seconds"] = round(val_seconds, 2)

    decision_rule = {"multipliers": {lbl: 1.0 for lbl in spec.labels},
                     "validation_macro_f1_without": validation["macro_f1"],
                     "validation_macro_f1_with": validation["macro_f1"],
                     "method": "no adjustment could be fitted (no suitable probabilities)"}
    if val_proba is not None:
        decision_rule = optimise_class_multipliers(val_proba, y_val, spec.labels)
        tuned_pred = apply_multipliers(val_proba, spec.labels, decision_rule["multipliers"])
        validation_after = classification_metrics(y_val, tuned_pred, labels=list(spec.labels))
        decision_rule["validation_with_adjustment"] = validation_after
        # keep the rule only when it genuinely helps on validation
        if validation_after["macro_f1"] <= validation["macro_f1"] + 1e-6:
            decision_rule["multipliers"] = {lbl: 1.0 for lbl in spec.labels}
            decision_rule["applied"] = False
            decision_rule["reason"] = "adjustment did not improve validation macro F1; multipliers reset to 1.0"
        else:
            decision_rule["applied"] = True
    if len(spec.labels) == 2 and decision_rule.get("applied"):
        m0, m1 = (decision_rule["multipliers"][spec.labels[0]],
                  decision_rule["multipliers"][spec.labels[1]])
        decision_rule["equivalent_probability_threshold"] = round(1.0 / (1.0 + (m1 / m0)), 4)

    # 2) frozen fit on train + validation, then the single test evaluation
    pool = pd.concat([train, val], ignore_index=True)
    imp_final = _as_pandas_imputer()
    X_pool = imp_final.fit_transform(pool[FEATURE_NAMES])
    X_test = imp_final.transform(test[FEATURE_NAMES])
    y_pool = pool["target"].to_numpy(dtype=object)
    y_test = test["target"].to_numpy(dtype=object)
    final_model, final_weights = weighted_estimator(
        make_estimator(candidate.family, candidate.params), pd.Series(y_pool),
        candidate.weight, spec.labels)
    t0 = time.perf_counter()
    fit_estimator(final_model, X_pool, y_pool, final_weights)
    final_seconds = time.perf_counter() - t0

    test_pred = final_model.predict(X_test)
    test_proba = aligned_probabilities(final_model, X_test, spec.labels)
    test_metrics = classification_metrics(y_test, test_pred, labels=list(spec.labels))
    test_metrics["fit_seconds"] = round(final_seconds, 2)

    adjusted_pred = (apply_multipliers(test_proba, spec.labels, decision_rule["multipliers"])
                     if test_proba is not None else None)
    test_adjusted = (classification_metrics(y_test, adjusted_pred, labels=list(spec.labels))
                     if adjusted_pred is not None else None)

    importance = _permutation_importance(final_model, X_val, y_val, spec.labels,
                                          permutation_sample)

    # 3) persist
    MODELS_PYTHON.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "imputer": imp_final,
                 "features": list(FEATURE_NAMES), "labels": list(spec.labels),
                 "target": spec.name, "candidate": asdict(candidate),
                 "decision_rule": decision_rule}, model_path(target_name))

    predictions = test[["trip_id", "route_id", "date", "is_peak", "time_band",
                        "departure_delay_min"]].copy()
    predictions["actual_target"] = y_test
    predictions["predicted_target"] = test_pred
    if adjusted_pred is not None:
        predictions["predicted_target_adjusted"] = adjusted_pred
    if test_proba is not None:
        for i, lbl in enumerate(spec.labels):
            predictions[f"prob_{lbl.replace(' ', '_')}"] = test_proba[:, i]
    predictions = predictions.sort_values(["route_id", "date", "time_band"]).reset_index(drop=True)
    saved_predictions = save_parquet(predictions, f"delay_classification_{target_name}_predictions")

    frozen = {
        "target": spec.name,
        "labels": list(spec.labels),
        "thresholds": list(spec.thresholds),
        "target_definition": spec.definition,
        "target_source": spec.source,
        "frozen_at": pd.Timestamp.utcnow().isoformat(),
        "test_evaluated": True,
        "test_evaluation_count": int((previous or {}).get("test_evaluation_count", 0)) + 1,
        "reproduction_run": bool(previous and force),
        "candidate": asdict(candidate),
        "selection": experiments["selected"],
        "experiments_artifact": {"path": str(experiments_path(target_name)),
                                 "digest": experiments.get("artifact_digest")},
        "split": {
            "train": {"rows": int(len(train)), "from": str(train["date"].min().date()),
                      "to": str(train["date"].max().date())},
            "validation": {"rows": int(len(val)), "from": str(val["date"].min().date()),
                           "to": str(val["date"].max().date())},
            "test": {"rows": int(len(test)), "from": str(test["date"].min().date()),
                     "to": str(test["date"].max().date())},
            "final_fit_rows": int(len(pool)),
        },
        "class_distribution": {
            split_name: {lbl: int((frame["target"] == lbl).sum()) for lbl in spec.labels}
            for split_name, frame in (("train", train), ("validation", val), ("test", test))
        },
        "validation": validation,
        "test": test_metrics,
        "test_with_decision_adjustment": test_adjusted,
        "decision_rule": decision_rule,
        "feature_importance": importance,
        "features": list(FEATURE_NAMES),
        "feature_catalogue": [asdict(s) for s in FEATURE_SPECS],
        "forbidden_columns": list(FORBIDDEN_COLUMNS),
        "environment": environment_info(df),
        "predictions_path": str(saved_predictions),
        "model_path": str(model_path(target_name)),
    }
    frozen["overfitting_gaps"] = {
        "test_minus_validation_accuracy": round(test_metrics["accuracy"] - validation["accuracy"], 4),
        "test_minus_validation_macro_f1": round(test_metrics["macro_f1"] - validation["macro_f1"], 4),
        "test_minus_validation_balanced_accuracy": round(
            test_metrics["balanced_accuracy"] - validation["balanced_accuracy"], 4),
    }
    frozen["metric_consistency_issues"] = {
        "test": check_metric_consistency(test_metrics),
        "validation": check_metric_consistency(validation),
    }
    frozen["artifact_digest"] = artifact_digest(frozen)

    frozen_path(target_name).write_text(json.dumps(frozen, indent=2, default=str),
                                         encoding="utf-8")
    metrics_path(target_name).write_text(json.dumps(frozen, indent=2, default=str),
                                         encoding="utf-8")
    logger.info("frozen %s: test acc=%.4f macroF1=%.4f", target_name,
                test_metrics["accuracy"], test_metrics["macro_f1"])
    return frozen


def _permutation_importance(model: Any, X_val: pd.DataFrame, y_val: np.ndarray,
                            labels: Sequence[str], sample: int, top: int = 20) -> list[dict]:
    try:
        from sklearn.inspection import permutation_importance
        n = min(sample, len(X_val))
        idx = np.random.default_rng(RANDOM_SEED).choice(len(X_val), n, replace=False)
        result = permutation_importance(model, X_val.iloc[idx], y_val[idx],
                                        scoring="f1_macro", n_repeats=5,
                                        random_state=RANDOM_SEED, n_jobs=-1)
        order = np.argsort(-result.importances_mean)[:top]
        return [{"feature": FEATURE_NAMES[i],
                 "importance": round(float(result.importances_mean[i]), 5),
                 "std": round(float(result.importances_std[i]), 5)}
                for i in order]
    except Exception as exc:  # noqa: BLE001
        logger.warning("permutation importance unavailable: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public API (dashboard + evaluation helpers)
# ---------------------------------------------------------------------------

def load_frozen(target_name: str = PRIMARY_TARGET) -> dict:
    path = frozen_path(target_name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m src.ml.delay_classification --stage all"
            f" --target {target_name}` first")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_delay_classifier(target_name: str = PRIMARY_TARGET) -> dict:
    """Recompute metrics from the stored test predictions (single source of truth)."""
    path = predictions_path(target_name)
    if not path.exists():
        return {"error": f"predictions not found: {path}"}
    preds = pd.read_parquet(path)
    labels = list(TARGETS[target_name].labels)
    out = classification_metrics(preds["actual_target"].to_numpy(dtype=object),
                                 preds["predicted_target"].to_numpy(dtype=object),
                                 labels=labels)
    out["cases"] = int(len(preds))
    if "predicted_target_adjusted" in preds.columns:
        out["adjusted"] = classification_metrics(
            preds["actual_target"].to_numpy(dtype=object),
            preds["predicted_target_adjusted"].to_numpy(dtype=object), labels=labels)
    return out


def predict_delay_severity(route_ids: list[str] | None = None,
                           date: str | None = None,
                           target_name: str = PRIMARY_TARGET) -> pd.DataFrame:
    """Predict the delay category for upcoming trips (latest trip per route/band)."""
    import joblib

    path = model_path(target_name)
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}. Run the final stage first.")
    bundle = joblib.load(path)
    spec = TARGETS[target_name]

    df = attach_target(build_features(), spec)
    if route_ids:
        df = df[df["route_id"].isin(route_ids)]
    if date:
        df = df[df["date"] == pd.Timestamp(date)]
    if df.empty:
        return pd.DataFrame(columns=["trip_id", "route_id", "date", "time_band",
                                     "actual_target", "predicted_target"])

    latest = (df.sort_values(["date", "route_id", "time_band", "scheduled_departure"])
              .groupby(["route_id", "date", "time_band"], sort=False).tail(1).copy())
    X = bundle["imputer"].transform(latest[bundle["features"]])
    pred = bundle["model"].predict(X)
    proba = aligned_probabilities(bundle["model"], X, bundle["labels"])

    out = latest[["trip_id", "route_id", "date", "is_peak", "time_band",
                  "departure_delay_min", "target"]].copy()
    out = out.rename(columns={"target": "actual_target"})
    out["predicted_target"] = pred
    if proba is not None:
        for i, lbl in enumerate(bundle["labels"]):
            out[f"prob_{lbl.replace(' ', '_')}"] = proba[:, i]
    rule = bundle.get("decision_rule", {})
    if proba is not None and rule.get("multipliers"):
        out["predicted_target_adjusted"] = apply_multipliers(proba, bundle["labels"],
                                                             rule["multipliers"])
    return out.sort_values(["route_id", "date", "time_band"]).reset_index(drop=True)


def load_saved_model_results(target_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Read persisted metrics for every target that has been trained."""
    names = list(target_names or TARGETS.keys())
    out: dict[str, Any] = {}
    for name in names:
        path = metrics_path(name)
        if not path.exists():
            continue
        try:
            out[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001
            logger.warning("could not read metrics for %s: %s", name, exc)
    return out


def retrain_frozen(target_name: str = PRIMARY_TARGET) -> dict:
    """Refit and re-evaluate the already-frozen configuration (dashboard button)."""
    return run_final(target_name, force=True)


def build_all_classifiers(target_name: str = PRIMARY_TARGET, full_sweep: bool = False) -> dict:
    """Run the staged protocol for one target (kept for the dashboard)."""
    experiments = run_selection(target_name, full_sweep=full_sweep)
    frozen = run_final(target_name, force=True)
    return {"selection": {"selected": experiments["selected"]["candidate"],
                          "candidates": len(experiments["candidates"])},
            "frozen": frozen}


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m src.ml.delay_classification")
    parser.add_argument("--stage", default="all",
                        choices=["selection", "final", "report", "all", "status"])
    parser.add_argument("--target", default=PRIMARY_TARGET, choices=list(TARGETS))
    parser.add_argument("--force", action="store_true",
                        help="re-evaluate an already frozen configuration (reproduction only)")
    parser.add_argument("--light-sweep", action="store_true",
                        help="evaluate one configuration per family instead of the full matrix")
    args = parser.parse_args(argv)

    if args.stage in ("selection", "all"):
        result = run_selection(args.target, full_sweep=not args.light_sweep)
        print(json.dumps({"selected": result["selected"]["candidate"],
                          "candidates": len(result["candidates"])}, indent=2, default=str))
    if args.stage in ("final", "all"):
        frozen = run_final(args.target, force=args.force)
        print(json.dumps({"test": frozen["test"],
                          "validation_accuracy": frozen["validation"]["accuracy"],
                          "overfitting_gaps": frozen["overfitting_gaps"]}, indent=2, default=str))
    if args.stage == "status":
        run_selection(TARGET_STATUS.name, full_sweep=False)
        frozen = run_final(TARGET_STATUS.name, force=args.force)
        print(json.dumps({"target": TARGET_STATUS.name, "test": frozen["test"]},
                         indent=2, default=str))
    if args.stage in ("report", "all"):
        from src.ml.delay_classification_report import write_report  # noqa: PLC0415
        print(write_report())


if __name__ == "__main__":
    main()

