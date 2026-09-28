"""Demand-forecasting data assembly and evaluation helpers.

Provides the single share of training rows, the feature/target split and the
metric functions used by BOTH the Spark MLlib and the scikit-learn demand
models so the two engines are compared on identical data and identical math.

Data: the analytic ``demand_features`` frame (per route x date x time band),
enriched with context features that are legitimately known at prediction time
(see :func:`add_context_features`).

Split: the most recent ``FORECAST_HORIZON_DAYS`` of data are the unseen test
window (models never train on rows after ``cutoff_test``); the 28 days before
that form a validation window used for all tuning and model selection. Lag and
rolling features only ever reference rows strictly before the row's own date,
so the evaluation stays a genuine out-of-time forecast. The scikit-learn
engine refits its selected configuration on train+validation before the single
final test evaluation; the Spark engine reports the train-only fit.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from config import settings
from src.analytics.core import read_features

TARGET = "boardings"

NUMERIC_FEATURES = [
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "day_type_code", "weather_code", "season_code", "time_band_code",
    "prev_week_boardings", "rolling_7d_boardings",
]

# Extra context features added by :func:`add_context_features`. Kept separate
# so existing consumers of ``NUMERIC_FEATURES`` (the delay/occupancy modules
# import the name) keep their original behaviour; the demand engines use
# ``MODEL_FEATURES``.
CONTEXT_FEATURES = [
    "trips_scheduled", "log_trips_scheduled", "event_multiplier",
    "route_category_code", "route_stops_count", "route_length_km",
    "lag1_boardings", "lag2_boardings", "lag7_boardings", "lag14_boardings",
    "roll7_mean_boardings", "roll7_median_boardings", "roll7_std_boardings",
    "roll28_mean_boardings",
]

MODEL_FEATURES = NUMERIC_FEATURES + CONTEXT_FEATURES

_DEMAND_KEY_COLS = ["route_id", "date", "time_band"]
_MAX_LAG_DAYS = 28


def demand_data() -> pd.DataFrame:
    frame = read_features("demand_features")[
        ["route_id", "date", "time_band"]
        + NUMERIC_FEATURES + [TARGET]]
    return frame.dropna(subset=[TARGET] + NUMERIC_FEATURES)


def _time_band(hour: int) -> str:
    """Same banding as the integrator so ``trips_scheduled`` merges cleanly."""
    if hour < 5:
        return "night"
    if hour < 10:
        return "morning peak"
    if hour < 16:
        return "midday"
    if hour < 20:
        return "evening peak"
    return "night"


def add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach prediction-time-safe context columns to ``demand_data()`` rows.

    Every column here is knowable *before* the day being predicted:

    * ``trips_scheduled`` / ``log_trips_scheduled`` — how many trips the
      timetable puts on this route/band/date. The schedule for a day is fixed
      before that day runs, so this is service-supply information, not an
      outcome of the demand we are predicting.
    * ``event_multiplier`` — the strongest demand_multiplier among city events
      touching the route's zones on that date (1.0 when none); the events
      calendar is published ahead of time.
    * ``route_category_code`` / ``route_stops_count`` / ``route_length_km`` —
      static route attributes.
    * ``lag*_boardings`` / ``roll*_boardings`` — historical demand on the same
      (route, time_band) series, taken from the *calendar date* exactly 1/2/7/
      14 days earlier and from the 7/28 calendar days before the row. The
      pre-existing row-count based ``prev_week_boardings`` /
      ``rolling_7d_boardings`` go stale on the ~10% of rows that follow a gap
      in a route-band series; these are gap-proof. Only strictly-past dates
      are ever referenced; missing history is NaN (imputed by the engines from
      training statistics only).
    """
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])

    # --- scheduled trips (timetable supply for the day) -------------------- #
    trips = pd.read_parquet(settings.PROJECT_ROOT / "data" / "raw" / "trips.parquet",
                            columns=["trip_id", "route_id", "date", "hour"])
    trips["date"] = pd.to_datetime(trips["date"])
    trips["time_band"] = trips["hour"].map(_time_band)
    served = (trips.groupby(["route_id", "date", "time_band"], as_index=False)
              .size().rename(columns={"size": "trips_scheduled"}))
    out = out.merge(served, on=_DEMAND_KEY_COLS, how="left")
    out["trips_scheduled"] = out["trips_scheduled"].fillna(0).astype(int)
    out["log_trips_scheduled"] = np.log1p(out["trips_scheduled"].astype(float))

    # --- city events touching the route's zones ---------------------------- #
    routes = pd.read_parquet(settings.PROJECT_ROOT / "data" / "raw" / "routes.parquet")
    stops = pd.read_parquet(settings.PROJECT_ROOT / "data" / "raw" / "stops.parquet",
                            columns=["stop_id", "zone"])
    rs = pd.read_parquet(settings.PROJECT_ROOT / "data" / "raw" / "route_stops.parquet",
                         columns=["route_id", "stop_id"])
    route_zones = rs.merge(stops, on="stop_id")[["route_id", "zone"]].drop_duplicates()
    events = pd.read_parquet(settings.PROJECT_ROOT / "data" / "raw" / "events.parquet",
                             columns=["date", "zone", "demand_multiplier"])
    events["date"] = pd.to_datetime(events["date"])
    per_route_day = (route_zones.merge(events, on="zone")
                     .groupby(["route_id", "date"])["demand_multiplier"]
                     .max().rename("event_multiplier").reset_index())
    out = out.merge(per_route_day, on=["route_id", "date"], how="left")
    out["event_multiplier"] = out["event_multiplier"].fillna(1.0)

    # --- static route attributes ------------------------------------------- #
    route_static = routes.rename(columns={
        "stops_count": "route_stops_count",
        "length_km": "route_length_km",
    })[["route_id", "category", "route_stops_count", "route_length_km"]]
    category_codes = {"core": 0, "feeder": 1, "express": 2}
    route_static["route_category_code"] = route_static["category"].map(
        category_codes).fillna(-1).astype(int)
    out = out.merge(route_static.drop(columns=["category"]),
                    on="route_id", how="left")

    # --- calendar-date-aware lags and rolling stats ------------------------- #
    out = out.sort_values(_DEMAND_KEY_COLS, kind="mergesort").reset_index(drop=True)
    key_route = out["route_id"].astype(str).to_numpy()
    key_band = out["time_band"].astype(str).to_numpy()
    # Dates as int64 (ns since epoch) so lookup keys hash consistently on both
    # sides (pd.Timestamp and np.datetime64 do not always hash alike).
    key_date = out["date"].astype("int64").to_numpy()
    vals = out["boardings"].to_numpy(dtype=float)
    position = pd.Series(
        np.arange(len(out)),
        index=pd.MultiIndex.from_arrays([key_route, key_band, key_date]))
    lookup = position.to_dict()

    def past_positions(days: int) -> np.ndarray:
        """Row positions holding the value of ``days`` calendar days earlier."""
        query_date = (out["date"] - pd.Timedelta(days=days)).astype("int64").to_numpy()
        return np.array([lookup.get((r, b, d), np.nan)
                         for r, b, d in zip(key_route, key_band, query_date)],
                        dtype=float)

    past = {d: past_positions(d) for d in range(1, _MAX_LAG_DAYS + 1)}

    def gather(positions: np.ndarray) -> np.ndarray:
        ok = ~np.isnan(positions)
        result = np.full(len(out), np.nan)
        result[ok] = vals[positions[ok].astype(int)]
        return result

    for days in (1, 2, 7, 14):
        out[f"lag{days}_boardings"] = gather(past[days])

    def rolling(days: int, fn: str) -> np.ndarray:
        window = np.vstack([gather(past[d]) for d in range(1, days + 1)])
        with warnings.catch_warnings():
            # Rows at the very start of a series have no history at all; their
            # all-NaN windows are a legitimate "no information" NaN, not an
            # operation to warn about.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if fn == "mean":
                return np.nanmean(window, axis=0)
            if fn == "median":
                return np.nanmedian(window, axis=0)
            return np.nanstd(window, axis=0)

    out["roll7_mean_boardings"] = rolling(7, "mean")
    out["roll7_median_boardings"] = rolling(7, "median")
    out["roll7_std_boardings"] = rolling(7, "std")
    out["roll28_mean_boardings"] = rolling(28, "mean")
    return out


def split_window(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Future-unseen train/test split using ``FORECAST_HORIZON_DAYS``."""
    cutoff = pd.Timestamp(frame["date"].max().normalize()) \
        - pd.Timedelta(days=settings.FORECAST_HORIZON_DAYS)
    train = frame[frame["date"] <= cutoff]
    test = frame[frame["date"] > cutoff]
    return train, test


def split_train_val_test(
        frame: pd.DataFrame,
        val_days: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leakage-safe chronological split: train | validation | test.

    The test window (the most recent ``FORECAST_HORIZON_DAYS``) is identical to
    :func:`split_window` so metrics stay comparable with earlier runs; the
    validation window sits directly before it and is the ONLY window used for
    model selection and hyperparameter tuning.
    """
    cutoff_test = pd.Timestamp(frame["date"].max().normalize()) \
        - pd.Timedelta(days=settings.FORECAST_HORIZON_DAYS)
    cutoff_val = cutoff_test - pd.Timedelta(days=val_days)
    train = frame[frame["date"] <= cutoff_val]
    val = frame[(frame["date"] > cutoff_val) & (frame["date"] <= cutoff_test)]
    test = frame[frame["date"] > cutoff_test]
    return train, val, test


def unseen_cases(test: pd.DataFrame, n: int = 100, seed: int = 7) -> pd.DataFrame:
    """Deterministic sample of ``n`` unseen (route, date, time band) cases.

    One case is drawn from each route first (so every route appears in the
    comparison table when possible), then the remaining slots are filled from
    the rest of the unseen window.
    """
    if not len(test):
        return test.head(0)
    n_routes = min(test["route_id"].nunique(), n)
    per_route = (test.groupby("route_id", group_keys=False)
                 .apply(lambda g: g.sample(1, random_state=seed)))
    rest = test.drop(per_route.index)
    fill = n - len(per_route)
    if fill > 0 and len(rest):
        extras = rest.sample(min(fill, len(rest)), random_state=seed)
        per_route = pd.concat([per_route, extras])
    return per_route.head(n).reset_index(drop=True)


def normalise_date_column(series: pd.Series) -> pd.Series:
    """Return ``series`` as ``datetime64[ns]`` regardless of how it was stored.

    Spark lands timestamp columns as integer microseconds since the epoch, while
    the pandas pipeline produces ``datetime64[ns]``. Comparing or merging the two
    engines' prediction frames therefore raised
    ``merge on datetime64[ns] and int64``. Normalising in one place keeps every
    persisted prediction artifact on the same key dtype.
    """
    if pd.api.types.is_integer_dtype(series):
        return pd.to_datetime(series, unit="us", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    resid = actual - pred
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mape = float(np.mean(np.abs(resid) / np.maximum(np.abs(actual), 1.0)) * 100.0)
    r2 = 1.0 - float(np.sum(resid ** 2)) / float(
        np.sum((actual - actual.mean()) ** 2) + 1e-12)
    return {"mae": round(mae, 3), "rmse": round(rmse, 3),
            "mape_pct": round(mape, 3), "r2": round(r2, 4),
            "n": int(len(actual))}


def train_set(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature-matrix columns (context features included when present)."""
    available = [c for c in MODEL_FEATURES if c in frame.columns]
    return frame[[TARGET] + available].copy()


def save_parquet(df: pd.DataFrame, name: str) -> Path:
    import pandas as pd  # noqa: F401

    from src.paths import MODELS_PYTHON

    MODELS_PYTHON.mkdir(parents=True, exist_ok=True)
    path = MODELS_PYTHON / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path
