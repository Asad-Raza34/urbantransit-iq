"""Delay prediction model for UrbanTransit IQ.

Predicts trip-level departure delay in minutes using historical trip data.
Features include route, stop, time, historical delay patterns, headway,
occupancy, and demand indicators.
"""

import json
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import settings
from src.analytics.core import read_processed
# One shared ``save_parquet``. This module previously carried its own copy that
# wrote to ``data/analytics`` while ``evaluate_delay_model`` read from
# ``models/python``, so evaluation could never find the predictions. The sibling
# occupancy module already imports the shared helper.
from src.ml.demand import save_parquet
from src.log import get_logger
from src.paths import MODELS_PYTHON

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


def metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    """Compute regression metrics."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    resid = actual - pred
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mape = float(np.mean(np.abs(resid) / np.maximum(np.abs(actual), 1.0)) * 100.0)
    r2 = 1.0 - float(np.sum(resid ** 2)) / float(
        np.sum((actual - actual.mean()) ** 2) + 1e-12)
    return {"mae": round(mae, 3), "rmse": round(rmse, 3),
            "mape_pct": round(mape, 3), "r2": round(r2, 4),
            "n": int(len(actual))}


@dataclass
class DelayModelMetrics:
    """Metrics for delay prediction model."""
    mae: float
    rmse: float
    mape_pct: float
    r2: float
    n: int


# Target: route-level average departure delay in minutes (aggregated to route/date/time_band)
DELAY_TARGET = "route_avg_delay"

MODEL_TYPES = ["random_forest", "gradient_boosting", "linear"]

# Validation window size (days) for model selection
VAL_DAYS = 28

# Features for delay prediction (expanded with additional lag/rolling features)
DELAY_FEATURES = [
    # Temporal features
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "day_type_code", "weather_code", "season_code", "time_band_code",
    "is_peak_int", "is_weekend_int",
    # Route features
    "route_category_code", "route_length_km",
    # Historical lag features (computed per route)
    "route_avg_delay_lag1", "route_avg_delay_lag7", "route_avg_delay_lag14",
    "route_avg_delay_rolling7", "route_avg_delay_rolling14", "route_avg_delay_rolling28",
    "route_p90_delay_lag1", "route_p90_delay_lag7", "route_p90_delay_lag14",
    "route_p90_delay_rolling7", "route_p90_delay_rolling14", "route_p90_delay_rolling28",
    "route_on_time_rate_lag1", "route_on_time_rate_lag7", "route_on_time_rate_lag14",
    "route_on_time_rate_rolling7", "route_on_time_rate_rolling14", "route_on_time_rate_rolling28",
    # Headway features
    "route_headway_cv",
    # Occupancy/demand features
    "avg_occupancy_avg_pct", "avg_occupancy_max_pct", "avg_boardings", "trip_count",
    # Bunching indicators
    "bunching_flag_route",
]


def _load_delay_features() -> pd.DataFrame:
    """Load and prepare delay prediction features from trip_facts, aggregated to route/date/time_band."""
    trip_facts = read_processed("trip_facts")
    
    df = trip_facts.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route_id", "date", "hour"]).reset_index(drop=True)
    
    # Encode categorical for aggregation
    DAY_TYPE_CODE = {"weekday": 0, "weekend": 1, "holiday": 2}
    WEATHER_CODE = {"clear": 0, "rain": 1, "snow": 2, "storm": 3}
    SEASON_CODE = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
    TIME_BAND_CODE = {"night": 0, "morning peak": 1, "midday": 2, "evening peak": 3}
    ROUTE_CATEGORY_CODE = {"core": 0, "feeder": 1, "express": 2}
    
    # Aggregate to route/date/time_band level
    daily_route_band = df.groupby(["route_id", "date", "time_band"]).agg(
        route_avg_delay=("departure_delay_min", "mean"),
        route_on_time_rate=("on_time", "mean"),
        route_headway_mean=("scheduled_headway_min", "mean"),
        route_headway_std=("scheduled_headway_min", "std"),
        avg_occupancy_avg_pct=("occupancy_avg_pct", "mean"),
        avg_occupancy_max_pct=("occupancy_max_pct", "mean"),
        avg_boardings=("boardings", "sum"),
        trip_count=("trip_id", "count"),
        distance_km=("distance_km", "mean"),
        route_length_km=("route_length_km", "first"),
        route_category=("route_category", "first"),
        day_type=("day_type", "first"),
        weather=("weather", "first"),
        season=("season", "first"),
        weekday=("weekday", "first"),
        month=("month", "first"),
        is_peak=("is_peak", "first"),
        is_weekend=("is_weekend", "first"),
    ).reset_index()
    
    # Compute p90 delay efficiently
    p90 = df.groupby(["route_id", "date", "time_band"])["departure_delay_min"].quantile(0.9).reset_index(name="route_p90_delay")
    daily_route_band = daily_route_band.merge(p90, on=["route_id", "date", "time_band"])
    
    # Compute headway CV
    daily_route_band["route_headway_cv"] = (
        daily_route_band["route_headway_std"] / daily_route_band["route_headway_mean"].replace(0, np.nan)
    ).fillna(0)
    
    # Re-encode categorical for aggregated data
    daily_route_band["day_type_code"] = daily_route_band["day_type"].map(DAY_TYPE_CODE).fillna(0).astype(int)
    daily_route_band["weather_code"] = daily_route_band["weather"].map(WEATHER_CODE).fillna(0).astype(int)
    daily_route_band["season_code"] = daily_route_band["season"].map(SEASON_CODE).fillna(0).astype(int)
    daily_route_band["time_band_code"] = daily_route_band["time_band"].map(TIME_BAND_CODE).fillna(0).astype(int)
    daily_route_band["route_category_code"] = daily_route_band["route_category"].map(ROUTE_CATEGORY_CODE).fillna(0).astype(int)
    daily_route_band["is_peak_int"] = daily_route_band["is_peak"].astype(int)
    daily_route_band["is_weekend_int"] = daily_route_band["is_weekend"].astype(int)
    daily_route_band["weekday_sin"] = np.sin(2 * np.pi * daily_route_band["weekday"] / 7)
    daily_route_band["weekday_cos"] = np.cos(2 * np.pi * daily_route_band["weekday"] / 7)
    daily_route_band["month_sin"] = np.sin(2 * np.pi * daily_route_band["month"] / 12)
    daily_route_band["month_cos"] = np.cos(2 * np.pi * daily_route_band["month"] / 12)
    
    # Sort for lag computation
    daily_route_band = daily_route_band.sort_values(["route_id", "date", "time_band"]).reset_index(drop=True)
    
    # Lag features (per route x time_band)
    lag_cols = ["route_avg_delay", "route_p90_delay", "route_on_time_rate"]
    for col in lag_cols:
        # Short-term lags
        daily_route_band[f"{col}_lag1"] = daily_route_band.groupby(["route_id", "time_band"])[col].shift(1)
        daily_route_band[f"{col}_lag7"] = daily_route_band.groupby(["route_id", "time_band"])[col].shift(7)
        daily_route_band[f"{col}_lag14"] = daily_route_band.groupby(["route_id", "time_band"])[col].shift(14)
        
        # Rolling windows (shift before rolling to avoid leakage)
        daily_route_band[f"{col}_rolling7"] = (
            daily_route_band.groupby(["route_id", "time_band"])[col]
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        )
        daily_route_band[f"{col}_rolling14"] = (
            daily_route_band.groupby(["route_id", "time_band"])[col]
            .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
        )
        daily_route_band[f"{col}_rolling28"] = (
            daily_route_band.groupby(["route_id", "time_band"])[col]
            .transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
        )
    
    # Bunching flag per route (overall)
    valid = df["scheduled_headway_min"].notna() & df["actual_headway_min"].notna() & (df["scheduled_headway_min"] > 0)
    df_valid = df[valid].copy()
    df_valid["is_bunching"] = df_valid["actual_headway_min"] < 0.5 * df_valid["scheduled_headway_min"]
    bunching_data = df_valid.groupby("route_id")["is_bunching"].mean().reset_index(name="bunching_flag_route")
    
    daily_route_band = daily_route_band.merge(bunching_data, on="route_id", how="left")
    
    # Fill NaN lag features with group mean then global mean
    lag_rolling_cols = [c for c in daily_route_band.columns if c.endswith(("_lag1", "_lag7", "_lag14", "_rolling7", "_rolling14", "_rolling28"))]
    for col in lag_rolling_cols:
        daily_route_band[col] = daily_route_band.groupby(["route_id", "time_band"])[col].transform(lambda s: s.fillna(s.mean()))
        daily_route_band[col] = daily_route_band[col].fillna(daily_route_band[col].mean())
    
    daily_route_band["bunching_flag_route"] = daily_route_band["bunching_flag_route"].fillna(0)
    
    return daily_route_band


def _split_train_val_test(
    df: pd.DataFrame,
    val_days: int = VAL_DAYS,
    horizon_days: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leakage-safe chronological split: train | validation | test.
    
    The test window (the most recent ``FORECAST_HORIZON_DAYS``) is identical to
    the original split; the validation window sits directly before it and is the
    ONLY window used for model selection and hyperparameter tuning.
    """
    if horizon_days is None:
        horizon_days = settings.FORECAST_HORIZON_DAYS
    
    max_date = df["date"].max().normalize()
    cutoff_test = max_date - pd.Timedelta(days=horizon_days)
    cutoff_val = cutoff_test - pd.Timedelta(days=val_days)
    
    train = df[df["date"] <= cutoff_val].copy()
    val = df[(df["date"] > cutoff_val) & (df["date"] <= cutoff_test)].copy()
    test = df[df["date"] > cutoff_test].copy()
    
    logger.info(f"Train: {len(train)} rows up to {cutoff_val.date()}")
    logger.info(f"Val:   {len(val)} rows from {val['date'].min().date()} to {val['date'].max().date()}")
    logger.info(f"Test:  {len(test)} rows from {test['date'].min().date()} to {test['date'].max().date()}")
    
    return train, val, test


def _prepare_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare feature matrix and target."""
    # Select features that exist
    available_features = [f for f in DELAY_FEATURES if f in df.columns]
    
    X = df[available_features].copy()
    y = df[DELAY_TARGET].copy()
    
    # Handle missing values
    X = X.ffill().bfill()
    medians = X.median(numeric_only=True)
    medians = medians.fillna(0)
    X = X.fillna(medians)
    y = y.fillna(y.median() if not y.isna().all() else 0)
    
    # Clip target to reasonable range
    y = y.clip(lower=-10, upper=60)
    
    return X, y, available_features


def _make_rf_model(params: dict):
    """Create RandomForestRegressor with given parameters."""
    return RandomForestRegressor(random_state=2024, n_jobs=-1, **params)


def _make_gbr_model(params: dict):
    """Create GradientBoostingRegressor with given parameters."""
    return GradientBoostingRegressor(random_state=2024, **params)


# Candidate configurations for Random Forest
RF_CANDIDATES = {
    "rf_shallow": dict(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=15,
        min_samples_split=30,
        max_features=0.4,
    ),
    "rf_medium": dict(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features=0.5,
    ),
    "rf_deep": dict(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=8,
        min_samples_split=15,
        max_features=0.6,
    ),
    "rf_wide_shallow": dict(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=20,
        min_samples_split=40,
        max_features=0.3,
    ),
}

# Candidate configurations for Gradient Boosting
GBR_CANDIDATES = {
    "gbr_conservative": dict(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=15,
        min_samples_split=30,
        learning_rate=0.02,
        subsample=0.7,
        max_features=0.4,
    ),
    "gbr_balanced": dict(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=10,
        min_samples_split=20,
        learning_rate=0.03,
        subsample=0.7,
        max_features=0.5,
    ),
    "gbr_flexible": dict(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=8,
        min_samples_split=15,
        learning_rate=0.04,
        subsample=0.8,
        max_features=0.6,
    ),
    "gbr_very_conservative": dict(
        n_estimators=400,
        max_depth=3,
        min_samples_leaf=20,
        min_samples_split=40,
        learning_rate=0.015,
        subsample=0.6,
        max_features=0.3,
    ),
}


def _fit_predict_rf(params: dict, X_train, y_train, X_eval):
    model = _make_rf_model(params)
    model.fit(X_train, y_train)
    return np.clip(model.predict(X_eval), -10, 60)


def _fit_predict_gbr(params: dict, X_train, y_train, X_eval):
    model = _make_gbr_model(params)
    model.fit(X_train, y_train)
    return np.clip(model.predict(X_eval), -10, 60)


def train_delay_model(
    model_type: str = "random_forest",
    horizon_days: int = None,
) -> dict[str, Any]:
    """Train delay prediction model with validation-based model selection."""
    import joblib
    
    t0 = time.perf_counter()
    
    logger.info("Loading delay features...")
    df = _load_delay_features()
    logger.info(f"Loaded {len(df)} delay feature rows")
    
    # Split: train / validation / test
    train, val, test = _split_train_val_test(df, val_days=VAL_DAYS, horizon_days=horizon_days)
    
    # Prepare features
    X_train, y_train, feature_names = _prepare_features_targets(train)
    X_val, y_val, _ = _prepare_features_targets(val)
    X_test, y_test, _ = _prepare_features_targets(test)
    
    # Model selection on validation window
    if model_type == "random_forest":
        logger.info("Selecting best Random Forest configuration on validation data...")
        selection = {}
        for name, params in RF_CANDIDATES.items():
            preds = _fit_predict_rf(params, X_train, y_train, X_val)
            selection[name] = metrics(preds, y_val.to_numpy())
            logger.info(f"  {name}: R²={selection[name]['r2']:.4f}, MAE={selection[name]['mae']:.3f}")
        best_name = max(selection, key=lambda k: selection[k]["r2"])
        best_params = RF_CANDIDATES[best_name]
        logger.info(f"Selected: {best_name}")
        
        # Final fit on train + validation
        X_pool = pd.concat([X_train, X_val], ignore_index=True)
        y_pool = pd.concat([y_train, y_val], ignore_index=True)
        model = _make_rf_model(best_params)
        model.fit(X_pool, y_pool)
        
    elif model_type == "gradient_boosting":
        logger.info("Selecting best Gradient Boosting configuration on validation data...")
        selection = {}
        for name, params in GBR_CANDIDATES.items():
            preds = _fit_predict_gbr(params, X_train, y_train, X_val)
            selection[name] = metrics(preds, y_val.to_numpy())
            logger.info(f"  {name}: R²={selection[name]['r2']:.4f}, MAE={selection[name]['mae']:.3f}")
        best_name = max(selection, key=lambda k: selection[k]["r2"])
        best_params = GBR_CANDIDATES[best_name]
        logger.info(f"Selected: {best_name}")
        
        # Final fit on train + validation
        X_pool = pd.concat([X_train, X_val], ignore_index=True)
        y_pool = pd.concat([y_train, y_val], ignore_index=True)
        model = _make_gbr_model(best_params)
        model.fit(X_pool, y_pool)
        
    elif model_type == "linear":
        # Linear model doesn't need hyperparameter selection
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
        model.fit(X_train, y_train)
        selection = {}
        best_name = "linear"
        best_params = {}
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Training final {model_type} model on {len(X_pool)} samples...")
    
    # Predict on test
    preds = model.predict(X_test)
    preds = np.clip(preds, -10, 60)
    
    # Calculate metrics
    m = metrics(preds, y_test.to_numpy())
    
    # Prepare output
    out = test[["route_id", "date", "time_band", DELAY_TARGET]].copy()
    out["predicted_delay_min"] = np.round(preds, 2)
    out = out.sort_values(["route_id", "date", "time_band"]).reset_index(drop=True)
    
    # Save model
    MODELS_PYTHON.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_PYTHON / f"delay_prediction_{model_type}.joblib"
    joblib.dump(model, model_path)
    
    # Save predictions
    pred_path = save_parquet(
        out[["route_id", "date", "time_band", DELAY_TARGET, "predicted_delay_min"]],
        f"delay_prediction_predictions_{model_type}"
    )
    
    # Save metrics with validation selection info
    metrics_path = MODELS_PYTHON / f"delay_prediction_{model_type}_metrics.json"
    metrics_path.write_text(
        json.dumps({
            **m,
            "engine": "scikit-learn",
            "algorithm": model_type,
            "target": DELAY_TARGET,
            "test_rows": int(len(test)),
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "final_fit_rows": int(len(X_pool)),
            "features": feature_names,
            "validation_selection": selection,
            "best_config": best_name,
            "best_params": best_params,
        }, indent=2, default=str),
        encoding="utf-8"
    )
    
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    
    return {
        "model": f"delay_prediction_{model_type}",
        "engine": "scikit-learn",
        "algorithm": model_type,
        "target": DELAY_TARGET,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "final_fit_rows": int(len(X_pool)),
        **{k: m[k] for k in ("mae", "rmse", "mape_pct", "r2")},
        "val_r2": selection.get(best_name, {}).get("r2"),
        "elapsed_ms": elapsed_ms,
        "model_path": str(model_path),
        "predictions_path": str(pred_path),
    }


def evaluate_delay_model(model_type: str = "random_forest") -> dict[str, Any]:
    """Evaluate delay prediction model on test set."""
    pred_path = MODELS_PYTHON / f"delay_prediction_predictions_{model_type}.parquet"
    if not pred_path.exists():
        return {"error": f"Predictions not found: {pred_path}"}
    
    preds = pd.read_parquet(pred_path)
    
    m = metrics(
        preds["predicted_delay_min"].to_numpy(),
        preds[DELAY_TARGET].to_numpy()
    )
    
    return {
        "cases": int(len(preds)),
        **m,
    }


def predict_delay(
    route_ids: list[str] = None,
    date: str = None,
    model_type: str = "random_forest",
) -> pd.DataFrame:
    """Generate delay predictions for specified routes/date."""
    import joblib
    
    model_path = MODELS_PYTHON / f"delay_prediction_{model_type}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first.")
    
    model = joblib.load(model_path)
    
    # Load latest features for prediction
    df = _load_delay_features()
    
    if route_ids:
        df = df[df["route_id"].isin(route_ids)]
    if date:
        df = df[df["date"] == pd.Timestamp(date)]
    
    # One row per route/date/time_band, matching the training grain.
    latest = df.groupby(["route_id", "date", "time_band"]).tail(1).copy()

    # Prepare features
    X, _, feature_names = _prepare_features_targets(latest)
    X = X[feature_names]

    # Predict. ``model.predict`` returns a NumPy array, so use positional bounds
    # with ``np.clip``: pandas' ``Series.clip(lower=, upper=)`` signature leaves
    # NumPy without bounds and raises "One of max or min must be given".
    preds = np.clip(model.predict(X), -10, 60)

    forecast = latest[["route_id", "date", "time_band", DELAY_TARGET]].copy()
    forecast["predicted_delay_min"] = np.round(preds, 2)
    forecast[DELAY_TARGET] = forecast[DELAY_TARGET].round(2)

    return forecast[["route_id", "date", "time_band", DELAY_TARGET, "predicted_delay_min"]]


def build_delay_models() -> dict[str, Any]:
    """Train all delay prediction models."""
    results = {}
    
    for model_type in MODEL_TYPES:
        try:
            logger.info(f"Training {model_type} delay model...")
            result = train_delay_model(model_type=model_type)
            results[model_type] = result
            logger.info(f"  {model_type}: MAE={result['mae']:.3f}, R²={result['r2']:.4f}")
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            results[model_type] = {"error": str(e)}
    
    return results


def load_saved_model_results() -> dict[str, Any]:
    """Rebuild the :func:`build_delay_models` result shape from saved metrics.

    Each model writes a *flat* metrics JSON keyed by metric name (``mae``,
    ``r2``, ...). The dashboard wants the nested ``{model_type: {metric: value}}``
    shape that training returns, so reassemble it here rather than in the page.
    Reading a single flat file and iterating it as a nested dict made the page
    raise ``TypeError: argument of type 'float' is not iterable``.
    """
    results: dict[str, Any] = {}
    for model_type in MODEL_TYPES:
        metrics_path = MODELS_PYTHON / f"delay_prediction_{model_type}_metrics.json"
        if not metrics_path.exists():
            continue
        try:
            results[model_type] = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not read saved metrics for %s: %s", model_type, exc)
    return results


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    result = build_delay_models()
    print(json.dumps(result, indent=2, default=str))
    print(f"Total time: {time.perf_counter() - t0:.1f}s")