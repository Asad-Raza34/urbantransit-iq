"""Occupancy forecasting for UrbanTransit IQ.

Predicts future route-level occupancy using historical data and lag features.
Reuses the existing demand forecasting infrastructure where possible.
"""

import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import settings
from src.analytics.core import read_processed
from src.log import get_logger
from src.ml.demand import NUMERIC_FEATURES, TARGET, metrics, save_parquet, unseen_cases
from src.paths import DATA_ANALYTICS, MODELS_PYTHON
from src.storage import write_dataset

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

# Model families trained by this module, in display order.
MODEL_TYPES = ["random_forest", "gradient_boosting", "linear"]


@dataclass
class OccupancyModelMetrics:
    """Metrics for occupancy forecasting model."""
    mae: float
    rmse: float
    mape_pct: float
    r2: float
    n: int


# Target: we predict occupancy_max_pct for the next time period
OCCUPANCY_TARGET = "occupancy_max_pct"

# Features for occupancy forecasting (from demand_features + route-level features)
OCCUPANCY_FEATURES = [
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    "day_type_code", "weather_code", "season_code", "time_band_code",
    "prev_week_boardings", "rolling_7d_boardings",
    # Add occupancy-specific features
    "prev_week_occupancy_max",
    "rolling_7d_occupancy_max",
    "prev_week_occupancy_avg",
    "rolling_7d_occupancy_avg",
]


def _load_occupancy_features() -> pd.DataFrame:
    """Load and prepare occupancy features from demand_features and route_metrics."""
    # Load demand features
    from src.paths import DATA_FEATURES
    from src.storage import read_dataset
    demand = read_dataset(DATA_FEATURES / "demand_features.parquet")
    
    # We need occupancy history - load from trip_facts aggregated
    trip_facts = read_processed("trip_facts")
    
    # Compute occupancy features at route x date x time_band level
    occ_features = trip_facts.groupby(
        ["route_id", "date", "time_band"], as_index=False
    ).agg(
        occupancy_max_pct=("occupancy_max_pct", "mean"),
        occupancy_avg_pct=("occupancy_avg_pct", "mean"),
        boardings=("boardings", "sum"),
    )
    
    # Convert date
    occ_features["date"] = pd.to_datetime(occ_features["date"])
    occ_features = occ_features.sort_values(["route_id", "time_band", "date"]).reset_index(drop=True)
    
    # Create lag features
    lag_key = ["route_id", "time_band"]
    
    # Previous week occupancy (7 days back)
    occ_features["prev_week_occupancy_max"] = (
        occ_features.groupby(lag_key)["occupancy_max_pct"].shift(7)
    )
    occ_features["prev_week_occupancy_avg"] = (
        occ_features.groupby(lag_key)["occupancy_avg_pct"].shift(7)
    )
    
    # Rolling 7-day average of occupancy
    occ_features["rolling_7d_occupancy_max"] = (
        occ_features.groupby(lag_key)["occupancy_max_pct"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    occ_features["rolling_7d_occupancy_avg"] = (
        occ_features.groupby(lag_key)["occupancy_avg_pct"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    
    # Rolling 7-day average of boardings
    occ_features["rolling_7d_boardings"] = (
        occ_features.groupby(lag_key)["boardings"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    occ_features["prev_week_boardings"] = (
        occ_features.groupby(lag_key)["boardings"].shift(7)
    )
    
    # Add time features from date
    occ_features["weekday"] = occ_features["date"].dt.weekday
    occ_features["month"] = occ_features["date"].dt.month
    occ_features["weekday_sin"] = np.sin(2 * np.pi * occ_features["weekday"] / 7)
    occ_features["weekday_cos"] = np.cos(2 * np.pi * occ_features["weekday"] / 7)
    occ_features["month_sin"] = np.sin(2 * np.pi * occ_features["month"] / 12)
    occ_features["month_cos"] = np.cos(2 * np.pi * occ_features["month"] / 12)
    
    # Time band code
    TIME_BAND_CODE = {"night": 0, "morning peak": 1, "midday": 2, "evening peak": 3}
    occ_features["time_band_code"] = occ_features["time_band"].map(TIME_BAND_CODE)
    
    # Day type and weather from demand features
    demand_extra = demand[["route_id", "date", "time_band", "day_type", "weather", "season"]].copy()
    demand_extra["date"] = pd.to_datetime(demand_extra["date"])
    occ_features = occ_features.merge(demand_extra, on=["route_id", "date", "time_band"], how="left")
    
    # Encode categorical
    DAY_TYPE_CODE = {"weekday": 0, "weekend": 1, "holiday": 2}
    WEATHER_CODE = {"clear": 0, "rain": 1, "snow": 2, "storm": 3}
    SEASON_CODE = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
    
    occ_features["day_type_code"] = occ_features["day_type"].map(DAY_TYPE_CODE).fillna(0).astype(int)
    occ_features["weather_code"] = occ_features["weather"].map(WEATHER_CODE).fillna(0).astype(int)
    occ_features["season_code"] = occ_features["season"].map(SEASON_CODE).fillna(0).astype(int)
    
    return occ_features


def _split_time_series(df: pd.DataFrame, horizon_days: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test using time-based split."""
    if horizon_days is None:
        horizon_days = settings.FORECAST_HORIZON_DAYS
    
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=horizon_days)
    
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    
    logger.info(f"Train: {len(train)} rows up to {cutoff.date()}")
    logger.info(f"Test: {len(test)} rows from {test['date'].min().date()} to {test['date'].max().date()}")
    
    return train, test


def _prepare_features_targets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare feature matrix and target."""
    # Select features that exist
    available_features = [f for f in OCCUPANCY_FEATURES if f in df.columns]
    
    # Add route_id as categorical - use integer codes
    if "route_id" in df.columns:
        df = df.copy()
        df["route_id_code"] = df["route_id"].astype("category").cat.codes
        available_features = [f for f in available_features if f != "route_id"] + ["route_id_code"]
    
    X = df[available_features].copy()
    y = df[OCCUPANCY_TARGET].copy()
    
    # Handle missing values - forward fill then backward fill for lag features, then median, then 0
    X = X.ffill().bfill()
    medians = X.median(numeric_only=True)
    # Replace any NaN medians with 0
    medians = medians.fillna(0)
    X = X.fillna(medians)
    y = y.fillna(y.median() if not y.isna().all() else 0)
    
    return X, y, available_features


def train_occupancy_model(
    model_type: str = "random_forest",
    horizon_days: int = None,
) -> dict[str, Any]:
    """Train occupancy forecasting model."""
    import joblib
    
    t0 = time.perf_counter()
    
    logger.info("Loading occupancy features...")
    df = _load_occupancy_features()
    logger.info(f"Loaded {len(df)} occupancy feature rows")
    
    # Split
    train, test = _split_time_series(df, horizon_days)
    
    # Prepare features
    X_train, y_train, feature_names = _prepare_features_targets(train)
    X_test, y_test, _ = _prepare_features_targets(test)
    
    # Build model
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=8,
            min_samples_split=20,
            max_features=0.6,
            random_state=2024,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=8,
            min_samples_split=20,
            learning_rate=0.03,
            subsample=0.7,
            max_features=0.6,
            random_state=2024,
        )
    elif model_type == "linear":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Training {model_type} model on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    
    # Predict on test set
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 150)
    
    # Calculate residuals for prediction intervals
    residuals = y_test.to_numpy() - preds
    residual_std = np.std(residuals)
    residual_q5 = np.percentile(residuals, 2.5)
    residual_q95 = np.percentile(residuals, 97.5)
    
    # Calculate metrics
    m = metrics(preds, y_test.to_numpy())
    
    # Prepare output
    out = test[["route_id", "date", "time_band", OCCUPANCY_TARGET]].copy()
    out["predicted"] = np.round(preds, 2)
    out = out.sort_values(["route_id", "date", "time_band"]).reset_index(drop=True)
    
    # Save model
    MODELS_PYTHON.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_PYTHON / f"occupancy_forecast_{model_type}.joblib"
    joblib.dump(model, model_path)
    
    # Save predictions
    pred_path = save_parquet(
        out[["route_id", "date", "time_band", OCCUPANCY_TARGET, "predicted"]],
        f"occupancy_forecast_predictions_{model_type}"
    )
    
    # Save residuals for prediction intervals
    residuals_path = MODELS_PYTHON / f"occupancy_forecast_{model_type}_residuals.npy"
    np.save(residuals_path, residuals)
    
    # Save metrics with interval info
    metrics_path = MODELS_PYTHON / f"occupancy_forecast_{model_type}_metrics.json"
    metrics_path.write_text(
        json.dumps({
            **m,
            "engine": "scikit-learn",
            "algorithm": model_type,
            "target": OCCUPANCY_TARGET,
            "test_rows": int(len(test)),
            "train_rows": int(len(train)),
            "features": feature_names,
            "residual_std": float(residual_std),
            "residual_q5": float(residual_q5),
            "residual_q95": float(residual_q95),
        }, indent=2),
        encoding="utf-8"
    )
    
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    
    return {
        "model": f"occupancy_forecast_{model_type}",
        "engine": "scikit-learn",
        "algorithm": model_type,
        "target": OCCUPANCY_TARGET,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        **{k: m[k] for k in ("mae", "rmse", "mape_pct", "r2")},
        "residual_std": float(residual_std),
        "residual_q5": float(residual_q5),
        "residual_q95": float(residual_q95),
        "elapsed_ms": elapsed_ms,
        "model_path": str(model_path),
        "predictions_path": str(pred_path),
    }


def evaluate_occupancy_forecast(model_type: str = "random_forest") -> dict[str, Any]:
    """Evaluate occupancy forecast on unseen cases."""
    # Load predictions
    pred_path = MODELS_PYTHON / f"occupancy_forecast_predictions_{model_type}.parquet"
    if not pred_path.exists():
        return {"error": f"Predictions not found: {pred_path}"}
    
    preds = pd.read_parquet(pred_path)
    
    # Get unseen cases
    cases = unseen_cases(preds, n=settings.COMPARE_UNSEEN_CASES)
    
    if cases.empty:
        return {"error": "No unseen cases available"}
    
    # Calculate metrics on unseen cases
    m = metrics(
        cases["predicted"].to_numpy(),
        cases["occupancy_max_pct"].to_numpy()
    )
    
    return {
        "cases": int(len(cases)),
        **m,
    }


def forecast_occupancy(
    route_ids: list[str] = None,
    horizon_days: int = 7,
    model_type: str = "random_forest",
) -> pd.DataFrame:
    """Generate occupancy forecasts for specified routes with prediction intervals."""
    import joblib
    
    model_path = MODELS_PYTHON / f"occupancy_forecast_{model_type}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first.")
    
    model = joblib.load(model_path)
    
    # Load metrics for interval calculation
    metrics_path = MODELS_PYTHON / f"occupancy_forecast_{model_type}_metrics.json"
    if metrics_path.exists():
        model_metrics = json.loads(metrics_path.read_text())
        residual_std = model_metrics.get("residual_std", 6.0)
        residual_q5 = model_metrics.get("residual_q5", -12.0)
        residual_q95 = model_metrics.get("residual_q95", 12.0)
    else:
        residual_std = 6.0
        residual_q5 = -12.0
        residual_q95 = 12.0
    
    # Load latest features for forecasting
    df = _load_occupancy_features()
    
    if route_ids:
        df = df[df["route_id"].isin(route_ids)]
    
    # Get latest date per route/time_band
    latest = df.groupby(["route_id", "time_band"]).tail(1).copy()
    
    # Prepare features
    X, _, feature_names = _prepare_features_targets(latest)
    X = X[feature_names]  # Ensure correct order
    
    # Predict. ``model.predict`` returns a NumPy array, so the bounds must go to
    # ``np.clip`` positionally: the pandas-only ``Series.clip(lower=, upper=)``
    # signature leaves NumPy with no bounds and raises
    # "One of max or min must be given".
    preds = np.clip(model.predict(X), 0, 150)

    # Compute prediction intervals using residual quantiles
    lower = np.clip(np.round(preds + residual_q5, 2), 0, None)
    upper = np.clip(np.round(preds + residual_q95, 2), None, 150)
    
    forecast = latest[["route_id", "date", "time_band", "occupancy_max_pct"]].copy()
    forecast["forecast_date"] = forecast["date"] + pd.Timedelta(days=horizon_days)
    forecast["predicted_occupancy_max_pct"] = np.round(preds, 2)
    forecast["lower_bound"] = lower
    forecast["upper_bound"] = upper
    forecast["horizon_days"] = horizon_days
    
    return forecast[["route_id", "date", "time_band", "forecast_date", 
                     "occupancy_max_pct", "predicted_occupancy_max_pct", 
                     "lower_bound", "upper_bound", "horizon_days"]]


def build_occupancy_models() -> dict[str, Any]:
    """Train all occupancy forecasting models."""
    results = {}
    
    for model_type in MODEL_TYPES:
        try:
            logger.info(f"Training {model_type} occupancy model...")
            result = train_occupancy_model(model_type=model_type)
            results[model_type] = result
            logger.info(f"  {model_type}: MAE={result['mae']:.3f}, R²={result['r2']:.4f}")
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            results[model_type] = {"error": str(e)}
    
    return results


def load_saved_model_results() -> dict[str, Any]:
    """Rebuild the :func:`build_occupancy_models` result shape from saved metrics.

    Each model writes a *flat* metrics JSON keyed by metric name (``mae``, ``r2``,
    ...). Consumers want the nested ``{model_type: {metric: value}}`` shape that
    training returns, so reassemble it here instead of duplicating the file
    naming convention in the dashboard layer.
    """
    results: dict[str, Any] = {}
    for model_type in MODEL_TYPES:
        metrics_path = MODELS_PYTHON / f"occupancy_forecast_{model_type}_metrics.json"
        if not metrics_path.exists():
            continue
        try:
            results[model_type] = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:  # noqa: PERF203
            logger.warning("could not read saved metrics for %s: %s", model_type, exc)
    return results


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    result = build_occupancy_models()
    print(json.dumps(result, indent=2, default=str))