"""Lightweight model version registry for UrbanTransit IQ.

Records model metadata, training details, and version history without
requiring a full MLOps platform.
"""

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.paths import MODELS_VERSIONS, MODELS_PYTHON, MODELS_SPARK
from src.storage import write_dataset
from src.log import get_logger

logger = get_logger(__name__)


@dataclass
class ModelVersion:
    """Model version metadata."""
    version_id: str
    model_name: str
    engine: str                 # "scikit-learn" or "spark-mllib"
    algorithm: str              # e.g., "random-forest", "linear-regression"
    trained_at: str
    training_data_version: str  # hash or identifier of training data
    features: list[str]
    target: str
    metrics: dict[str, float]   # MAE, RMSE, MAPE, R2, etc.
    hyperparameters: dict[str, Any]
    artifact_path: str          # path to model file
    notes: str = ""
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    """Simple file-based model registry."""

    def __init__(self):
        MODELS_VERSIONS.mkdir(parents=True, exist_ok=True)
        self._registry_path = MODELS_VERSIONS / "registry.json"
        self._registry: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        return []

    def _save(self) -> None:
        self._registry_path.write_text(
            json.dumps(self._registry, indent=2, default=str), encoding="utf-8"
        )

    def register(self, version: ModelVersion) -> str:
        """Register a new model version."""
        entry = version.to_dict()
        entry["registered_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._registry.append(entry)
        self._save()
        logger.info(f"Registered model version: {version.version_id}")
        return version.version_id

    def get(self, version_id: str) -> dict | None:
        """Get a specific model version."""
        for v in self._registry:
            if v["version_id"] == version_id:
                return v
        return None

    def get_latest(self, model_name: str, engine: str = None) -> dict | None:
        """Get latest version for a model."""
        filtered = [v for v in self._registry if v["model_name"] == model_name]
        if engine:
            filtered = [v for v in filtered if v["engine"] == engine]
        if not filtered:
            return None
        # Sort by trained_at descending
        filtered.sort(key=lambda x: x["trained_at"], reverse=True)
        return filtered[0]

    def list_versions(self, model_name: str = None, engine: str = None) -> list[dict]:
        """List all versions, optionally filtered."""
        filtered = self._registry
        if model_name:
            filtered = [v for v in filtered if v["model_name"] == model_name]
        if engine:
            filtered = [v for v in filtered if v["engine"] == engine]
        return sorted(filtered, key=lambda x: x["trained_at"], reverse=True)

    def to_dataframe(self) -> pd.DataFrame:
        """Get registry as DataFrame."""
        if not self._registry:
            return pd.DataFrame()
        return pd.DataFrame(self._registry)


# Convenience functions for auto-registration from existing models

def register_sklearn_model(metrics_path: Path = None, model_path: Path = None) -> str:
    """Register the current scikit-learn model from artifacts."""
    if metrics_path is None:
        metrics_path = MODELS_PYTHON / "demand_forecast_metrics.json"
    if model_path is None:
        model_path = MODELS_PYTHON / "demand_forecast.joblib"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Load feature info
    features_path = MODELS_PYTHON / "demand_forecast_features.json"
    features = []
    if features_path.exists():
        features = json.loads(features_path.read_text(encoding="utf-8")).get("features", [])

    # Algorithm name lives in the metrics file since the demand-model
    # optimization (hist_gbr_poisson vs random_forest is chosen on the
    # validation window); older artifacts fall back to the previous engine.
    algorithm = metrics.get("algorithm", "random-forest")
    hyperparameters = metrics.get("hyperparameters") or {
        "selection_metric": "val_r2",
        "val_r2": metrics.get("val_r2"),
        "final_fit_rows": metrics.get("final_fit_rows"),
    }

    version = ModelVersion(
        version_id=f"sklearn-{uuid.uuid4().hex[:8]}",
        model_name="demand_forecast",
        engine="scikit-learn",
        algorithm=algorithm,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        training_data_version="demand_features_v1",
        features=features,
        target="boardings",
        metrics={
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mape_pct": metrics.get("mape_pct"),
            "r2": metrics.get("r2"),
            "n": metrics.get("n"),
        },
        hyperparameters=hyperparameters,
        artifact_path=str(model_path),
        tags=["production", "demand_forecast"],
    )

    registry = ModelRegistry()
    return registry.register(version)


def register_mllib_model(metrics_path: Path = None, model_path: Path = None) -> str:
    """Register the current Spark MLlib model from artifacts."""
    if metrics_path is None:
        metrics_path = MODELS_SPARK / "demand_forecast_metrics.json"
    if model_path is None:
        model_path = MODELS_SPARK / "demand_forecast"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Load feature info
    features_path = MODELS_SPARK / "demand_forecast_features.json"
    features = []
    if features_path.exists():
        features = json.loads(features_path.read_text(encoding="utf-8")).get("features", [])

    # Same treatment as the sklearn side: pull the selected regularisation
    # from the metrics file instead of assuming fixed hyperparameters.
    hyperparameters = metrics.get("hyperparameters") or {
        "regParam": metrics.get("regParam", 0.01),
        "elasticNetParam": metrics.get("elasticNetParam", 0.5),
        "maxIter": 200,
        "solver": "l-bfgs",
        "val_r2": metrics.get("val_r2"),
        "final_fit_rows": metrics.get("final_fit_rows"),
    }

    version = ModelVersion(
        version_id=f"mllib-{uuid.uuid4().hex[:8]}",
        model_name="demand_forecast",
        engine="spark-mllib",
        algorithm=metrics.get("algorithm", "linear-regression"),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        training_data_version="demand_features_v1",
        features=features,
        target="boardings",
        metrics={
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mape_pct": metrics.get("mape_pct"),
            "r2": metrics.get("r2"),
            "n": metrics.get("n"),
        },
        hyperparameters=hyperparameters,
        artifact_path=str(model_path),
        tags=["production", "demand_forecast"],
    )

    registry = ModelRegistry()
    return registry.register(version)


def auto_register_current_models() -> dict[str, str]:
    """Register both current models from existing artifacts."""
    results = {}
    try:
        results["sklearn"] = register_sklearn_model()
    except Exception as e:
        logger.warning(f"Failed to register sklearn model: {e}")
        results["sklearn"] = f"error: {e}"

    try:
        results["mllib"] = register_mllib_model()
    except Exception as e:
        logger.warning(f"Failed to register MLlib model: {e}")
        results["mllib"] = f"error: {e}"

    return results


if __name__ == "__main__":
    # Auto-register current models
    results = auto_register_current_models()
    print("Registered:", results)

    # Show registry
    registry = ModelRegistry()
    print("\nRegistry:")
    print(registry.to_dataframe().to_string(index=False))