"""scikit-learn demand-forecast model (comparison engine).

Trains on the exact same feature set, split and metrics as the Spark MLlib
model, so the two engines can be compared on equal terms (see
``src.ml.parity``).

Validation protocol (leakage-safe):

* ``split_train_val_test`` cuts the year into train (Jan 1-Nov 5), validation
  (Nov 6-Dec 3) and test (Dec 4-31, ``FORECAST_HORIZON_DAYS``) windows.
* Candidate model configurations are compared on the validation window only;
  the test window is never touched during selection.
* The selected configuration is then refitted on train+validation and scored
  exactly once on the untouched test window. Those test metrics are what get
  persisted and reported.

Route identity enters the trees as a stable integer ``route_code`` (mapped
from the full frame, so no window can contain an unseen code). HistGradient
Boosting declares it a true categorical feature (native categorical splits);
RandomForest splits on it numerically. Both are far stronger than the previous
pipeline's ordinal-encoding-through-OrdinalEncoder on a single engine and
installable on any Windows box — no Spark dependency for the python engine.
"""

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from src.ml.demand import (MODEL_FEATURES, TARGET, add_context_features,
                           demand_data, metrics, save_parquet,
                           split_train_val_test, unseen_cases)
from src.paths import MODELS_PYTHON

MODEL_NAME = "demand_forecast"
RANDOM_STATE = 2024
VAL_DAYS = 28

# Candidate configurations. Selection happens on the validation window only.
CANDIDATES = {
    "random_forest": dict(
        kind="rf",
        n_estimators=400, min_samples_leaf=2, max_features=0.6,
        n_jobs=-1,
    ),
    "hist_gbr": dict(
        kind="hist", loss="squared_error",
        learning_rate=0.06, max_iter=600, max_leaf_nodes=41,
        min_samples_leaf=15, l2_regularization=0.5,
    ),
    "hist_gbr_poisson": dict(
        kind="hist", loss="poisson",
        learning_rate=0.06, max_iter=600, max_leaf_nodes=41,
        min_samples_leaf=15, l2_regularization=0.5,
    ),
}


def _prepare(frame: pd.DataFrame, route_codes: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix: model features + the shared ``route_code`` map."""
    data = frame.copy()
    features = [c for c in MODEL_FEATURES if c in data.columns]
    x = data[features].copy()
    # NaN history (rows without past observations) -> a distinct code that
    # trees can route around; linear engines impute instead.
    x = x.fillna(-1.0)
    x["route_code"] = data["route_id"].map(route_codes).astype("int16")
    y = data[TARGET]
    return x, y


def _make_model(spec: dict):
    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    if spec["kind"] == "rf":
        return RandomForestRegressor(random_state=RANDOM_STATE, **{
            k: v for k, v in spec.items() if k != "kind"})
    # route_code values are 0..99, comfortably below max_bins (255).
    return HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        categorical_features=["route_code"],
        **{k: v for k, v in spec.items() if k != "kind"})


def _fit_predict(spec: dict, x_train, y_train, x_eval):
    model = _make_model(spec)
    model.fit(x_train, y_train)
    return np.clip(model.predict(x_eval), 0, None)


def build() -> dict:
    t0 = time.perf_counter()
    frame = add_context_features(demand_data())
    train, val, test = split_train_val_test(frame, val_days=VAL_DAYS)

    # Route codes are built from the FULL frame so every window shares the
    # same mapping and no route can be unseen after the split.
    route_codes = {r: i for i, r in enumerate(sorted(frame["route_id"].unique()))}

    x_train, y_train = _prepare(train, route_codes)
    x_val, y_val = _prepare(val, route_codes)
    x_test, y_test = _prepare(test, route_codes)

    # --- model selection on the VALIDATION window only --------------------- #
    selection = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for name, spec in CANDIDATES.items():
            preds = _fit_predict(spec, x_train, y_train, x_val)
            selection[name] = metrics(preds, y_val.to_numpy())
    best_name = max(selection, key=lambda k: selection[k]["r2"])
    best_spec = CANDIDATES[best_name]

    # --- final fit on train+validation, single evaluation on test ---------- #
    pooled = pd.concat([train, val], ignore_index=True)
    x_pool, y_pool = _prepare(pooled, route_codes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        final_model = _make_model(best_spec)
        final_model.fit(x_pool, y_pool)
        preds = np.clip(final_model.predict(x_test), 0, None)

    out = test[["route_id", "date", "time_band_code", "time_band", TARGET]].copy()
    out["predicted"] = preds.round(2)
    out = out.sort_values(["route_id", "date", "time_band_code"]).reset_index(drop=True)

    m = metrics(preds, y_test.to_numpy())
    m["engine"] = "scikit-learn"
    m["algorithm"] = best_name

    MODELS_PYTHON.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODELS_PYTHON / f"{MODEL_NAME}.joblib")
    save_parquet(out, "demand_forecast_predictions_sklearn")

    (MODELS_PYTHON / f"{MODEL_NAME}_metrics.json").write_text(
        json.dumps({"mae": m["mae"], "rmse": m["rmse"], "mape_pct": m["mape_pct"],
                    "r2": m["r2"], "n": m["n"], "engine": "scikit-learn",
                    "algorithm": best_name,
                    "train_rows": int(len(train)), "val_rows": int(len(val)),
                    "test_rows": int(len(test)),
                    "final_fit_rows": int(len(pooled)),
                    "validation_selection": selection,
                    "features": [c for c in x_train.columns],
                    "cases_table": int(len(unseen_cases(test)))},
                   indent=2), encoding="utf-8")

    return {"model": MODEL_NAME, "engine": "scikit-learn",
            "algorithm": best_name,
            "train_rows": int(len(train)), "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "val_r2": selection[best_name]["r2"],
            **{k: m[k] for k in ("mae", "rmse", "mape_pct", "r2")},
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}


def main() -> None:
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
