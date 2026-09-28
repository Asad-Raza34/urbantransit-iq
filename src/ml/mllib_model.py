"""Spark MLlib demand-forecast model.

Note on the JVM-only policy: on this machine the pyspark python-worker bridge
is unreliable (python workers cannot even be spawned for tree-model scoring),
so the MLlib engine here uses LinearRegression: its transform is a pure JVM
operation, and predictions are landed as a parquet sink (no python worker is
ever started for training or scoring). Tree-based MLlib regressors would fail
on this Windows box and are NOT faked; the scikit-learn engine carries the
tree/boosting side of the comparison honestly.

Validation protocol (leakage-safe, mirrors ``sklearn_model``):

* ``split_train_val_test`` cuts the year into train | validation | test;
* the regularisation grid (``regParam`` x ``elasticNetParam``) is scored on
  the validation window only, and the best pair is then re-fitted on
  train+validation for the single final test evaluation;
* the test window is never used for tuning.

The route id is mapped to a stable numeric code upfront and fed as a plain
continuous feature; using a Spark ``StringIndexer`` would declare it a
categorical attribute, which forces ``maxBins`` >= 100 for this network.
Features are standardised so the elastic-net penalty treats them evenly.
"""

import json
import time

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression

from src.ml.demand import (CONTEXT_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES,
                           TARGET, add_context_features, demand_data, metrics,
                           normalise_date_column, save_parquet,
                           split_train_val_test)
from src.paths import MODELS_SPARK
from src.spark_context import spark_available, spark_session
from src.storage import make_spark_safe

MODEL_NAME = "demand_forecast"
CACHE_ROOT = MODELS_SPARK / "cache"
ROUTE_CODE = "route_code"
VAL_DAYS = 28

REG_GRID = [
    (0.001, 0.0),
    (0.003, 0.0),
    (0.01, 0.0),
    (0.03, 0.0),
    (0.001, 0.5),
    (0.01, 0.5),
    (0.001, 1.0),
    (0.01, 1.0),
]


def _code_routes(train: pd.DataFrame, *others: pd.DataFrame) -> tuple:
    labels = sorted(set().union(*[set(f["route_id"]) for f in (train, *others)]))
    codes = {r: i for i, r in enumerate(labels)}

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        return (frame.assign(route_code=frame["route_id"].map(codes))
                .drop(columns=["route_id"]))

    return tuple(apply(f) for f in (train, *others))


def _spark_frame(spark, frame: pd.DataFrame, name: str) -> str:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = CACHE_ROOT / f"{name}.parquet"
    make_spark_safe(frame).to_parquet(path, index=False)
    return str(path)


def _feature_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in MODEL_FEATURES if c in frame.columns] + [ROUTE_CODE]


def _make_pipeline(feature_cols: list[str], reg_param: float,
                   enet_param: float) -> Pipeline:
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                            withMean=True, withStd=True)
    regressor = LinearRegression(featuresCol="features", labelCol=TARGET,
                                 regParam=reg_param, elasticNetParam=enet_param,
                                 maxIter=200, solver="l-bfgs")
    return Pipeline(stages=[assembler, scaler, regressor])


def _impute(train: pd.DataFrame, frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Fill NaN history/context values with training-column medians."""
    medians = train[NUMERIC_FEATURES + CONTEXT_FEATURES].median(numeric_only=True)
    medians = medians.fillna(0.0)

    def apply(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in medians.index:
            if col in out.columns:
                out[col] = out[col].fillna(medians[col])
        return out

    return [apply(f) for f in frames]


def _train_and_score(train_pdf: pd.DataFrame, val_pdf: pd.DataFrame,
                     reg_param: float, enet_param: float) -> tuple[Pipeline, dict]:
    """Fit one grid point on the training window and score validation."""
    spark = spark_session()
    train_path = _spark_frame(spark, train_pdf, "train")
    spark.read.parquet(train_path).createOrReplaceTempView("train_ml")
    pipeline = _make_pipeline(_feature_cols(train_pdf), reg_param, enet_param)
    model = pipeline.fit(spark.table("train_ml"))

    val_path = _spark_frame(spark, val_pdf, "val")
    spark.read.parquet(val_path).createOrReplaceTempView("val_ml")
    score_path = CACHE_ROOT / "val_predictions.parquet"
    model.transform(spark.table("val_ml")).drop("raw_features", "features") \
        .write.mode("overwrite").parquet(str(score_path))
    preds = pd.read_parquet(score_path)
    m = metrics(np.clip(preds["prediction"].to_numpy(), 0, None),
                preds[TARGET].to_numpy())
    return model, m


def _evaluate(counts: dict, model: Pipeline) -> tuple[pd.DataFrame, dict]:
    """Score the unseen window with the freshly-fitted model.

    The model object is handed in rather than reloaded from disk: on this
    machine reloading a saved pipeline routes through the python worker, which
    cannot start (``pyspark.daemon`` imports POSIX-only signals and crashes on
    Windows).
    """
    spark = spark_session()
    test = counts["test"]
    test_path = _spark_frame(spark, test, "test")
    spark.read.parquet(test_path).createOrReplaceTempView("test_ml")

    preds_path = CACHE_ROOT / "predictions.parquet"
    model.transform(spark.table("test_ml")) \
        .drop("raw_features", "features") \
        .write.mode("overwrite") \
        .parquet(str(preds_path))

    preds = pd.read_parquet(preds_path)
    # Spark stores the timestamp as integer microseconds; normalise to datetime so
    # this artifact shares its key dtype with the sklearn predictions (the parity
    # comparison merges the two frames on ``date``).
    preds["date"] = normalise_date_column(preds["date"])
    preds["predicted"] = preds["prediction"].clip(lower=0)

    # re-attach the route ids the codes stand for
    route_lookup = counts["routes"].set_index(ROUTE_CODE)["route_id"]
    preds.insert(0, "route_id", preds[ROUTE_CODE].map(route_lookup).values)

    preds = preds.sort_values(["route_id", "date", "time_band_code"]).reset_index(drop=True)
    m = metrics(preds["predicted"].to_numpy(), preds[TARGET].to_numpy())
    m["engine"] = "spark-mllib"
    m["algorithm"] = "linear-regression"
    return preds, m


def build() -> dict:
    """Train/evaluate the MLlib demand model and persist everything."""
    if not spark_available()[0]:
        return {"status": "skipped", "message": "spark unavailable"}

    t0 = time.perf_counter()
    frame = add_context_features(demand_data())
    train, val, test = split_train_val_test(frame, val_days=VAL_DAYS)

    train, val, test = _impute(train, [train, val, test])
    train, val, test = _code_routes(train, val, test)
    pooled = pd.concat([train, val], ignore_index=True)

    labels = sorted(set(frame["route_id"]))
    routes = pd.DataFrame({"route_id": labels,
                           ROUTE_CODE: np.arange(len(labels), dtype=int)})
    counts = {"train": pooled, "test": test, "routes": routes}

    # --- grid selection on the VALIDATION window only ---------------------- #
    selection = {}
    best = None
    for reg_param, enet_param in REG_GRID:
        _, m = _train_and_score(train, val, reg_param, enet_param)
        selection[f"regParam={reg_param},elasticNet={enet_param}"] = m
        if best is None or m["r2"] > best[2]["r2"]:
            best = (reg_param, enet_param, m)
    reg_param, enet_param, best_val = best

    # --- final fit on train+validation, single evaluation on test ---------- #
    spark = spark_session()
    pool_path = _spark_frame(spark, pooled, "train")
    spark.read.parquet(pool_path).createOrReplaceTempView("train_ml")
    pipeline = _make_pipeline(_feature_cols(pooled), reg_param, enet_param)
    model = pipeline.fit(spark.table("train_ml"))

    model.write().overwrite().save(str(MODELS_SPARK / MODEL_NAME))
    (MODELS_SPARK / f"{MODEL_NAME}_features.json").write_text(
        json.dumps({"features": _feature_cols(pooled),
                    "target": TARGET,
                    "engine": "spark-mllib",
                    "algorithm": "linear-regression",
                    "regParam": reg_param,
                    "elasticNetParam": enet_param,
                    "validation_selection": selection},
                   indent=2), encoding="utf-8")

    preds, m = _evaluate(counts, model)

    save_parquet(preds[["route_id", "date", "time_band_code", "time_band",
                        TARGET, "predicted"]], "demand_forecast_predictions_mllib")

    (MODELS_SPARK / f"{MODEL_NAME}_metrics.json").write_text(
        json.dumps({"mae": m["mae"], "rmse": m["rmse"], "mape_pct": m["mape_pct"],
                    "r2": m["r2"], "n": m["n"], "engine": "spark-mllib",
                    "algorithm": "linear-regression",
                    "regParam": reg_param, "elasticNetParam": enet_param,
                    "train_rows": int(len(train)), "val_rows": int(len(val)),
                    "test_rows": int(len(test)),
                    "final_fit_rows": int(len(pooled)),
                    "val_r2": best_val["r2"]},
                   indent=2), encoding="utf-8")

    return {"model": MODEL_NAME, "engine": "spark-mllib",
            "algorithm": "linear-regression",
            "train_rows": int(len(train)), "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "val_r2": best_val["r2"],
            **{k: m[k] for k in ("mae", "rmse", "mape_pct", "r2")},
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}


def main() -> None:
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()
