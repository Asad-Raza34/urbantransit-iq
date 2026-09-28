"""Dual-engine comparisons required by the SRS.

Two parity evidence sets are produced:

1. **Forecast parity over 100 unseen cases**: a deterministic sample of
   future-unseen route/date/time-band cases scored by the Spark MLlib demand
   model and the scikit-learn demand model, reported against actual demand.

   Outputs: ``models/python/forecast_parity.parquet`` +
   ``forecast_parity_summary.json``.

2. **Analytics pipeline parity**: the same route reliability metrics computed
   through the pandas pipeline (``route_metrics``) and the Spark SQL pipeline
   (``spark_route_reliability``) compared across all 100 routes, proving the
   two engines agree on the business numbers.

   Outputs: ``models/python/pipeline_parity.parquet`` +
   ``pipeline_parity_summary.json``.
"""

import json

import numpy as np
import pandas as pd

from src.analytics.core import read_processed
from src.ml.demand import metrics, normalise_date_column, save_parquet, unseen_cases
from src.paths import DATA_ANALYTICS, MODELS_PYTHON
from src.storage import read_dataset

CASE_COUNT = 100

KEY_COLS = ["route_id", "date", "time_band_code"]
PREDICTION_CACHE = {
    "mllib": "demand_forecast_predictions_mllib.parquet",
    "sklearn": "demand_forecast_predictions_sklearn.parquet",
}


def _forecast_parity() -> dict:
    mllib = read_dataset(MODELS_PYTHON / PREDICTION_CACHE["mllib"])
    sklearn = read_dataset(MODELS_PYTHON / PREDICTION_CACHE["sklearn"])

    # The two engines store ``date`` differently (Spark: integer microseconds,
    # pandas: datetime64). Merging them un-normalised raised
    # "merge on datetime64[ns] and int64" and the parity comparison never ran.
    # Normalise both sides so existing and future artifacts both work.
    mllib["date"] = normalise_date_column(mllib["date"])
    sklearn["date"] = normalise_date_column(sklearn["date"])

    cases = unseen_cases(sklearn, CASE_COUNT)
    if not len(cases):
        return {"status": "no predictions", "cases": 0}

    # ``time_band`` is a display label that has to be carried through the join;
    # it used to be requested in ``keep`` without ever being selected, which
    # raised KeyError as soon as the dtype fix let execution reach this point.
    label_cols = [c for c in ("time_band",) if c in cases.columns]
    sk = cases[KEY_COLS + label_cols + ["boardings", "predicted"]].rename(
        columns={"predicted": "forecast_sklearn"})
    ml = mllib[KEY_COLS + ["predicted"]].rename(columns={"predicted": "forecast_mllib"})

    table = sk.merge(ml, on=KEY_COLS, how="left")
    table["actual"] = table["boardings"]
    table["forecast_mllib"] = table["forecast_mllib"].round(2)
    table["forecast_sklearn"] = table["forecast_sklearn"].round(2)
    table["error_mllib"] = (table["forecast_mllib"] - table["actual"]).round(2)
    table["error_sklearn"] = (table["forecast_sklearn"] - table["actual"]).round(2)

    keep = KEY_COLS + label_cols + ["actual", "forecast_mllib", "forecast_sklearn",
                                    "error_mllib", "error_sklearn"]
    table = table[keep].sort_values(KEY_COLS).reset_index(drop=True)
    save_parquet(table, "forecast_parity")

    summary = {
        "cases": int(len(table)),
        "mllib": metrics(table["forecast_mllib"].to_numpy(), table["actual"].to_numpy()),
        "sklearn": metrics(table["forecast_sklearn"].to_numpy(), table["actual"].to_numpy()),
        "mean_abs_error_gap_mllib_minus_sklearn": round(
            float((table["error_mllib"].abs() - table["error_sklearn"].abs()).mean()), 3),
    }
    (MODELS_PYTHON / "forecast_parity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return {"cases": int(len(table)),
            "mllib_rmse": summary["mllib"]["rmse"],
            "sklearn_rmse": summary["sklearn"]["rmse"],
            "mllib_mape_pct": summary["mllib"]["mape_pct"],
            "sklearn_mape_pct": summary["sklearn"]["mape_pct"]}


def _pipeline_parity() -> dict:
    pandas_metrics = read_processed("trip_facts")[["route_id", "on_time", "departure_delay_min"]] \
        .groupby("route_id").agg(
            on_time_rate_pandas=("on_time", lambda s: round(100.0 * s.mean(), 2)),
            avg_delay_pandas=("departure_delay_min", "mean")).reset_index()
    spark_metrics = read_dataset(DATA_ANALYTICS / "spark_route_reliability.parquet") \
        [["route_id", "on_time_rate", "avg_delay_min"]]

    table = pandas_metrics.merge(spark_metrics, on="route_id")
    table["on_time_delta"] = (table["on_time_rate_pandas"] - table["on_time_rate"]).round(3)
    table["delay_delta"] = (table["avg_delay_pandas"] - table["avg_delay_min"]).round(3)
    table = table.sort_values("route_id").reset_index(drop=True)
    save_parquet(table, "pipeline_parity")

    summary = {
        "routes": int(len(table)),
        "max_on_time_delta": float(table["on_time_delta"].abs().max()),
        "mean_on_time_delta_abs": float(table["on_time_delta"].abs().mean()),
        "max_delay_delta": float(table["delay_delta"].abs().max()),
        "mean_delay_delta_abs": float(table["delay_delta"].abs().mean()),
    }
    (MODELS_PYTHON / "pipeline_parity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build() -> dict:
    forecast = _forecast_parity()
    pipeline = _pipeline_parity()
    return {"forecast_parity": forecast, "pipeline_parity": pipeline}


def main() -> None:
    print(json.dumps(build(), indent=2))


if __name__ == "__main__":
    main()