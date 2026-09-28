"""Decisive check: can this machine run Spark work that stays inside the JVM?

The PySpark python-worker bridge is unreliable on this Windows box (worker
exits during task execution), but pure JVM-side operations - parquet IO,
Spark SQL, Spark MLlib estimators - never invoke a python worker. This
script exercises exactly that path end to end:

    pandas -> parquet -> Spark SQL agg -> MLlib fit/transform -> parquet -> pandas

If this passes, the Spark layer of the project can be driven through
file-based DataFrames and avoid the worker bridge entirely.
"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if not os.environ.get("JAVA_HOME"):
    jdk_dirs = sorted(Path("C:/Java").glob("jdk-*"))
    if jdk_dirs:
        os.environ["JAVA_HOME"] = str(jdk_dirs[-1])

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    sample_dir = PROJECT_ROOT / "data" / "samples" / "jvm_check"
    sample_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n = 5000
    frame = pd.DataFrame(
        {
            "route_id": [f"R-{i % 20:03d}" for i in range(n)],
            "hour": rng.integers(5, 23, n),
            "distance_km": rng.uniform(2, 25, n).round(2),
            "base_delay": rng.normal(4, 2, n).round(2).clip(min=0),
        }
    )
    frame["delay_min"] = (
        frame["base_delay"] + frame["hour"].isin([7, 8, 17, 18]) * 2.5
        + frame["distance_km"] * 0.05 + rng.normal(0, 0.4, n)
    ).clip(lower=0).round(2)
    src = sample_dir / "trip_features.parquet"
    frame.to_parquet(src, index=False)
    print(f"[1] pandas wrote {len(frame)} rows -> {src.name}")

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("utiq-jvm-path-check")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    t0 = time.time()

    df = spark.read.parquet(str(src))
    print(f"[2] spark.read.parquet -> {df.count()} rows")

    df.createOrReplaceTempView("trip_features")
    agg = spark.sql(
        """
        select route_id,
               count(*)        as trips,
               round(avg(delay_min), 3) as avg_delay
        from trip_features
        group by route_id
        order by avg_delay desc
        """
    )
    top = agg.limit(3).collect()
    print(f"[3] Spark SQL aggregation + JVM collect -> {len(top)} rows, e.g. {top[0]}")

    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression

    features = VectorAssembler(
        inputCols=["hour", "distance_km", "base_delay"], outputCol="features"
    )
    model = LinearRegression(featuresCol="features", labelCol="delay_min", maxIter=10)
    pipeline_model = model.fit(features.transform(df))
    summary = pipeline_model.summary
    print(
        f"[4] MLlib LinearRegression fit -> r2={summary.r2:.4f} "
        f"rmse={summary.rootMeanSquaredError:.4f}"
    )

    out = sample_dir / "predictions"
    predictions = pipeline_model.transform(features.transform(df))
    predictions.select("route_id", "hour", "delay_min", "prediction").write.mode(
        "overwrite"
    ).parquet(str(out))
    print(f"[5] JVM wrote prediction parquet -> {out.name}")

    back = spark.read.parquet(str(out)).toPandas()
    print(f"[6] toPandas round-trip -> {len(back)} rows, cols={list(back.columns)}")

    spark.stop()
    print(f"JVM-ONLY SPARK PATH VERIFIED ({time.time() - t0:.1f}s of spark work)")


if __name__ == "__main__":
    main()
