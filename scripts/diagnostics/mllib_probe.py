"""Probe 3: full JVM pipeline - parquet read, Spark SQL, MLlib fit/transform,
parquet write, pandas round-trip - with the native-lib fix applied."""
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HADOOP = PROJECT_ROOT / "hadoop"
os.environ.setdefault("JAVA_HOME", "C:/Java/jdk-17.0.20.1+1")
os.environ["HADOOP_HOME"] = str(HADOOP)


def short_path(path: Path) -> str:
    ps = (
        '$fso = New-Object -ComObject Scripting.FileSystemObject;\n'
        f'$fso.GetFolder("{path}").ShortPath'
    )
    return subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


BIN = short_path(HADOOP / "bin").replace("\\", "/")
os.environ["PATH"] = BIN.replace("/", "\\") + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

out_dir = PROJECT_ROOT / "data" / "samples" / "mllib_probe3"
out_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(7)
n = 8000
frame = pd.DataFrame(
    {
        "route_id": [f"R-{i % 12:03d}" for i in range(n)],
        "hour": rng.integers(5, 23, n),
        "distance_km": rng.uniform(2, 25, n).round(2),
        "occupancy": rng.uniform(0.2, 1.1, n).round(3),
    }
)
frame["delay_min"] = (
    2 + frame["hour"].isin([7, 8, 17, 18]) * 3 + frame["distance_km"] * 0.08
    + frame["occupancy"] * 2 + rng.normal(0, 0.5, n)
).clip(lower=0).round(2)
src = out_dir / "trip_features.parquet"
frame.to_parquet(src, index=False)
print(f"[1] wrote {len(frame)} input rows")

spark = (
    SparkSession.builder.master("local[2]")
    .appName("mllib-probe3")
    .config("spark.ui.enabled", "false")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.sql.shuffle.partitions", "2")
    .config("spark.driver.extraJavaOptions", f"-Djava.library.path={BIN}")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(str(src))
print(f"[2] spark read {df.count()} rows")

df.createOrReplaceTempView("tf")
agg = spark.sql(
    "select route_id, count(*) as trips, round(avg(delay_min),3) as avg_delay "
    "from tf group by route_id order by avg_delay desc"
).limit(3).collect()
print(f"[3] SQL agg -> {agg}")

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

va = VectorAssembler(inputCols=["hour", "distance_km", "occupancy"], outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol="delay_min", maxIter=10)
model = lr.fit(va.transform(df))
print(f"[4] LR fit r2={model.summary.r2:.4f}")

preds = model.transform(va.transform(df)).select("route_id", "delay_min", "prediction")
preds.write.mode("overwrite").parquet(str(out_dir / "preds"))
pandas_back = spark.read.parquet(str(out_dir / "preds")).toPandas()
print(f"[5] write+read+toPandas -> {len(pandas_back)} rows {list(pandas_back.columns)}")

spark.stop()
print("FULL_JVM_PATH_OK")