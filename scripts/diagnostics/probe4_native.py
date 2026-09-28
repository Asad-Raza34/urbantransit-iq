"""Probe 4: isolate the NativeIO problem - check java.library.path from inside
the JVM, try to load the hadoop native library, and test write committers."""
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


BIN = short_path(HADOOP / "bin")
os.environ["PATH"] = BIN + os.pathsep + os.environ.get("PATH", "")

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

out_dir = PROJECT_ROOT / "data" / "samples" / "probe4"
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
n = 8000
frame = pd.DataFrame(
    {
        "route_id": [f"R-{i % 12:03d}" for i in range(n)],
        "hour": rng.integers(5, 23, n),
        "delay_min": rng.uniform(0, 15, n).round(2),
    }
)
src = out_dir / "trip_features.parquet"
frame.to_parquet(src, index=False)

spark = (
    SparkSession.builder.master("local[2]")
    .appName("probe4")
    .config("spark.ui.enabled", "false")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.sql.shuffle.partitions", "2")
    .config("spark.driver.extraJavaOptions", f"-Djava.library.path={BIN}")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

jvm = spark._jvm
print("java.library.path =", jvm.System.getProperty("java.library.path"))

ni = jvm.org.apache.hadoop.io.nativeio.NativeIO
print("NativeIO.isAvailable =", ni.isAvailable())
try:
    print("NativeIO.getLinkCount dir =", ni.getLinkCount(str(out_dir)))
except Exception as e:  # noqa: BLE001
    print("getLinkCount failed:", type(e).__name__, str(e)[:200])

try:
    jvm.java.lang.System.loadLibrary("hadoop")
    print("System.loadLibrary(hadoop) = OK")
except Exception as e:  # noqa: BLE001
    print("System.loadLibrary(hadoop) failed:", str(e)[:200])

df = spark.read.parquet(str(src)).repartition(2)
try:
    df.write.mode("overwrite").parquet(str(out_dir / "baseline"))
    print("BASELINE_WRITE_OK", spark.read.parquet(str(out_dir / "baseline")).count())
except Exception as e:  # noqa: BLE001
    msg = str(e)
    print("BASELINE_FAIL", "access0" in msg, msg[:120])

spark.stop()