"""Probe 2: put hadoop/bin on PATH and java.library.path for the Spark JVM.

The project root contains a space ("UrbanTransit IQ"), which Spark's launcher
mis-parses inside -D options, so the native library path is passed in its
Windows 8.3 short form (no spaces).
"""
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HADOOP = PROJECT_ROOT / "hadoop"


def short_path(path: Path) -> str:
    ps = (
        '$fso = New-Object -ComObject Scripting.FileSystemObject;\n'
        f'$fso.GetFolder("{path}").ShortPath'
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


BIN = Path(short_path(HADOOP / "bin"))

os.environ.setdefault("JAVA_HOME", "C:/Java/jdk-17.0.20.1+1")
os.environ["HADOOP_HOME"] = str(HADOOP)
os.environ["PATH"] = str(BIN) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.master("local[2]")
    .appName("path-probe2")
    .config("spark.ui.enabled", "false")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.sql.shuffle.partitions", "2")
    .config(
        "spark.driver.extraJavaOptions",
        f"-Djava.library.path={BIN.as_posix()}",
    )
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

try:
    spark.sql("select 1 as a, 'x' as b").write.mode("overwrite").parquet(
        str(PROJECT_ROOT / "data" / "samples" / "path_probe2_out")
    )
    rows = spark.read.parquet(str(PROJECT_ROOT / "data" / "samples" / "path_probe2_out")).collect()
    print("WRITE_OK", rows)
finally:
    spark.stop()