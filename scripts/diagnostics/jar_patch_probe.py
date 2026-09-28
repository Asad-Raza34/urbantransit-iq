"""Probe: can Spark write parquet on Windows once the hadoop 3.3.6 patch jars
are put on the classpath in front of the jars bundled with Spark 3.5.3?"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("JAVA_HOME", "C:/Java/jdk-17.0.20.1+1")
os.environ["HADOOP_HOME"] = str(PROJECT_ROOT / "hadoop")
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

patch = PROJECT_ROOT / "hadoop" / "jars-patch"
jars = [
    str(patch / "hadoop-client-api-3.3.6.jar"),
    str(patch / "hadoop-client-runtime-3.3.6.jar"),
]

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.master("local[2]")
    .appName("jar-patch-probe")
    .config("spark.ui.enabled", "false")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.sql.shuffle.partitions", "2")
    .config("spark.jars", ",".join(jars))
    .config("spark.jars.driverClassPath", ",".join(jars))
    .config("spark.executor.extraClassPath", ";".join(jars))
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

try:
    spark.sql("select 1 as a, 'x' as b").write.mode("overwrite").parquet(
        str(PROJECT_ROOT / "data" / "samples" / "jar_probe_out")
    )
    back = spark.read.parquet(str(PROJECT_ROOT / "data" / "samples" / "jar_probe_out")).collect()
    print("WRITE_OK", back)
finally:
    spark.stop()