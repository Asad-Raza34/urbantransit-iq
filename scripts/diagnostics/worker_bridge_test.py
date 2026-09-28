"""Minimal python-worker bridge test (self-contained environment repair).

Git Bash / MSYS hands child processes a mangled PATH, and the Spark driver JVM
passes its environment to every python worker it spawns. Rather than trusting
the inherited PATH, this script constructs a clean one before the JVM starts.

Success line: WORKER_BRIDGE_OK <n>
"""
import os
import sys


def build_clean_path() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    entries = [
        os.path.join(project_root, "hadoop", "bin"),
        r"C:\Java\jdk-17.0.20.1+1\bin",
        os.path.join(sys.prefix, "Library", "bin"),
        os.path.join(sys.prefix, "Scripts"),
        r"C:\Windows\system32",
        r"C:\Windows",
        r"C:\Windows\System32\Wbem",
    ]
    return os.pathsep.join(dict.fromkeys(entries))  # dedupe, keep order


def prepare_environment() -> None:
    os.environ["PATH"] = build_clean_path()
    os.environ.setdefault("JAVA_HOME", r"C:\Java\jdk-17.0.20.1+1")
    os.environ.setdefault("HADOOP_HOME", r"C:\Users\HP\Desktop\UrbanTransit IQ\hadoop")
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def main() -> None:
    prepare_environment()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("utiq-worker-bridge-test")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        rows = spark.createDataFrame([(1, "a"), (2, "b")], ["i", "g"]).count()
        print(f"WORKER_BRIDGE_OK {rows}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
