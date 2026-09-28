"""Spark session management.

Encapsulates the Windows-specific setup that makes Spark usable here:

* JAVA_HOME discovery (JDK 17 installed under C:\\Java).
* HADOOP_HOME pointing at the bundled winutils/hadoop.dll.
* The native library path passed to the JVM in *short* (space-free) form.
* The python-worker bridge pinned to the running conda interpreter
  (``spark.pyspark.python``), which is what makes MLlib model persistence and
  small ``toPandas`` calls work at all on Windows.

:func:`spark_available` is the single source of truth used everywhere else;
call it once per process and cache the result.
"""

import os
import sys
import threading
from glob import glob
from pathlib import Path

from src.config import settings
from src.log import get_logger

logger = get_logger(__name__)

_SPARK_SESSION = None
_AVAILABILITY = None
_lock = threading.Lock()


def _discover_java_home() -> Path | None:
    env = os.environ.get("JAVA_HOME")
    if env and Path(env).exists():
        return Path(env)
    candidates = sorted(glob("C:/Java/jdk-*"))
    return Path(candidates[-1]) if candidates else None


def _configure_windows_native(short_bin: str | None) -> str | None:
    """Make sure the JVM can load hadoop.dll, returning the java.library.path."""
    if sys.platform != "win32":
        return None
    from src.hdfs_store import windows_short_path

    hadoop_home = Path(settings.HADOOP_HOME)
    original_bin = hadoop_home / "bin"
    bin_dir = short_bin or windows_short_path(original_bin)
    os.environ["PATH"] = bin_dir.replace("/", "\\") + os.pathsep + os.environ.get("PATH", "")
    return bin_dir


def spark_session():
    """Return the shared SparkSession, creating it on first use."""
    global _SPARK_SESSION
    with _lock:
        if _SPARK_SESSION is not None:
            return _SPARK_SESSION
        _SPARK_SESSION = _build_session()
        return _SPARK_SESSION


def _build_session():
    from pyspark.sql import SparkSession

    java_home = _discover_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = str(java_home)
        os.environ["PATH"] = str(java_home / "bin") + os.pathsep + os.environ.get("PATH", "")

    os.environ["HADOOP_HOME"] = str(settings.HADOOP_HOME)
    native_path = _configure_windows_native(settings.SPARK_NATIVE_BIN or None)

    # Windows gotcha: the python worker bridge starts ``python`` from PATH,
    # which resolves to the broken Microsoft Store stub. Pin the interpreter
    # to the same conda env Python currently running so worker-based APIs
    # (model load/save, pandas conversions) work deterministically. The worker
    # inherits no PYTHONPATH from the driver, so hand it the pyspark/py4j
    # locations on both the inherited env and the JVM ``python.path`` property.
    worker_python = sys.executable.replace("\\", "/")
    python_dir = str(Path(sys.executable).parent)
    os.environ["PATH"] = python_dir + os.pathsep + os.environ.get("PATH", "")
    import py4j  # noqa: PLC0415
    import pyspark  # noqa: PLC0415

    worker_pythonpath = os.pathsep.join(
        [os.environ.get("PYTHONPATH", ""), pyspark.__path__[0],
         str(Path(py4j.__file__).parent.parent), str(settings.PROJECT_ROOT)])
    os.environ["PYTHONPATH"] = worker_pythonpath

    builder = (
        SparkSession.builder.master(settings.SPARK_MASTER)
        .appName(settings.SPARK_APP_NAME)
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.pyspark.python", worker_python)
        .config("spark.pyspark.driver.python", worker_python)
        .config("spark.python.worker.reuse", "false")
        .config("spark.sql.shuffle.partitions", str(settings.SPARK_SHUFFLE_PARTITIONS))
        .config("spark.driver.memory", settings.SPARK_DRIVER_MEMORY)
    )
    if native_path:
        java_opts = settings.SPARK_JAVA_OPTS
        java_opts = java_opts + f" -Djava.library.path={native_path}".strip() if java_opts else f"-Djava.library.path={native_path}"
        builder = builder.config("spark.driver.extraJavaOptions", java_opts)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    logger.info("Spark session ready (master=%s, hadoop_home=%s)", settings.SPARK_MASTER, settings.HADOOP_HOME)
    return spark


def spark_available() -> tuple[bool, str]:
    """Probe whether a usable SparkSession can be created.

    Returns ``(ok, message)``. The probe is executed once per process.
    """
    global _AVAILABILITY
    with _lock:
        if _AVAILABILITY is not None:
            return _AVAILABILITY
    try:
        spark = spark_session()
        row = spark.sql("SELECT 1 AS ok").collect()
        ok = bool(row and row[0][0] == 1)
        msg = "Spark ready" if ok else "Spark session returned no result"
    except Exception as exc:  # noqa: BLE001
        ok, msg = False, f"Spark unavailable: {type(exc).__name__}: {str(exc)[:300]}"
        _stop_session()
    with _lock:
        _AVAILABILITY = (ok, msg)
    return ok, msg


def _stop_session():
    global _SPARK_SESSION, _AVAILABILITY
    if _SPARK_SESSION is not None:
        try:
            _SPARK_SESSION.stop()
        except Exception:  # noqa: BLE001
            pass
    _SPARK_SESSION = None
    _AVAILABILITY = None


def spark_status() -> dict:
    """Human/dashboard friendly summary of the Spark environment."""
    ok, msg = spark_available()
    return {
        "available": ok,
        "message": msg,
        "master": settings.SPARK_MASTER,
        "java_home": os.environ.get("JAVA_HOME", ""),
        "hadoop_home": settings.HADOOP_HOME,
    }


class JobMonitor:
    """Captures real job/stage information from the Spark status tracker.

    The numbers come directly from the JVM status tracker (job status, stage
    task counts) so the monitoring page reflects what Spark actually did.
    """

    def __init__(self, spark):
        self._tracker = spark.sparkContext.statusTracker()

    def recent(self, limit: int = 200):
        jobs = []
        for job_id in self._tracker.getJobIdsForGroup():
            info = self._tracker.getJobInfo(job_id)
            if info is None:
                continue
            stages = []
            for stage_id in info.stageIds:
                stage = self._tracker.getStageInfo(stage_id)
                if stage is not None:
                    stages.append(
                        {
                            "stage_id": stage.stageId,
                            "tasks": stage.numTasks,
                            "completed": stage.numCompletedTasks,
                            "failed": stage.numFailedTasks,
                            "active": stage.numActiveTasks,
                        }
                    )
            jobs.append(
                {
                    "job_id": int(info.jobId),
                    "status": info.status,
                    "num_stages": len(stages),
                    "stages": stages,
                }
            )
        jobs.sort(key=lambda row: row["job_id"], reverse=True)
        return jobs[:limit]


__all__ = ["spark_session", "spark_available", "spark_status", "JobMonitor"]