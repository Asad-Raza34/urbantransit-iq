"""HDFS integration layer.

Two operating modes are supported and are never silently mixed:

* ``local``   - writes to the ``hdfs/`` mirror directory. This is the default:
                the dev machine and the lightweight deployment run against the
                mirror, which keeps everything reproducible without a cluster.
* ``hadoop``  - configured by setting ``UTIQ_HADOOP_MODE=hadoop``; datasets are
                then referenced with ``hdfs://`` URIs and the caller is
                responsible for a reachable cluster.

The active mode is exposed through :func:`mode` and reported in logs and in
the storage diagnostics page; nothing pretends local mode is a real cluster.
"""

import os
import subprocess
from pathlib import Path
from typing import Any

from src.config import cfg
from src.log import get_logger
from src.paths import HDFS_ROOT

logger = get_logger(__name__)

_ACTIVE_MODE = None


def mode() -> str:
    """'hadoop' only when explicitly configured, otherwise 'local'."""
    global _ACTIVE_MODE
    if _ACTIVE_MODE is None:
        configured = os.environ.get("UTIQ_HADOOP_MODE", "local").strip().lower()
        _ACTIVE_MODE = "hadoop" if configured == "hadoop" else "local"
        logger.info("storage mode = %s", _ACTIVE_MODE)
    return _ACTIVE_MODE


def logical_uri(area: str, name: str, ext: str = "parquet"):
    """The URI used to reference a dataset in the active environment."""
    if mode() == "hadoop":
        namenode = os.environ.get("UTIQ_HADOOP_NAMENODE", "localhost:9000")
        return f"hdfs://{namenode}/{area}/{name}.{ext}"
    return (HDFS_ROOT / area / f"{name}.{ext}").as_posix()


def exists(area: str, name: str, ext: str = "parquet") -> bool:
    """Check if a dataset exists in the active storage."""
    uri = logical_uri(area, name, ext)
    if mode() == "hadoop":
        return _hdfs_exists(uri)
    return Path(uri).exists()


def _hdfs_exists(uri: str) -> bool:
    """Check if a path exists in HDFS using hdfs dfs -test."""
    try:
        result = subprocess.run(
            ["hdfs", "dfs", "-test", "-e", uri],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning("HDFS exists check failed for %s: %s", uri, e)
        return False


def list_area(area: str) -> list[str]:
    """Names of datasets present in an HDFS area."""
    if mode() == "local":
        root = HDFS_ROOT / area
    else:
        return _hdfs_list_area(area)
    if not root.exists():
        return []
    return sorted(p.stem for p in root.iterdir() if p.is_file() and not p.suffix == ".crc")


def _hdfs_list_area(area: str) -> list[str]:
    """List files in an HDFS area using hdfs dfs -ls."""
    namenode = os.environ.get("UTIQ_HADOOP_NAMENODE", "localhost:9000")
    uri = f"hdfs://{namenode}/{area}"
    try:
        result = subprocess.run(
            ["hdfs", "dfs", "-ls", uri],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return []
        files = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 8 and not parts[0].startswith("d"):
                fname = parts[-1].split("/")[-1]
                files.append(fname.replace(".parquet", "").replace(".csv", "").replace(".json", ""))
        return sorted(files)
    except Exception as e:
        logger.warning("HDFS list failed for %s: %s", uri, e)
        return []


def status() -> dict:
    """Diagnostic summary of the storage layer (used by the dashboard)."""
    areas = {}
    for area in cfg("HDFS_AREAS"):
        areas[area] = list_area(area)
    return {
        "mode": mode(),
        "areas": areas,
        "root": str(HDFS_ROOT),
        "hdfs_available": _check_hdfs_connection(),
    }


def _check_hdfs_connection() -> bool:
    """Check if HDFS is accessible in hadoop mode."""
    if mode() != "hadoop":
        return False
    try:
        result = subprocess.run(
            ["hdfs", "dfs", "-test", "-e", "/"],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False


def bin_available() -> bool:
    """True when winutils/hadoop.dll exist for the local native libs."""
    bindir = cfg("HADOOP_HOME") / "bin" if cfg("HADOOP_HOME") else None
    return bool(bindir and (bindir / "winutils.exe").exists() and (bindir / "hadoop.dll").exists())


def windows_short_path(path: str | Path) -> str:
    """Return the 8.3 short path for a Windows folder (no spaces allowed).

    Spark's launcher splits ``-D`` values on whitespace, so any native library
    path must be space-free.
    """
    ps = (
        "$fso = New-Object -ComObject Scripting.FileSystemObject;\n"
        f'$fso.GetFolder("{path}").ShortPath'
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return out.stdout.strip().replace("\\", "/")
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not compute short path for %s: %s", path, exc)
        return str(path).replace("\\", "/")


def read_hdfs_parquet(uri: str) -> "pd.DataFrame | None":
    """Read a Parquet file from HDFS using Spark (requires Spark session)."""
    try:
        from src.spark_context import spark_session
        spark = spark_session()
        return spark.read.parquet(uri).toPandas()
    except Exception as e:
        logger.error("Failed to read HDFS parquet %s: %s", uri, e)
        return None


def write_hdfs_parquet(df: "pd.DataFrame", uri: str) -> bool:
    """Write a DataFrame to HDFS as Parquet using Spark."""
    try:
        from src.spark_context import spark_session
        spark = spark_session()
        spark_df = spark.createDataFrame(df)
        spark_df.write.mode("overwrite").parquet(uri)
        return True
    except Exception as e:
        logger.error("Failed to write HDFS parquet %s: %s", uri, e)
        return False


def test_hdfs_connection() -> dict[str, Any]:
    """Test HDFS connection and return diagnostic info."""
    results = {
        "mode": mode(),
        "hdfs_mode": False,
        "connection_ok": False,
        "namenode": None,
        "version": None,
        "areas": {},
        "error": None,
        "real_cluster_verified": False,
        "verification_note": "Real HDFS cluster verification not available in current environment. Local mirror mode is fully functional. To verify real cluster: set UTIQ_HADOOP_MODE=hadoop, configure UTIQ_HADOOP_NAMENODE, and ensure Hadoop cluster is accessible."
    }

    if mode() != "hadoop":
        results["error"] = "Not in hadoop mode (UTIQ_HADOOP_MODE != hadoop)"
        return results

    results["hdfs_mode"] = True
    results["namenode"] = os.environ.get("UTIQ_HADOOP_NAMENODE", "localhost:9000")

    try:
        result = subprocess.run(
            ["hdfs", "version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            results["version"] = result.stdout.split("\n")[0].strip()
        else:
            results["error"] = f"hdfs version failed: {result.stderr}"
            return results
    except Exception as e:
        results["error"] = f"hdfs command not found: {e}"
        return results

    results["connection_ok"] = _check_hdfs_connection()

    if results["connection_ok"]:
        for area in cfg("HDFS_AREAS"):
            results["areas"][area] = _hdfs_list_area(area)
        # If we get here in hadoop mode with connection_ok, consider it verified
        results["real_cluster_verified"] = True
        results["verification_note"] = "Real HDFS cluster connection verified successfully."

    return results


if __name__ == "__main__":
    print(json.dumps(test_hdfs_connection(), indent=2))