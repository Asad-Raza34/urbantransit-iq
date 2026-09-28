"""Spark-native reliability analytics.

JVM-only Spark job (no python workers): reads timestamp-safe copies of the
integrated frames and computes route reliability and stop punctuality entirely
with Spark SQL, demonstrating the platform's large-data path on the ~6.6M-row
``stop_facts`` layer.

Two frames are produced and saved through the standard analytics contract
(``analytics_version`` tag + HDFS mirror):

* ``spark_route_reliability`` - per-route reliability profile with a
  healthy/degraded/critical flag (thresholds from ``config.settings``);
* ``spark_stop_punctuality``  - per (route, time-band) arrival-delay and load
  picture aggregated inside Spark.

Spark 3.5 cannot read pandas' INT64(TIMESTAMP_NANOS) parquet type, so the
``scheduled_/actual_*`` dt columns are exported as epoch-microsecond integers
into a cache under ``data/analytics/_spark`` before hand-over to Spark.
"""

import hashlib
import json
import time

import pandas as pd

from config import settings
from src.analytics.core import save
from src.audit import record
from src.log import get_logger
from src.paths import DATA_ANALYTICS
from src.spark_context import spark_available, spark_session
from src.storage import make_spark_safe

logger = get_logger(__name__)

SPARK_CACHE = DATA_ANALYTICS / "cache" / "spark"


def _cache_path(name: str) -> tuple:
    return SPARK_CACHE / f"{name}.parquet", SPARK_CACHE / f"{name}.meta.json"


def _refresh_if_needed(frame: pd.DataFrame, name: str, source_path) -> str:
    """Write a Spark-safe parquet cache once; reuse it while inputs match."""
    SPARK_CACHE.mkdir(parents=True, exist_ok=True)
    pq_path, meta_path = _cache_path(name)
    source = None if source_path is None else source_path.stat()
    digest = hashlib.md5(
        json.dumps(
            {"cols": list(frame.columns), "dtypes": [str(t) for t in frame.dtypes],
             "n": len(frame),
             "src": None if source_path is None else source.st_size,
             "src_mtime": None if source_path is None else int(source.st_mtime_ns)},
            sort_keys=True,
        ).encode()
    ).hexdigest()

    if meta_path.exists() and pq_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("digest") == digest:
                return str(pq_path)
        except (json.JSONDecodeError, OSError):
            pass

    safe = make_spark_safe(frame)
    safe.to_parquet(pq_path, index=False)
    meta_path.write_text(json.dumps({"digest": digest, "n": len(safe)}, indent=2),
                         encoding="utf-8")
    return str(pq_path)


def _reliability(frame: pd.DataFrame, source_path) -> pd.DataFrame:
    """Route reliability profile via Spark SQL over the trip layer."""
    spark = spark_session()
    path = _refresh_if_needed(frame, "trips", source_path)
    spark.read.parquet(path).createOrReplaceTempView("trips")

    rows = spark.sql(
        """
        SELECT
            route_id,
            COUNT(*)                                           AS trips,
            COUNT(DISTINCT date)                               AS active_days,
            ROUND(COALESCE(AVG(departure_delay_min), 0.0), 2)  AS avg_delay_min,
            ROUND(COALESCE(PERCENTILE(departure_delay_min, 0.9), 0.0), 2) AS p90_delay_min,
            ROUND(100.0 * SUM(CASE WHEN on_time THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                              AS on_time_rate,
            ROUND(100.0 * SUM(CASE WHEN delay_severity IN ('severe', 'critical')
                                   THEN 1 ELSE 0 END) / COUNT(*), 2) AS severe_rate,
            ROUND(COALESCE(STDDEV_POP(scheduled_headway_min), 0.0), 2) AS headway_std,
            ROUND(100.0 * SUM(CASE WHEN scheduled_headway_min > 2
                                    AND actual_headway_min < 0.5 * scheduled_headway_min
                                   THEN 1 ELSE 0 END)
                  / NULLIF(SUM(CASE WHEN scheduled_headway_min > 2 THEN 1 ELSE 0 END), 0),
                  2)                                          AS bunching_rate,
            ROUND(AVG(occupancy_max_pct), 1)                   AS avg_occupancy_max_pct
        FROM trips
        GROUP BY route_id
        ORDER BY route_id
        """
    )
    out = rows.toPandas()
    out = out.astype({col: "float64" for col in out.columns
                      if col not in ("route_id", "time_band")})
    out["reliability_flag"] = out["on_time_rate"].map(
        lambda r: ("critical" if r <= settings.RELIABILITY_CRITICAL * 100
                   else "degraded" if r <= settings.RELIABILITY_DEGRADED * 100
                   else "healthy"))
    out["delay_score"] = out["avg_delay_min"].map(
        lambda d: round(max(0.0, min(100.0, 100.0 - d * (100.0 / 30.0))), 1))
    out["quality_score"] = (0.7 * out["on_time_rate"] + 0.3 * out["delay_score"]).round(1)
    return out


def _stop_punctuality(frame: pd.DataFrame, source_path) -> pd.DataFrame:
    """Per (route, time-band) arrival-delay and load stats via Spark SQL."""
    spark = spark_session()
    path = _refresh_if_needed(frame, "stops", source_path)
    spark.read.parquet(path).createOrReplaceTempView("stops")

    rows = spark.sql(
        """
        SELECT
            route_id,
            time_band,
            COUNT(*)                                          AS observations,
            ROUND(COALESCE(AVG(arr_delay_min), 0.0), 2)       AS avg_arr_delay_min,
            ROUND(100.0 * SUM(CASE WHEN arr_delay_min > 5 THEN 1 ELSE 0 END) / COUNT(*),
                  2)                                          AS late_share,
            ROUND(AVG(onboard_after), 1)                      AS avg_onboard,
            ROUND(AVG(occupancy_after), 3)                    AS avg_occupancy
        FROM stops
        GROUP BY route_id, time_band
        ORDER BY route_id, time_band
        """
    )
    return rows.toPandas()


def build() -> dict:
    """Run both Spark SQL analytics and persist the results."""
    ok, msg = spark_available()
    if not ok:
        logger.warning("spark analytics skipped: %s", msg)
        return {"status": "skipped", "message": msg}

    from src.analytics.core import read_processed
    from src.paths import DATA_PROCESSED

    t0 = time.perf_counter()
    trip_facts = read_processed("trip_facts")
    stop_facts = read_processed("stop_facts")

    route_reliability = save("spark_route_reliability",
                             _reliability(trip_facts, DATA_PROCESSED / "trip_facts.parquet"))
    stop_punctuality = save("spark_stop_punctuality",
                            _stop_punctuality(stop_facts, DATA_PROCESSED / "stop_facts.parquet"))

    record(
        action="analytics:spark",
        component="analytics",
        status="ok",
        details=json.dumps({
            "route_reliability": int(len(route_reliability)),
            "stop_punctuality": int(len(stop_punctuality)),
        }),
        duration_ms=(time.perf_counter() - t0) * 1000,
    )
    return {
        "spark_route_reliability": int(len(route_reliability)),
        "spark_stop_punctuality": int(len(stop_punctuality)),
    }


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()