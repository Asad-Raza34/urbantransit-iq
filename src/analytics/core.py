"""Shared helpers for the analytics package.

Keeps the individual analytic modules focused on computation: loading is done
through the storage layer, thresholds come from ``config.settings``, and the
outputs are written atomically and mirrored to the HDFS layout with one
consistent contract (every frame has an ``analytics_version`` column that says
which rules produced it).
"""

import time

import pandas as pd

from config import settings
from src.audit import record
from src.log import get_logger
from src.paths import DATA_ANALYTICS, DATA_PROCESSED, DATA_FEATURES, DATA_CLEANED
from src.storage import mirror_to_hdfs, read_dataset, write_dataset

logger = get_logger(__name__)

ANALYTICS_VERSION = "1.0"


def read_processed(name: str) -> pd.DataFrame:
    return read_dataset(DATA_PROCESSED / f"{name}.parquet")


def read_features(name: str) -> pd.DataFrame:
    return read_dataset(DATA_FEATURES / f"{name}.parquet")


def read_clean(name: str) -> pd.DataFrame:
    return read_dataset(DATA_CLEANED / f"{name}.parquet")


def _tag(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["analytics_version"] = ANALYTICS_VERSION
    return frame


def save(name: str, frame: pd.DataFrame, index: bool = False,
         summary_rows: int | None = None) -> pd.DataFrame:
    """Write + mirror one analytics frame, returning the tagged copy."""
    frame = _tag(frame)
    path = DATA_ANALYTICS / f"{name}.parquet"
    write_dataset(frame, path, index=index)
    mirror_to_hdfs(frame, "analytics", name)
    logger.info("analytics[%s]: %d rows", name, len(frame))
    return frame


def weighted_score(parts: dict[str, float], weights: dict[str, float]) -> float:
    """Composite 0..100 score from named components and their weights."""
    total = sum(weights.get(k, 0.0) * max(0.0, min(100.0, v)) for k, v in parts.items())
    return round(total, 1)


def clip100(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=100.0)


def analytics_summary() -> dict:
    """Row counts of every analytics frame (used by the dashboard + audit)."""
    if not DATA_ANALYTICS.exists():
        return {}
    counts = {}
    for path in sorted(DATA_ANALYTICS.glob("*.parquet")):
        counts[path.stem] = int(len(read_dataset(path)))
    return counts


def run_blocked(name: str, fn):
    """Run one analytics step with timing + audit entry."""
    t0 = time.perf_counter()
    result = fn()
    record(
        action=f"analytics:{name}",
        component="analytics",
        status="ok",
        details=f"rows={result}" if isinstance(result, int) else "",
        duration_ms=(time.perf_counter() - t0) * 1000,
    )
    return result