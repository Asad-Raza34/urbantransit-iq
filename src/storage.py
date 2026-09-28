"""Dataframe IO helpers shared by the pipeline and the app.

Covers the three supported formats (CSV / JSON / Parquet), atomic writes and
the logical-HDFS mirroring used when Hadoop is not available.
"""

import os
import tempfile
import time
from pathlib import Path

import pandas as pd

from config.settings import HDFS_AREAS
from src.log import get_logger
from src.paths import HDFS_ROOT

logger = get_logger(__name__)


def _replace_with_retry(source: Path, target: Path, attempts: int = 6) -> None:
    """Atomic rename with retries for Windows file-lock transients."""
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.3 + attempt * 0.2)


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read a dataset based on its file extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".json", ".jsonl"):
            return pd.read_json(path, lines=True, orient="records") if suffix == ".jsonl" else pd.read_json(path)
        if suffix == ".xlsx":
            return pd.read_excel(path)
    except Exception as exc:
        raise ValueError(f"Failed to read {path}: {exc}") from exc
    raise ValueError(f"Unsupported dataset format: {suffix}")


def make_spark_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a pandas frame into a copy Spark can consume directly.

    pandas writes parquet timestamps as INT64(TIMESTAMP_NANOS), which Spark 3.5
    refuses to open. Datetime columns are therefore exported as epoch-microsecond
    integers; everything else is left untouched.
    """
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = (out[column].astype("int64") // 1000).astype("int64")
    return out


def write_dataset(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Write a dataframe to CSV/JSON/Parquet.

    The write goes to a temporary file first and is then renamed into place so
    a failed write can never leave a half-written dataset behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=suffix)
    os.close(fd)  # release the handle immediately; Windows locks rename/delete otherwise
    try:
        if suffix == ".parquet":
            df.to_parquet(tmp, index=index)
        elif suffix == ".csv":
            df.to_csv(tmp, index=index)
        elif suffix == ".json":
            df.to_json(tmp, orient="records", lines=True, index=index)
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}")
        _replace_with_retry(Path(tmp), path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return path


def mirror_to_hdfs(df: pd.DataFrame, area: str, name: str, formats=("parquet",)) -> dict[str, Path]:
    """Store a dataset in the local HDFS mirror layout.

    ``area`` must be one of the HDFS logical areas defined in settings. The
    mirrors physically live under ``hdfs/<area>/<name>.<ext>`` so the rest of
    the project can treat them as the HDFS representation.
    """
    if area not in HDFS_AREAS:
        raise ValueError(f"Unknown HDFS area: {area} (valid: {list(HDFS_AREAS)})")

    if isinstance(formats, str):
        formats = (formats,)
    written = []
    for ext in formats:
        path = HDFS_ROOT / area / f"{name}.{ext}"
        write_dataset(df, path)
        written.append(path)
    logger.info("mirrored %s/%s -> %s", area, name, [str(p) for p in written])
    return {"paths": written, "rows": len(df)}


def dataset_exists(area: str, name: str, ext: str = "parquet") -> bool:
    return (HDFS_ROOT / area / f"{name}.{ext}").exists()