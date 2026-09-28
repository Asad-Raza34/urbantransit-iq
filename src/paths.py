"""Centralised filesystem paths for the project.

Every module should resolve paths through :mod:`src.paths` instead of
building them ad-hoc, so renames and the local-vs-HDFS distinction stay
manageable.
"""

from config.settings import (  # noqa: F401  (re-exported for convenience)
    DATA_ANALYTICS,
    DATA_CLEANED,
    DATA_FEATURES,
    DATA_GENERATED,
    DATA_METADATA,
    DATA_PROCESSED,
    DATA_RAW,
    DATA_ROOT,
    DATA_SAMPLES,
    DOCS_ROOT,
    HDFS_ROOT,
    LOGS_ROOT,
    MODELS_PYTHON,
    MODELS_ROOT,
    MODELS_SPARK,
    MODELS_VERSIONS,
    PROJECT_ROOT,
    REPORTS_ROOT,
    SEGMENT_RULES,
)


def ensure_layout() -> None:
    """Create the standard directory tree idempotently."""
    dirs = [
        DATA_RAW,
        DATA_GENERATED,
        DATA_CLEANED,
        DATA_PROCESSED,
        DATA_METADATA,
        DATA_SAMPLES,
        MODELS_SPARK,
        MODELS_PYTHON,
        MODELS_VERSIONS,
        REPORTS_ROOT,
        LOGS_ROOT,
        "app/pages",
        "app/components",
        "app/charts",
        "app/maps",
        "app/styles",
        "data/raw",
        "data/generated",
        "data/cleaned",
        "data/processed",
        HDFS_ROOT / "raw",
        HDFS_ROOT / "cleaned",
        HDFS_ROOT / "processed",
        HDFS_ROOT / "features",
        HDFS_ROOT / "analytics",
        HDFS_ROOT / "models",
    ]
    for path in dirs:
        Path(path).mkdir(parents=True, exist_ok=True)
    return None


from pathlib import Path  # noqa: E402


def dataset_path(area: str, name: str, ext: str = "parquet") -> Path:
    """Path to a dataset within a data area.

    Parameters
    ----------
    area : str
        One of ``raw``, ``generated``, ``cleaned``, ``processed``.
    name : str
        Logical dataset name (no extension).
    ext : str
        File format extension, one of csv|json|parquet.
    """
    root = {
        "raw": DATA_RAW,
        "generated": DATA_GENERATED,
        "cleaned": DATA_CLEANED,
        "processed": DATA_PROCESSED,
    }[area]
    return root / f"{name}.{ext}"


def report_path(name: str, ext: str = "csv") -> Path:
    return REPORTS_ROOT / f"{name}.{ext}"


def log_path(name: str = "application") -> Path:
    return LOGS_ROOT / f"{name}.log"