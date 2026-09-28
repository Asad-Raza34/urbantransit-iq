"""
UrbanTransit IQ - central application settings.

This module is intentionally dependency-free (stdlib only) so it can be
imported by every part of the project, including bootstrap scripts, without
triggering heavy imports. Values fall back to environment variables where the
developer may reasonably want to override defaults (see the `_env` helper).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Populate ``os.environ`` from a ``.env`` file using only the stdlib.

    The README instructs users to copy ``.env.example`` to ``.env``, but nothing
    used to read that file, so the settings it documents had no effect. Real
    environment variables always win over the file. No third-party dependency is
    required (keeps ``requirements.txt`` unchanged).
    """
    if not path.exists():
        return
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        if isinstance(default, bool):
            return value.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
    except ValueError:
        pass
    return value


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_GENERATED = DATA_ROOT / "generated"
DATA_CLEANED = DATA_ROOT / "cleaned"
DATA_PROCESSED = DATA_ROOT / "processed"
DATA_FEATURES = DATA_ROOT / "features"
DATA_ANALYTICS = DATA_ROOT / "analytics"
DATA_SAMPLES = DATA_ROOT / "samples"
DATA_METADATA = DATA_ROOT / "metadata"

MODELS_ROOT = PROJECT_ROOT / "models"
MODELS_SPARK = MODELS_ROOT / "spark"
MODELS_PYTHON = MODELS_ROOT / "python"
MODELS_VERSIONS = MODELS_ROOT / "versions"

REPORTS_ROOT = PROJECT_ROOT / "reports"
LOGS_ROOT = PROJECT_ROOT / "logs"
DOCS_ROOT = PROJECT_ROOT / "docs"

# Local mirror of the HDFS logical layout. When Hadoop is unavailable this is
# what the pipeline writes to; when Hadoop IS available the same logical areas
# are used against an hdfs:// filesystem instead (see hdfs_store).
HDFS_ROOT = PROJECT_ROOT / "hdfs"
HDFS_AREAS = {
    "raw": "raw",
    "cleaned": "cleaned",
    "processed": "processed",
    "features": "features",
    "analytics": "analytics",
    "models": "models",
}

# ---------------------------------------------------------------------------
# Spark / Hadoop environment
# ---------------------------------------------------------------------------

SPARK_MASTER = _env("UTIQ_SPARK_MASTER", "local[2]")
SPARK_APP_NAME = "UrbanTransitIQ"
SPARK_DRIVER_MEMORY = _env("UTIQ_SPARK_DRIVER_MEMORY", "2g")
SPARK_SHUFFLE_PARTITIONS = _env("UTIQ_SPARK_PARTITIONS", 4)
SPARK_JAVA_OPTS = _env("UTIQ_SPARK_JAVA_OPTS", "")

HADOOP_HOME = Path(_env("UTIQ_HADOOP_HOME", PROJECT_ROOT / "hadoop"))

# If HADOOP_HOME/bin is missing its hadoop.dll/winutils, Spark writes fail on
# Windows. The short (8.3) path of the bin directory is used because Spark's
# launcher mis-parses path arguments that contain spaces.
SPARK_NATIVE_BIN = _env("UTIQ_SPARK_NATIVE_BIN", "")

# ---------------------------------------------------------------------------
# Data generation scale
# ---------------------------------------------------------------------------

GEN_SEED = _env("UTIQ_GEN_SEED", 2024)
GEN_MONTHS = _env("UTIQ_GEN_MONTHS", 12)

GEN_SCALE = {
    "routes": _env("UTIQ_GEN_ROUTES", 100),
    "stops": _env("UTIQ_GEN_STOPS", 500),
    "vehicles": _env("UTIQ_GEN_VEHICLES", 250),
    "passengers": _env("UTIQ_GEN_PASSENGERS", 50000),
}

# Errors/bandwidth available in the raw data. These are introduced on purpose
# and are expected to be resolved by the cleaning layer.
QUALITY = {
    "missing_rate": _env("UTIQ_QUALITY_MISSING", 0.008),
    "duplicate_rate": _env("UTIQ_QUALITY_DUP", 0.005),
    "invalid_timestamp_rate": _env("UTIQ_QUALITY_BADTS", 0.0015),
    "corrupt_id_rate": _env("UTIQ_QUALITY_BADID", 0.004),
    "invalid_value_rate": _env("UTIQ_QUALITY_BADVAL", 0.006),
}

# ---------------------------------------------------------------------------
# Analytics thresholds. Kept in one place so the business rules are easy to
# review and adjust rather than scattered across the code base.
# ---------------------------------------------------------------------------

# Crowding is expressed as occupancy ratio (onboard / vehicle capacity).
CROWDING_HIGH = _env("UTIQ_CROWDING_HIGH", 0.85)
CROWDING_CRITICAL = _env("UTIQ_CROWDING_CRITICAL", 1.0)
PERSISTENT_CROWDING_MIN_DAYS = _env("UTIQ_PERSIST_CROWDING_DAYS", 15)

# Delay severity buckets (minutes). The integrator, the analytics modules and
# the delay-classification model all cut the severity scale at these points.
DELAY_ON_TIME_MAX = 5.0
DELAY_MODERATE_MAX = 9.0
DELAY_CRITICAL_MAX = 30.0

# Reliability: share of trips considered on time (delay <= DELAY_ON_TIME_MAX).
RELIABILITY_DEGRADED = _env("UTIQ_RELIABILITY_DEGRADED", 0.75)
RELIABILITY_CRITICAL = _env("UTIQ_RELIABILITY_CRITICAL", 0.55)

# Underutilisation (a trip is underused when even its fullest moment stays
# below this occupancy ratio: measured on occupancy_max, since average
# occupancy is naturally low on any system with peaks and would flag
# nearly every band). Applied per (route, time-band) in the crowding module.
UNDERUTILIZATION_MAX = _env("UTIQ_UNDERUTIL_MAX", 0.25)

# Anomaly detection uses z-score magnitude.
ANOMALY_ZSCORE = _env("UTIQ_ANOMALY_Z", 2.5)

# Demand forecast horizon (days) and seasonal period for daily series.
FORECAST_HORIZON_DAYS = _env("UTIQ_FORECAST_HORIZON", 28)
FORECAST_SEASONAL_PERIOD = 7

# Route scoring weights. Must sum to 1.0.
ROUTE_SCORE_WEIGHTS = {
    "demand": 0.20,
    "reliability": 0.25,
    "occupancy": 0.15,
    "delay": 0.15,
    "adherence": 0.10,
    "bunching": 0.15,
}

# Route classification bands (0..100 scale for the composite score).
ROUTE_CLASS_BANDS = [
    ("Top Performer", 80.0, 100.1),
    ("Solid", 65.0, 80.0),
    ("Needs Attention", 50.0, 65.0),
    ("Action Required", 0.0, 50.0),
]

# Fare model: base charge plus per-km rate, discounted by ticket type.
FARE_BASE = _env("UTIQ_FARE_BASE", 18.0)
FARE_PER_KM = _env("UTIQ_FARE_PER_KM", 4.2)
FARE_DISCOUNT = {
    "single": 1.0,
    "day": 0.75,
    "weekly": 0.50,
    "monthly": 0.35,
}

# Passenger segmentation
SEGMENT_RULES = {
    "frequent_trips_min": 90,  # trips / year for "frequent" travellers
    "peak_share_min": 0.7,     # share of peak-hour trips for "commuter" label
}

# Model settings
MODEL_TEST_SPLIT = _env("UTIQ_MODEL_TEST_SPLIT", 0.25)
MODEL_FORECAST_SEASONAL = _env("UTIQ_FORECAST_SEASONAL", 7)
COMPARE_UNSEEN_CASES = _env("UTIQ_COMPARE_CASES", 100)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

APP_TITLE = "UrbanTransit IQ"
APP_VERSION = _env("UTIQ_APP_VERSION", "1.0.0")
APP_THEME = "default"
REPORT_PAGE_LIMIT = _env("UTIQ_REPORT_PAGE_LIMIT", 50)

# ---------------------------------------------------------------------------
# Authentication (SRS section 44). Disabled by default so a fresh clone runs in
# open mode; enable with ``UTIQ_AUTH_ENABLED=true``. There is deliberately no
# hardcoded default secret (SRS section 45) -- ``UTIQ_AUTH_SECRET_KEY`` must be
# supplied by the environment when authentication is enabled.
# ---------------------------------------------------------------------------

AUTH_ENABLED = _env("UTIQ_AUTH_ENABLED", False)
AUTH_SECRET_KEY = _env("UTIQ_AUTH_SECRET_KEY", "")
AUTH_COOKIE_NAME = _env("UTIQ_AUTH_COOKIE_NAME", "urbantransit_auth")
AUTH_COOKIE_EXPIRY_DAYS = _env("UTIQ_AUTH_COOKIE_EXPIRY_DAYS", 30)
AUTH_MAX_FAILED_ATTEMPTS = _env("UTIQ_AUTH_MAX_ATTEMPTS", 5)
AUTH_LOCKOUT_SECONDS = _env("UTIQ_AUTH_LOCKOUT_SECONDS", 300)