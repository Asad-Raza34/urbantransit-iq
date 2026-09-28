"""UrbanTransit IQ - end-to-end pipeline orchestrator.

Runs the whole data workflow in dependency order and prints, for every stage,
the wall-clock time and the number of rows produced. Previously the README asked
the user to run eleven separate ``python -m ...`` commands by hand; a single entry
point makes the project reproducible and demo-safe.

Usage::

    python scripts/run_pipeline.py                 # everything except Spark
    python scripts/run_pipeline.py --with-spark    # also run the Spark SQL jobs
    python scripts/run_pipeline.py --only analytics recommend
    python scripts/run_pipeline.py --skip cleaner  # skip by stage name
    python scripts/run_pipeline.py --only generate --months 2   # small smoke run

Stage names: generate, clean, validate, integrate, features, analytics, ml,
delay, clustering, forecasting, recommend, alerts, spark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

STAGE_ORDER = [
    "generate", "clean", "validate", "integrate", "features",
    "analytics", "ml", "delay", "clustering", "forecasting", "recommend",
    "alerts", "spark",
]
SPARK_STAGES = {"spark"}


def _errors_in(obj, prefix: str = "") -> list[str]:
    """Collect ``"error"`` entries from a stage result.

    Several stages catch per-step exceptions and return ``{"error": ...}``
    instead of raising. Without this check a stage that silently failed eight of
    its nine steps would still print ``[ OK ]``.
    """
    found: list[str] = []
    if isinstance(obj, dict):
        if obj.get("error"):
            found.append(f"{prefix or 'stage'}: {obj['error']}")
        for key, value in obj.items():
            if key == "error":
                continue
            found.extend(_errors_in(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found.extend(_errors_in(value, f"{prefix}[{i}]"))
    return found


def _rows(obj) -> str:
    """Human-readable row counts from whatever a stage returns."""
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return f"{len(obj):,} rows"
    if isinstance(obj, dict):
        parts = []
        for key, value in list(obj.items())[:12]:
            if isinstance(value, pd.DataFrame):
                parts.append(f"{key}={len(value):,}")
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                parts.append(f"{key}={value:,}" if isinstance(value, int) else f"{key}={value}")
            elif isinstance(value, (list, tuple)):
                parts.append(f"{key}={len(value):,}")
        return " ".join(parts) if parts else json.dumps(obj, default=str)[:180]
    return str(obj)[:180]


def run_generate(months=None, scale=None, seed=None):
    from config import settings
    from src.seed.generator import generate

    frames = generate(seed=seed or settings.GEN_SEED,
                      scale=scale or dict(settings.GEN_SCALE),
                      months=months or settings.GEN_MONTHS)
    return {k: v for k, v in frames.items() if hasattr(v, "__len__")}


def run_clean():
    from src.clean.cleaner import clean_all

    cleaned, report = clean_all()
    return {"cleaned": cleaned, "repair_operations": len(report)}


def run_validate():
    from src.validate.validator import raw_layer, generated_layer, cleaned_layer, validate_layer

    raw_report = validate_layer(raw_layer(), reference=generated_layer())
    cleaned_report = validate_layer(cleaned_layer(), reference=generated_layer())
    return {
        "raw_checks": len(raw_report),
        "raw_failures": int((raw_report["status"] != "pass").sum()) if "status" in raw_report else 0,
        "cleaned_checks": len(cleaned_report),
        "cleaned_failures": int((cleaned_report["status"] != "pass").sum()) if "status" in cleaned_report else 0,
    }


def run_integrate():
    from src.integrate.integrator import integrate

    return integrate()


def run_features():
    from src.features.engineer import engineer, persist

    features = engineer(full=True)
    persist(features, full=True)
    return features


def run_analytics():
    from src.analytics.run import run_all

    return run_all()


def run_ml():
    from src.ml.run import run_all

    return run_all()


def run_delay():
    """Train the delay-prediction models (SRS 18).

    Without this stage ``models/python/delay_prediction_*.joblib`` is never
    written, so the Delay & Reliability page's "Generate Prediction" button
    fails with "Model not found. Train first." on a fresh clone.
    """
    from src.ml.delay_prediction import build_delay_models

    return build_delay_models()


def run_clustering():
    from src.clustering.route_clustering import build_clustering

    return build_clustering()


def run_forecasting():
    from src.forecasting.occupancy_forecast import build_occupancy_models

    return build_occupancy_models()


def run_recommend():
    from src.recommend.engine import build_recommendations

    return build_recommendations()


def run_alerts():
    from src.alerts.center import build_alerts

    return build_alerts()


def run_spark():
    from src.analytics.spark_job import build

    return build()


STAGES = {
    "generate": run_generate,
    "clean": run_clean,
    "validate": run_validate,
    "integrate": run_integrate,
    "features": run_features,
    "analytics": run_analytics,
    "ml": run_ml,
    "delay": run_delay,
    "clustering": run_clustering,
    "forecasting": run_forecasting,
    "recommend": run_recommend,
    "alerts": run_alerts,
    "spark": run_spark,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the UrbanTransit IQ pipeline.")
    parser.add_argument("--only", nargs="*", choices=STAGE_ORDER, default=None,
                        help="run only these stages (still in dependency order)")
    parser.add_argument("--skip", nargs="*", choices=STAGE_ORDER, default=[],
                        help="skip these stages")
    parser.add_argument("--with-spark", action="store_true",
                        help="include the Spark SQL stage (excluded by default)")
    parser.add_argument("--months", type=int, default=None,
                        help="override months of synthetic data (e.g. 2 for a quick run)")
    parser.add_argument("--seed", type=int, default=None, help="override the generation seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the stage plan in execution order without running it")
    args = parser.parse_args()

    stages = list(args.only) if args.only else list(STAGE_ORDER)
    if not args.only and not args.with_spark:
        stages = [s for s in stages if s not in SPARK_STAGES]
    stages = [s for s in stages if s not in args.skip]
    stages.sort(key=STAGE_ORDER.index)

    print("UrbanTransit IQ pipeline")
    print(f"  stages : {', '.join(stages)}")
    if args.months:
        print(f"  months : {args.months} (override)")
    if args.dry_run:
        for i, stage in enumerate(stages, 1):
            print(f"  {i:2d}. {stage}")
        print("=" * 74)
        print("dry run: nothing executed")
        return 0
    print("=" * 74)

    failures: list[tuple[str, str]] = []
    total_start = time.perf_counter()
    for stage in stages:
        fn = STAGES[stage]
        start = time.perf_counter()
        try:
            if stage == "generate" and (args.months or args.seed):
                result = fn(months=args.months, seed=args.seed)
            else:
                result = fn()
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            failures.append((stage, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {stage:12s} {elapsed:7.1f}s  {type(exc).__name__}: {exc}")
            print(traceback.format_exc()[-1200:])
            continue
        elapsed = time.perf_counter() - start
        errors = _errors_in(result)
        if errors:
            failures.append((stage, "; ".join(errors)))
            print(f"[FAIL] {stage:12s} {elapsed:7.1f}s  {_rows(result)}")
            for message in errors:
                print(f"        inner error -> {message}")
            continue
        print(f"[ OK ] {stage:12s} {elapsed:7.1f}s  {_rows(result)}")

    total = time.perf_counter() - total_start
    print("=" * 74)
    print(f"TOTAL {total:.1f}s | {len(stages) - len(failures)}/{len(stages)} stages succeeded")
    for stage, message in failures:
        print(f"  FAILED {stage}: {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
