"""Analytics orchestrator.

Runs every analytics module, records an audit entry with the produced frame
counts and writes ``data/analytics/analytics_summary.json`` used by the app.
"""

import json
import time
from datetime import datetime, timezone

import pandas as pd

from src.analytics import crowding, demand, frequency, route_metrics, segments, stop_metrics
from src.audit import record
from src.log import get_logger
from src.paths import DATA_METADATA

logger = get_logger(__name__)


STEPS = {
    "route_metrics": route_metrics.build,
    "stop_metrics": stop_metrics.build,
    "demand": demand.build,
    "crowding": crowding.build,
    "segmentation": segments.build,
    "frequency": frequency.build,
}


def _normalize_counts(counts) -> dict:
    if isinstance(counts, pd.DataFrame):
        return {"rows": len(counts)}
    if isinstance(counts, dict):
        return {k: (len(v) if isinstance(v, pd.DataFrame) else v)
                for k, v in counts.items()}
    return {"rows": counts}


def run_all(skip: tuple[str, ...] = ()) -> dict[str, dict]:
    """Execute every analytics step and report rows produced per module."""
    t0 = time.perf_counter()
    summary: dict[str, dict] = {}
    for name, fn in STEPS.items():
        if name in skip:
            continue
        step_t0 = time.perf_counter()
        try:
            counts = fn()
            norm = _normalize_counts(counts)
            summary[name] = norm
            record(
                action=f"analytics:{name}",
                component="analytics",
                status="ok",
                details=json.dumps(norm, default=str),
                duration_ms=(time.perf_counter() - step_t0) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("analytics step %s failed: %s", name, exc)
            summary[name] = {"error": str(exc)[:300]}
            record(action=f"analytics:{name}", component="analytics",
                   status="failed", details=str(exc)[:300])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        "steps": summary,
    }
    DATA_METADATA.mkdir(parents=True, exist_ok=True)
    DATA_METADATA.joinpath("analytics_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("analytics run finished in %.1fs", payload["duration_ms"])
    record(action="analytics", component="analytics", status="ok",
           details=json.dumps({k: v for k, v in summary.items() if "error" not in v}),
           duration_ms=payload["duration_ms"])
    return summary


if __name__ == "__main__":
    out = run_all()
    for name, counts in out.items():
        print(f"{name:18s} {counts}")