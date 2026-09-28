"""Model training orchestrator.

Runs the Spark MLlib and scikit-learn demand models plus the 100-case parity
comparison, and records the outcome in the audit store.
"""

import json

from src.audit import record
from src.log import get_logger

logger = get_logger(__name__)


def run_all() -> dict:
    from src.ml import mllib_model, parity, sklearn_model

    out: dict = {}
    for name in ("mllib", "sklearn", "parity"):
        try:
            step_t0 = __import__("time").perf_counter()
            result = {"mllib": mllib_model.build,
                      "sklearn": sklearn_model.build,
                      "parity": parity.build}[name]()
            out[name] = result
            record(
                action=f"ml:{name}", component="ml", status="ok",
                details=json.dumps(result, default=str),
                duration_ms=round((__import__("time").perf_counter() - step_t0) * 1000, 1),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ml step %s failed: %s", name, exc)
            out[name] = {"error": str(exc)[:300]}
            record(action=f"ml:{name}", component="ml", status="failed",
                   details=str(exc)[:300])
    return out


def main() -> None:
    print(json.dumps(run_all(), indent=2, default=str))


if __name__ == "__main__":
    main()