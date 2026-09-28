"""Diagnostic shim.

Spark launches the python worker with PYTHON_WORKER_FACTORY_PORT in its
environment, and that variable is present in the worker process only. So this
module is a no-op for the driver and a stderr tap for workers.

Add the containing directory to PYTHONPATH to activate.
"""
import os
import sys

if os.environ.get("PYTHON_WORKER_FACTORY_PORT") and os.environ.get("UTIQ_WORKER_STDERR_LOG"):
    log_path = os.environ["UTIQ_WORKER_STDERR_LOG"]
    try:
        fh = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
        sys.stderr = fh
        sys.stdout = fh
        import atexit
        import faulthandler

        faulthandler.enable(file=fh)
        atexit.register(fh.close)
    except Exception:  # noqa: BLE001
        pass
