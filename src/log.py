"""Application wide logging helper.

Writes to both stderr (so the Streamlit UI can surface messages) and a
rotating file under ``logs/``. A handful of named loggers is enough - every
component reuses ``root_logger``.
"""

import logging
from logging.handlers import RotatingFileHandler

from src.paths import LOGS_ROOT

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "utiq"):
    """Return (creating once) a logger with console + file handlers."""
    logger = logging.getLogger(name)
    if getattr(logger, "_utiq_initialized", False):
        return logger
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOGS_ROOT / "pipeline.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_FMT))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger._utiq_initialized = True
    return logger