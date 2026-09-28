"""Facade over config.settings + a small validation helper.

The raw values live in ``config/settings.py``; this module is the only place
that imports that file so the rest of the code base uses :func:`cfg`.
"""

from config import settings


def cfg(name: str):
    """Read a setting by dotted attribute name, e.g. ``cfg("GEN_SCALE.routes")``."""
    value = settings
    for part in name.split("."):
        value = getattr(value, part)
    return value


def validate_settings():
    """Sanity-check the most important settings at startup."""
    weights = settings.ROUTE_SCORE_WEIGHTS
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ROUTE_SCORE_WEIGHTS must sum to 1.0, got {total}")


def ensure_environment():
    """Validate that required paths exist and create the layout if needed."""
    from src.paths import ensure_layout

    validate_settings()
    ensure_layout()
    return True


__all__ = ["cfg", "validate_settings", "ensure_environment", "settings"]