"""Machine-learning and monitoring helpers for Chaos Observatory."""

from __future__ import annotations

from importlib import import_module

_MODULE_ALIASES = {
    "change_detection": "ml.ml_change_detection",
    "topic_convergence": "ml.ml_topic_convergence",
    "ml_change_detection": "ml.ml_change_detection",
    "ml_topic_convergence": "ml.ml_topic_convergence",
}

__all__ = sorted(_MODULE_ALIASES)


def __getattr__(name: str):
    if name in _MODULE_ALIASES:
        return import_module(_MODULE_ALIASES[name])
    raise AttributeError(f"module 'ml' has no attribute {name!r}")
