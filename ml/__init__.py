"""Machine-learning and monitoring helpers for Chaos Observatory."""

from __future__ import annotations

from importlib import import_module

_MODULE_ALIASES = {
    "change_detection": "ml.ml_change_detection",
    "embeddings": "ml.ml_embeddings",
    "evaluation": "ml.ml_evaluation",
    "ml_change_detection": "ml.ml_change_detection",
    "ml_embeddings": "ml.ml_embeddings",
    "ml_evaluation": "ml.ml_evaluation",
    "ml_semantic_linker": "ml.ml_semantic_linker",
    "ml_sentiment_shift": "ml.ml_sentiment_shift",
    "ml_similarity_thresholds": "ml.ml_similarity_thresholds",
    "ml_topic_convergence": "ml.ml_topic_convergence",
    "semantic_linker": "ml.ml_semantic_linker",
    "sentiment_shift": "ml.ml_sentiment_shift",
    "similarity_thresholds": "ml.ml_similarity_thresholds",
    "topic_convergence": "ml.ml_topic_convergence",
    "vector_store": "ml.vector_store",
}

__all__ = sorted(_MODULE_ALIASES)


def __getattr__(name: str):
    if name in _MODULE_ALIASES:
        return import_module(_MODULE_ALIASES[name])
    raise AttributeError(f"module 'ml' has no attribute {name!r}")
