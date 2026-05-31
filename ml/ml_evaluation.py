from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LabeledMatch:
    article_id: str
    matched_article_id: str
    predicted: bool
    relevant: bool


def precision_at_k(results: Iterable[bool], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    sliced = list(results)[:k]
    if not sliced:
        return 0.0
    return sum(1 for value in sliced if value) / len(sliced)


def recall_at_k(results: Iterable[bool], total_relevant: int, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    if total_relevant < 0:
        raise ValueError("total_relevant must be >= 0")
    if total_relevant == 0:
        return 0.0
    sliced = list(results)[:k]
    return sum(1 for value in sliced if value) / total_relevant


def duplicate_false_positive_rate(predictions: Iterable[LabeledMatch]) -> float:
    items = list(predictions)
    predicted_positive = [item for item in items if item.predicted]
    if not predicted_positive:
        return 0.0
    false_positives = [item for item in predicted_positive if not item.relevant]
    return len(false_positives) / len(predicted_positive)
