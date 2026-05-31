from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SemanticSimilarityConfig:
    enabled: bool = True
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    duplicate_threshold: float = 0.90
    near_duplicate_threshold: float = 0.80
    related_context_threshold: float = 0.65
    index_path: str = "data/vector_index/semantic_articles.faiss"
    database_path: str = "data/chaos_observatory.db"
    review_output_path: str = "reports/semantic_review.csv"

    def validate(self) -> None:
        thresholds = (
            self.related_context_threshold,
            self.near_duplicate_threshold,
            self.duplicate_threshold,
        )
        if not (0.0 <= thresholds[0] <= thresholds[1] <= thresholds[2] <= 1.0):
            raise ValueError("Thresholds must satisfy 0 <= related <= near <= duplicate <= 1.")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0.")


def classify_score(score: float, config: SemanticSimilarityConfig) -> str:
    if score >= config.duplicate_threshold:
        return "duplicate"
    if score >= config.near_duplicate_threshold:
        return "near_duplicate"
    if score >= config.related_context_threshold:
        return "related_context"
    return "unrelated"
