from ml.ml_similarity_thresholds import (SemanticSimilarityConfig,
                                         classify_score)


def test_classify_score_duplicate() -> None:
    config = SemanticSimilarityConfig()
    assert classify_score(0.91, config) == "duplicate"


def test_classify_score_near_duplicate() -> None:
    config = SemanticSimilarityConfig()
    assert classify_score(0.85, config) == "near_duplicate"


def test_classify_score_related_context() -> None:
    config = SemanticSimilarityConfig()
    assert classify_score(0.70, config) == "related_context"


def test_classify_score_unrelated() -> None:
    config = SemanticSimilarityConfig()
    assert classify_score(0.40, config) == "unrelated"
