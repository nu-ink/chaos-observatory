from ml.ml_evaluation import (
    LabeledMatch,
    duplicate_false_positive_rate,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k() -> None:
    assert precision_at_k([True, False, True, True], 3) == 2 / 3


def test_recall_at_k() -> None:
    assert recall_at_k([True, False, True, True], total_relevant=4, k=3) == 2 / 4


def test_duplicate_false_positive_rate() -> None:
    predictions = [
        LabeledMatch("a1", "a2", predicted=True, relevant=True),
        LabeledMatch("a1", "a3", predicted=True, relevant=False),
        LabeledMatch("a1", "a4", predicted=False, relevant=False),
    ]
    assert duplicate_false_positive_rate(predictions) == 0.5
