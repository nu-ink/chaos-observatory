from __future__ import annotations

import csv
import json

from ml import ml_sentiment_shift


def test_score_article_detects_urgent_negative_tone() -> None:
    article = ml_sentiment_shift.Article(
        article_id="a1",
        source="ReliefWeb",
        topic="humanitarian",
        title="Emergency warning issued",
        text="Officials reported a critical crisis and urgent evacuation.",
        published_at="2026-05-30T10:00:00Z",
    )

    score = ml_sentiment_shift.score_article(article)

    assert score.tone_label == "urgent"
    assert score.urgent_score > 0
    assert "urgent" in score.matched_terms["urgent"]


def test_sentiment_cli_writes_outputs(tmp_path) -> None:
    input_path = tmp_path / "articles.csv"
    scores_path = tmp_path / "scores.json"
    alerts_path = tmp_path / "alerts.json"
    report_path = tmp_path / "report.md"

    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "article_id",
                "source",
                "topic",
                "title",
                "text",
                "published_at",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "article_id": "a1",
                "source": "BBC",
                "topic": "world",
                "title": "Routine market update",
                "text": "Markets were stable and normal.",
                "published_at": "2026-05-30T08:00:00Z",
            }
        )
        writer.writerow(
            {
                "article_id": "a2",
                "source": "BBC",
                "topic": "world",
                "title": "Emergency alert issued",
                "text": "A critical crisis warning raised urgent concern.",
                "published_at": "2026-05-30T09:00:00Z",
            }
        )
        writer.writerow(
            {
                "article_id": "a3",
                "source": "BBC",
                "topic": "world",
                "title": "Evacuation warning expands",
                "text": "Emergency officials warned of a critical threat.",
                "published_at": "2026-05-30T10:00:00Z",
            }
        )

    ml_sentiment_shift.main(
        [
            "--input",
            str(input_path),
            "--scores-output",
            str(scores_path),
            "--alerts-output",
            str(alerts_path),
            "--report-output",
            str(report_path),
            "--min-articles",
            "3",
            "--alert-threshold",
            "0.1",
        ]
    )

    scores = json.loads(scores_path.read_text(encoding="utf-8"))

    assert len(scores) == 3
    assert alerts_path.exists()
    assert report_path.exists()
