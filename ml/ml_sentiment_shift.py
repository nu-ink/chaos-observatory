#!/usr/bin/env python3
"""
sentiment_shift_ml.py

Purpose:
    MVP sentiment / tone-shift detector for chaos-observatory.

What it does now:
    - Loads article/event text from CSV or JSON
    - Scores tone using domain-specific lexicons
    - Produces sentiment/tone scores
    - Aggregates scores by source/topic
    - Detects sudden tone shifts using rolling baseline logic
    - Writes alerts to a JSON file

Future ML path:
    - Replace or enhance rule scoring with TF-IDF + Logistic Regression
    - Later upgrade to DistilBERT or another transformer model
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Domain tone lexicons
# ---------------------------------------------------------------------

LEXICONS: dict[str, list[str]] = {
    "urgent": [
        "breaking",
        "urgent",
        "emergency",
        "crisis",
        "immediate",
        "rapidly",
        "evacuate",
        "warning",
        "alert",
        "threat",
        "critical",
        "surge",
        "escalation",
        "collapse",
    ],
    "uncertain": [
        "may",
        "might",
        "could",
        "possibly",
        "unclear",
        "unknown",
        "uncertain",
        "unconfirmed",
        "reportedly",
        "allegedly",
        "appears",
        "suggests",
        "risk",
    ],
    "hostile": [
        "attack",
        "war",
        "conflict",
        "strike",
        "retaliation",
        "invasion",
        "militant",
        "hostile",
        "violence",
        "clash",
        "threatened",
    ],
    "negative": [
        "decline",
        "loss",
        "failure",
        "failed",
        "falling",
        "worsen",
        "worsening",
        "damage",
        "shortage",
        "inflation",
        "recession",
        "fear",
        "concern",
        "concerns",
    ],
    "positive": [
        "growth",
        "improved",
        "improvement",
        "stable",
        "stabilized",
        "agreement",
        "recovery",
        "gain",
        "peace",
        "resolved",
        "progress",
    ],
    "calm": [
        "steady",
        "unchanged",
        "routine",
        "normal",
        "stable",
        "measured",
        "expected",
        "planned",
    ],
}


# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------


@dataclass
class Article:
    article_id: str
    source: str
    topic: str
    title: str
    text: str
    published_at: str


@dataclass
class ToneScore:
    article_id: str
    source: str
    topic: str
    published_at: str
    sentiment_score: float
    urgent_score: float
    uncertain_score: float
    hostile_score: float
    negative_score: float
    positive_score: float
    calm_score: float
    tone_label: str
    matched_terms: dict[str, list[str]]


@dataclass
class ToneShiftAlert:
    source: str
    topic: str
    current_score: float
    baseline_score: float
    shift_score: float
    alert_level: str
    explanation: str


# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------


def load_articles(input_path: Path) -> list[Article]:
    """
    Load articles from CSV or JSON.

    Expected fields:
        article_id, source, topic, title, text, published_at

    The script is forgiving:
        - If article_id is missing, it creates one.
        - If topic is missing, it uses "unknown".
        - If published_at is missing, it uses current timestamp.
    """

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".csv":
        return load_articles_from_csv(input_path)

    if input_path.suffix.lower() == ".json":
        return load_articles_from_json(input_path)

    raise ValueError("Input must be a .csv or .json file")


def load_articles_from_csv(input_path: Path) -> list[Article]:
    articles: list[Article] = []

    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for idx, row in enumerate(reader, start=1):
            articles.append(
                Article(
                    article_id=str(row.get("article_id") or idx),
                    source=str(row.get("source") or "unknown"),
                    topic=str(row.get("topic") or "unknown"),
                    title=str(row.get("title") or ""),
                    text=str(row.get("text") or row.get("summary") or ""),
                    published_at=str(
                        row.get("published_at") or datetime.utcnow().isoformat()
                    ),
                )
            )

    return articles


def load_articles_from_json(input_path: Path) -> list[Article]:
    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        records = payload.get("articles") or payload.get("items") or []
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("JSON input must be a list or an object with articles/items")

    articles: list[Article] = []

    for idx, row in enumerate(records, start=1):
        articles.append(
            Article(
                article_id=str(row.get("article_id") or row.get("id") or idx),
                source=str(row.get("source") or "unknown"),
                topic=str(row.get("topic") or "unknown"),
                title=str(row.get("title") or ""),
                text=str(
                    row.get("text")
                    or row.get("summary")
                    or row.get("description")
                    or ""
                ),
                published_at=str(
                    row.get("published_at")
                    or row.get("published")
                    or datetime.utcnow().isoformat()
                ),
            )
        )

    return articles


# ---------------------------------------------------------------------
# Rule-based scoring
# ---------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """
    Normalize text for simple rule matching.
    """

    return text.lower().replace("\n", " ").strip()


def find_matched_terms(text: str) -> dict[str, list[str]]:
    """
    Find lexicon terms that appear in the article text.
    """

    normalized = normalize_text(text)
    matches: dict[str, list[str]] = {}

    for label, terms in LEXICONS.items():
        found = [term for term in terms if term in normalized]
        matches[label] = found

    return matches


def score_category(matches: list[str], total_words: int) -> float:
    """
    Score one tone category.

    This is intentionally simple for the MVP:
        score = matched_terms / sqrt(total_words)

    This avoids giving very long articles unfairly high scores.
    """

    if total_words <= 0:
        return 0.0

    raw_score = len(matches) / math.sqrt(total_words)

    return round(min(raw_score, 1.0), 4)


def score_article(article: Article) -> ToneScore:
    """
    Score a single article using rule-based domain lexicons.
    """

    combined_text = f"{article.title} {article.text}"
    normalized = normalize_text(combined_text)
    total_words = max(len(normalized.split()), 1)

    matched_terms = find_matched_terms(combined_text)

    urgent_score = score_category(matched_terms["urgent"], total_words)
    uncertain_score = score_category(matched_terms["uncertain"], total_words)
    hostile_score = score_category(matched_terms["hostile"], total_words)
    negative_score = score_category(matched_terms["negative"], total_words)
    positive_score = score_category(matched_terms["positive"], total_words)
    calm_score = score_category(matched_terms["calm"], total_words)

    sentiment_score = round(
        positive_score + calm_score - negative_score - hostile_score - urgent_score,
        4,
    )

    tone_label = choose_tone_label(
        {
            "urgent": urgent_score,
            "uncertain": uncertain_score,
            "hostile": hostile_score,
            "negative": negative_score,
            "positive": positive_score,
            "calm": calm_score,
        }
    )

    return ToneScore(
        article_id=article.article_id,
        source=article.source,
        topic=article.topic,
        published_at=article.published_at,
        sentiment_score=sentiment_score,
        urgent_score=urgent_score,
        uncertain_score=uncertain_score,
        hostile_score=hostile_score,
        negative_score=negative_score,
        positive_score=positive_score,
        calm_score=calm_score,
        tone_label=tone_label,
        matched_terms=matched_terms,
    )


def choose_tone_label(scores: dict[str, float]) -> str:
    """
    Choose the strongest tone label.

    If all scores are zero, return neutral.
    """

    strongest_label = max(scores, key=scores.get)
    strongest_score = scores[strongest_label]

    if strongest_score == 0:
        return "neutral"

    return strongest_label


# ---------------------------------------------------------------------
# Aggregation and shift detection
# ---------------------------------------------------------------------


def group_scores_by_source_topic(
    scores: list[ToneScore],
) -> dict[tuple[str, str], list[ToneScore]]:
    """
    Group scored articles by source and topic.
    """

    grouped: dict[tuple[str, str], list[ToneScore]] = {}

    for score in scores:
        key = (score.source, score.topic)
        grouped.setdefault(key, []).append(score)

    return grouped


def average_risk_score(scores: list[ToneScore]) -> float:
    """
    Risk-style tone score.

    Higher means more urgent/uncertain/hostile/negative.
    """

    if not scores:
        return 0.0

    values = [
        item.urgent_score
        + item.uncertain_score
        + item.hostile_score
        + item.negative_score
        - item.calm_score
        - item.positive_score
        for item in scores
    ]

    return round(statistics.mean(values), 4)


def detect_tone_shifts(
    scores: list[ToneScore],
    min_articles: int = 3,
    alert_threshold: float = 0.25,
) -> list[ToneShiftAlert]:
    """
    Detect tone shifts by comparing the newest article score against
    the group's baseline average.

    MVP logic:
        - group by source/topic
        - use all but the newest item as baseline
        - compare newest score to baseline
        - alert if difference exceeds threshold

    Later upgrade:
        - daily aggregation
        - 7-day rolling average
        - z-score
        - change-point detection with ruptures
    """

    alerts: list[ToneShiftAlert] = []
    grouped = group_scores_by_source_topic(scores)

    for (source, topic), group in grouped.items():
        if len(group) < min_articles:
            continue

        sorted_group = sorted(group, key=lambda item: item.published_at)

        baseline_group = sorted_group[:-1]
        current_group = [sorted_group[-1]]

        baseline_score = average_risk_score(baseline_group)
        current_score = average_risk_score(current_group)

        shift_score = round(current_score - baseline_score, 4)

        if shift_score >= alert_threshold:
            alerts.append(
                ToneShiftAlert(
                    source=source,
                    topic=topic,
                    current_score=current_score,
                    baseline_score=baseline_score,
                    shift_score=shift_score,
                    alert_level=classify_alert_level(shift_score),
                    explanation=(
                        f"Tone risk increased for source='{source}', topic='{topic}'. "
                        f"Current score={current_score}, baseline={baseline_score}, "
                        f"shift={shift_score}."
                    ),
                )
            )

    return alerts


def classify_alert_level(shift_score: float) -> str:
    """
    Convert shift score into human-readable alert level.
    """

    if shift_score >= 0.75:
        return "high"

    if shift_score >= 0.5:
        return "medium"

    return "low"


# ---------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------


def write_json(path: Path, data: Any) -> None:
    """
    Write JSON output.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_markdown_report(
    path: Path, scores: list[ToneScore], alerts: list[ToneShiftAlert]
) -> None:
    """
    Write a simple markdown report.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Sentiment / Tone Shift Report")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC")
    lines.append("")
    lines.append(f"Articles scored: {len(scores)}")
    lines.append(f"Alerts generated: {len(alerts)}")
    lines.append("")

    if alerts:
        lines.append("## Alerts")
        lines.append("")

        for alert in alerts:
            lines.append(
                f"### {alert.alert_level.upper()} — {alert.source} / {alert.topic}"
            )
            lines.append("")
            lines.append(f"- Current score: `{alert.current_score}`")
            lines.append(f"- Baseline score: `{alert.baseline_score}`")
            lines.append(f"- Shift score: `{alert.shift_score}`")
            lines.append(f"- Explanation: {alert.explanation}")
            lines.append("")
    else:
        lines.append("## Alerts")
        lines.append("")
        lines.append("No tone-shift alerts detected.")
        lines.append("")

    lines.append("## Recent Article Scores")
    lines.append("")

    for score in scores[-20:]:
        lines.append(
            f"- `{score.source}` / `{score.topic}` / `{score.tone_label}` "
            f"/ sentiment `{score.sentiment_score}` / article `{score.article_id}`"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Future ML placeholder
# ---------------------------------------------------------------------


def future_ml_classifier_placeholder() -> None:
    """
    Future implementation idea.

    Phase 2:
        - Train TF-IDF + Logistic Regression using labeled CSV.
        - Labels: calm, neutral, concerned, urgent, hostile, uncertain.

    Phase 3:
        - Fine-tune DistilBERT using Hugging Face transformers.
        - Compare against baseline using F1 score.

    This function intentionally does nothing right now.
    """

    return None


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sentiment / tone-shift ML-ready script for chaos-observatory"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a built-in non-destructive smoke test without writing files",
    )

    parser.add_argument(
        "--input",
        required=False,
        help="Path to input CSV or JSON file",
    )

    parser.add_argument(
        "--scores-output",
        default="data/sentiment/sentiment_scores.json",
        help="Path to write article-level tone scores",
    )

    parser.add_argument(
        "--alerts-output",
        default="data/sentiment/tone_shift_alerts.json",
        help="Path to write tone-shift alerts",
    )

    parser.add_argument(
        "--report-output",
        default="reports/sentiment_shift_report.md",
        help="Path to write markdown report",
    )

    parser.add_argument(
        "--min-articles",
        type=int,
        default=3,
        help="Minimum articles required per source/topic before shift detection",
    )

    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.25,
        help="Minimum shift score required to generate alert",
    )

    return parser.parse_args(argv)


def run_dry_run() -> dict[str, Any]:
    articles = [
        Article(
            article_id="health-1",
            source="health",
            topic="smoke",
            title="Routine status update",
            text="Operations were stable and normal.",
            published_at="2026-06-01T00:00:00Z",
        ),
        Article(
            article_id="health-2",
            source="health",
            topic="smoke",
            title="Emergency warning issued",
            text="Officials reported a critical urgent crisis warning.",
            published_at="2026-06-01T01:00:00Z",
        ),
        Article(
            article_id="health-3",
            source="health",
            topic="smoke",
            title="Evacuation alert expands",
            text="Emergency teams warned of an urgent threat.",
            published_at="2026-06-01T02:00:00Z",
        ),
    ]
    scores = [score_article(article) for article in articles]
    alerts = detect_tone_shifts(scores, min_articles=3, alert_threshold=0.1)
    return {
        "status": "ok",
        "event": "ml_sentiment_shift_dry_run",
        "articles": len(articles),
        "scores": len(scores),
        "alerts": len(alerts),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.dry_run:
        print(json.dumps(run_dry_run(), indent=2, sort_keys=True))
        return

    if not args.input:
        raise SystemExit("ERROR: --input is required unless --dry-run is set")

    input_path = Path(args.input)
    scores_output = Path(args.scores_output)
    alerts_output = Path(args.alerts_output)
    report_output = Path(args.report_output)

    articles = load_articles(input_path)

    scores = [score_article(article) for article in articles]

    alerts = detect_tone_shifts(
        scores=scores,
        min_articles=args.min_articles,
        alert_threshold=args.alert_threshold,
    )

    write_json(scores_output, [asdict(score) for score in scores])
    write_json(alerts_output, [asdict(alert) for alert in alerts])
    write_markdown_report(report_output, scores, alerts)

    print(f"Articles loaded: {len(articles)}")
    print(f"Scores written: {scores_output}")
    print(f"Alerts written: {alerts_output}")
    print(f"Report written: {report_output}")
    print(f"Alerts generated: {len(alerts)}")


if __name__ == "__main__":
    main()
