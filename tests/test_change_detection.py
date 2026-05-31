from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml import ml_change_detection


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_events_from_normalized_uses_repo_fields(tmp_path):
    write_jsonl(
        tmp_path / "2026-05-29" / "items.jsonl",
        [
            {
                "published_at_utc": "2026-05-29T12:00:00Z",
                "source_label": "ReliefWeb",
                "category": "humanitarian",
                "title": "Flood response expands",
            }
        ],
    )

    events = ml_change_detection.load_events_from_normalized(
        normalized_dir=tmp_path,
        start_day=ml_change_detection.parse_ymd("2026-05-29"),
        end_day=ml_change_detection.parse_ymd("2026-05-29"),
    )

    assert list(events.columns) == ["timestamp", "source", "topic"]
    assert events.iloc[0]["source"] == "ReliefWeb"
    assert events.iloc[0]["topic"] == "humanitarian"


def test_aggregate_counts_fills_expected_group_counts():
    events = pd.DataFrame(
        [
            {"timestamp": "2026-05-29T01:00:00Z", "source": "BBC", "topic": "world"},
            {"timestamp": "2026-05-29T10:00:00Z", "source": "BBC", "topic": "world"},
            {"timestamp": "2026-05-30T01:00:00Z", "source": "BBC", "topic": "world"},
        ]
    )
    events = ml_change_detection.normalize_events_frame(events)

    counts = ml_change_detection.aggregate_counts(events, bucket_size="1D")

    assert counts["count"].tolist() == [2, 1]


def test_time_to_detection_keeps_first_non_negative_detection():
    alerts = pd.DataFrame(
        [
            {"source": "BBC", "topic": "world", "breakpoint_time": "2026-05-29T23:00:00Z"},
            {"source": "BBC", "topic": "world", "breakpoint_time": "2026-05-30T01:30:00Z"},
            {"source": "BBC", "topic": "world", "breakpoint_time": "2026-05-30T03:00:00Z"},
        ]
    )
    incidents = pd.DataFrame(
        [{"source": "BBC", "topic": "world", "incident_start": "2026-05-30T01:00:00Z"}]
    )

    result = ml_change_detection.time_to_detection(alerts, incidents)

    assert len(result) == 1
    assert result.iloc[0]["time_to_detection_minutes"] == 30.0
