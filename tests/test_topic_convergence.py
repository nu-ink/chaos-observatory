from __future__ import annotations

import sqlite3

import pandas as pd

from ml import ml_topic_convergence


def test_load_records_reads_repo_documents_schema(tmp_path):
    db_path = tmp_path / "chaos.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                normalized_at_utc TEXT NOT NULL,
                ingested_at_utc TEXT,
                published_at_utc TEXT,
                source_id TEXT NOT NULL,
                source_label TEXT,
                region TEXT,
                category TEXT,
                feed_url TEXT,
                url TEXT,
                title TEXT,
                body_text TEXT,
                raw_json TEXT
            )
            """)
        conn.execute(
            """
            INSERT INTO documents (
                doc_id,
                normalized_at_utc,
                published_at_utc,
                source_id,
                source_label,
                url,
                title,
                body_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "doc-1",
                "2026-05-30T12:00:00Z",
                "2026-05-30T10:00:00Z",
                "reliefweb",
                "ReliefWeb",
                "https://example.test/item",
                "Flood response expands",
                "Emergency teams widened the regional response effort.",
            ),
        )

        records = ml_topic_convergence.load_records(conn, "2026-05-30")

    assert len(records) == 1
    assert records.iloc[0]["document_id"] == "doc-1"
    assert records.iloc[0]["source"] == "ReliefWeb"
    assert records.iloc[0]["text"] == (
        "Flood response expands. Emergency teams widened the regional response effort."
    )


def test_load_records_returns_empty_when_no_source_table(tmp_path):
    db_path = tmp_path / "empty.db"
    with sqlite3.connect(db_path) as conn:
        records = ml_topic_convergence.load_records(conn, "2026-05-30")

    assert records.empty


def test_summarize_clusters_orders_by_convergence_score():
    df = pd.DataFrame(
        [
            {
                "cluster_id": 1,
                "text": "ports logistics shipping disruption",
                "source": "A",
                "title": "Port disruption",
            },
            {
                "cluster_id": 1,
                "text": "shipping logistics port delays",
                "source": "B",
                "title": "Shipping delays",
            },
            {
                "cluster_id": 2,
                "text": "central bank policy rate",
                "source": "A",
                "title": "Policy rate",
            },
            {
                "cluster_id": -1,
                "text": "noise item",
                "source": "C",
                "title": "Noise",
            },
        ]
    )

    summaries = ml_topic_convergence.summarize_clusters(df)

    assert [item["cluster_id"] for item in summaries] == [1, 2]
    assert summaries[0]["source_count"] == 2
