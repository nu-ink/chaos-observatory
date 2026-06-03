import os
import tempfile
import json
from pathlib import Path

from storage.db import DB, DBConfig, open_db


def _create_minimal_documents_table(conn):
    conn.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            published_at_utc TEXT,
            source_id TEXT,
            source_label TEXT,
            region TEXT,
            category TEXT,
            url TEXT,
            title TEXT,
            body_text TEXT
        )
        """)


def test_query_documents_window_limits_and_returns_rows():
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.close()
    try:
        db = DB(DBConfig(path=tf.name))
        conn = db.connect()
        _create_minimal_documents_table(conn)

        # insert two rows with different published timestamps
        conn.execute(
            "INSERT INTO documents(doc_id, published_at_utc, source_id, title) VALUES (?, ?, ?, ?)",
            ("d1", "2026-01-01T00:00:00Z", "s1", "t1"),
        )
        conn.execute(
            "INSERT INTO documents(doc_id, published_at_utc, source_id, title) VALUES (?, ?, ?, ?)",
            ("d2", "2026-01-02T00:00:00Z", "s1", "t2"),
        )

        res = db.query_documents_window(
            published_start_utc="2026-01-01T00:00:00Z",
            published_end_utc="2026-12-31T00:00:00Z",
            limit=10,
        )
        assert isinstance(res, list)
        assert len(res) == 2

        # limit enforced
        res2 = db.query_documents_window(
            published_start_utc="2026-01-01T00:00:00Z",
            published_end_utc="2026-12-31T00:00:00Z",
            limit=1,
        )
        assert len(res2) == 1

    finally:
        db.close()
        os.unlink(tf.name)
