#!/usr/bin/env python3
"""Import normalized JSONL partitions into storage/chaos.db for ML workflows.

Reads all `*.jsonl` files under a date-partitioned `data/normalized` directory
and writes them into a `documents` table usable by ML scripts.

Usage:
  python scripts/import_normalized_to_db.py --in data/normalized --db storage/chaos.db

"""
from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_id_from_row(row: dict[str, Any]) -> str:
    src = row.get("id") or row.get("doc_id") or ""
    if src:
        return str(src)
    raw = f"{row.get('title','')}|{row.get('url','')}|{row.get('published_at_utc','')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            normalized_at_utc TEXT NOT NULL,
            ingested_at_utc TEXT,
            published_at_utc TEXT,
            source_id TEXT,
            source_label TEXT,
            region TEXT,
            category TEXT,
            feed_url TEXT,
            url TEXT,
            title TEXT,
            body_text TEXT,
            raw_json TEXT
        );
        """
    )


def insert_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    doc_id = stable_id_from_row(row)
    normalized_at = row.get("normalized_at_utc") or utc_now()
    ingested = row.get("ingested_at_utc")
    published = row.get("published_at_utc") or row.get("published") or row.get("published_at")
    source_id = row.get("source_id") or row.get("source")
    source_label = row.get("source_label") or row.get("source") or source_id
    category = row.get("category")
    feed_url = row.get("feed_url") or row.get("feed")
    url = row.get("url") or row.get("link")
    title = row.get("title")
    body = row.get("body_text") or row.get("body") or row.get("summary") or row.get("content")
    raw = json.dumps(row, ensure_ascii=False)

    conn.execute(
        """
        INSERT OR REPLACE INTO documents (
            doc_id, normalized_at_utc, ingested_at_utc, published_at_utc,
            source_id, source_label, category, feed_url, url, title, body_text, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            normalized_at,
            ingested,
            published,
            source_id,
            source_label,
            category,
            feed_url,
            url,
            title,
            body,
            raw,
        ),
    )


def import_normalized(base_dir: Path, db_path: Path) -> int:
    files = list(base_dir.rglob("*.jsonl"))
    if not files:
        print(json.dumps({"event": "no_files_found", "base_dir": str(base_dir)}))
        return 0

    conn = sqlite3.connect(str(db_path))
    ensure_table(conn)

    total = 0
    for p in sorted(files):
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                insert_row(conn, row)
                total += 1

    conn.commit()
    conn.close()
    print(json.dumps({"event": "import_done", "files": len(files), "rows": total, "db": str(db_path)}))
    return total


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="base_dir", default="data/normalized", help="Base normalized dir")
    ap.add_argument("--db", dest="db_path", default="storage/chaos.db", help="SQLite DB path")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    base = Path(args.base_dir)
    db = Path(args.db_path)

    if not base.exists():
        print(json.dumps({"event": "base_missing", "base": str(base)}))
        return 1

    rows = import_normalized(base, db)
    return 0 if rows >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
