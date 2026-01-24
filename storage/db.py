#!/usr/bin/env python3
"""
Chaos-Observatory DB wrapper (v1)

SQLite-first (built-in), designed for:
- init_db() using storage/schema.sql
- insert/upsert raw_items and documents
- store analysis results (JSON)
- query docs by date window for analyzers/reports

No heavy ORM. Parameterized queries. Safe-by-default.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

DEFAULT_DB_PATH = "data/chaos_observatory.sqlite3"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_hex(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


@dataclass
class DBConfig:
    path: str = DEFAULT_DB_PATH
    timeout_sec: float = 30.0
    busy_retry_sec: float = 0.2
    busy_retries: int = 25


class DB:
    """
    Minimal SQLite DB helper with:
      - connection lifecycle
      - init schema
      - upserts
      - common queries
    """

    def __init__(self, cfg: DBConfig):
        self.cfg = cfg
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.cfg.path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                self.cfg.path,
                timeout=self.cfg.timeout_sec,
                isolation_level=None,  # autocommit; we'll manage transactions explicitly
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        self._exec_with_retry(conn, "BEGIN;")
        try:
            yield conn
            self._exec_with_retry(conn, "COMMIT;")
        except Exception:
            self._exec_with_retry(conn, "ROLLBACK;")
            raise

    def init_db(self, schema_path: str = "storage/schema.sql") -> None:
        conn = self.connect()
        sql = Path(schema_path).read_text(encoding="utf-8")
        # Executescript may fail on FTS5 on some builds; we handle that gracefully.
        try:
            conn.executescript(sql)
        except sqlite3.OperationalError as e:
            # Try again without the FTS table if that's the failure.
            # We keep this conservative: if it fails, we remove the FTS block.
            if "fts5" in str(e).lower():
                cleaned = self._strip_fts_block(sql)
                conn.executescript(cleaned)
            else:
                raise

    @staticmethod
    def _strip_fts_block(sql: str) -> str:
        lines = sql.splitlines()
        out: List[str] = []
        in_fts = False
        for ln in lines:
            if ln.strip().startswith("CREATE VIRTUAL TABLE") and "documents_fts" in ln:
                in_fts = True
            if not in_fts:
                out.append(ln)
            if in_fts and ln.strip().endswith(");"):
                in_fts = False
        return "\n".join(out)

    def _exec_with_retry(self, conn: sqlite3.Connection, query: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        last_err: Optional[Exception] = None
        for _ in range(self.cfg.busy_retries):
            try:
                return conn.execute(query, params)
            except sqlite3.OperationalError as e:
                last_err = e
                msg = str(e).lower()
                if "database is locked" in msg or "database table is locked" in msg:
                    time.sleep(self.cfg.busy_retry_sec)
                    continue
                raise
        raise RuntimeError(f"DB busy after retries: {last_err!r}")

    # ----------------------------
    # Runs
    # ----------------------------
    def run_start(self, mode: str, config_path: Optional[str] = None, notes: Optional[str] = None) -> str:
        run_id = sha256_hex(mode, utc_now_iso(), str(os.getpid()))
        with self.transaction() as conn:
            self._exec_with_retry(
                conn,
                """
                INSERT INTO runs(run_id, started_at_utc, status, mode, config_path, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, utc_now_iso(), "ok", mode, config_path, notes),
            )
        return run_id

    def run_finish(self, run_id: str, status: str = "ok") -> None:
        with self.transaction() as conn:
            self._exec_with_retry(
                conn,
                "UPDATE runs SET finished_at_utc=?, status=? WHERE run_id=?",
                (utc_now_iso(), status, run_id),
            )

    # ----------------------------
    # Sources
    # ----------------------------
    def upsert_source(
        self,
        source_id: str,
        label: Optional[str] = None,
        region: Optional[str] = None,
        category: Optional[str] = None,
        feed_url: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        now = utc_now_iso()
        with self.transaction() as conn:
            self._exec_with_retry(
                conn,
                """
                INSERT INTO sources(source_id, label, region, category, feed_url, enabled, created_at_utc, updated_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                  label=excluded.label,
                  region=excluded.region,
                  category=excluded.category,
                  feed_url=excluded.feed_url,
                  enabled=excluded.enabled,
                  updated_at_utc=excluded.updated_at_utc
                """,
                (source_id, label, region, category, feed_url, 1 if enabled else 0, now, now),
            )

    # ----------------------------
    # Raw items
    # ----------------------------
    def insert_raw_item(
        self,
        *,
        run_id: Optional[str],
        ingested_at_utc: str,
        source_id: str,
        feed_url: Optional[str],
        title: Optional[str],
        url: Optional[str],
        published_raw: Optional[str],
        raw_json_obj: Dict[str, Any],
    ) -> str:
        raw_id = sha256_hex(source_id, url or "", published_raw or "", title or "")
        with self.transaction() as conn:
            self._exec_with_retry(
                conn,
                """
                INSERT OR IGNORE INTO raw_items(
                  raw_id, run_id, ingested_at_utc, source_id, feed_url,
                  title, url, published_raw, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    raw_id,
                    run_id,
                    ingested_at_utc,
                    source_id,
                    feed_url,
                    title,
                    url,
                    published_raw,
                    json.dumps(raw_json_obj, ensure_ascii=False),
                ),
            )
        return raw_id

    # ----------------------------
    # Documents
    # ----------------------------
    def upsert_document(
        self,
        *,
        doc_id: str,
        run_id: Optional[str],
        normalized_at_utc: str,
        ingested_at_utc: Optional[str],
        published_at_utc: Optional[str],
        source_id: str,
        source_label: Optional[str],
        region: Optional[str],
        category: Optional[str],
        feed_url: Optional[str],
        url: Optional[str],
        title: Optional[str],
        body_text: Optional[str],
        raw_json_obj: Optional[Dict[str, Any]] = None,
        write_fts: bool = True,
    ) -> None:
        raw_json = json.dumps(raw_json_obj, ensure_ascii=False) if raw_json_obj is not None else None

        with self.transaction() as conn:
            self._exec_with_retry(
                conn,
                """
                INSERT INTO documents(
                  doc_id, run_id, normalized_at_utc, ingested_at_utc, published_at_utc,
                  source_id, source_label, region, category, feed_url, url,
                  title, body_text, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                  run_id=excluded.run_id,
                  normalized_at_utc=excluded.normalized_at_utc,
                  ingested_at_utc=excluded.ingested_at_utc,
                  published_at_utc=excluded.published_at_utc,
                  source_id=excluded.source_id,
                  source_label=excluded.source_label,
                  region=excluded.region,
                  category=excluded.category,
                  feed_url=excluded.feed_url,
                  url=excluded.url,
                  title=excluded.title,
                  body_text=excluded.body_text,
                  raw_json=excluded.raw_json
                """,
                (
                    doc_id,
                    run_id,
                    normalized_at_utc,
                    ingested_at_utc,
                    published_at_utc,
                    source_id,
                    source_label,
                    region,
                    category,
                    feed_url,
                    url,
                    title,
                    body_text,
                    raw_json,
                ),
            )

            # Optional FTS mirror (safe if table exists; ignore if not)
            if write_fts:
                try:
                    self._exec_with_retry(
                        conn,
                        """
                        INSERT INTO documents_fts(doc_id, title, body_text, source_label, region, category)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (doc_id, title or "", body_text or "", source_label or "", region or "", category or ""),
                    )
                except sqlite3.OperationalError:
                    # FTS table missing (disabled) — that's fine.
                    pass

    def query_documents_window(
        self,
        *,
        published_start_utc: str,
        published_end_utc: str,
        region: Optional[str] = None,
        source_id: Optional[str] = None,
        limit: int = 100000,
    ) -> List[Dict[str, Any]]:
        """
        Query docs by published_at_utc range inclusive.

        published_start_utc / end_utc should be ISO or YYYY-MM-DD boundaries in UTC.
        """
        conn = self.connect()
        where = ["published_at_utc IS NOT NULL", "published_at_utc >= ?", "published_at_utc <= ?"]
        params: List[Any] = [published_start_utc, published_end_utc]

        if region:
            where.append("region = ?")
            params.append(region)
        if source_id:
            where.append("source_id = ?")
            params.append(source_id)

        q = f"""
        SELECT doc_id, published_at_utc, source_id, source_label, region, category, url, title, body_text
        FROM documents
        WHERE {' AND '.join(where)}
        ORDER BY published_at_utc ASC
        LIMIT ?
        """
        params.append(limit)

        cur = self._exec_with_retry(conn, q, params)
        return [dict(r) for r in cur.fetchall()]

    # ----------------------------
    # Analysis results
    # ----------------------------
    def upsert_analysis_result(
        self,
        *,
        run_id: Optional[str],
        module: str,
        window_start_utc: str,
        window_end_utc: str,
        baseline_start_utc: Optional[str],
        baseline_end_utc: Optional[str],
        group_by: Optional[str],
        params: Dict[str, Any],
        result: Dict[str, Any],
    ) -> str:
        result_id = sha256_hex(
            module,
            window_start_utc,
            window_end_utc,
            baseline_start_utc or "",
            baseline_end_utc or "",
            group_by or "",
            json.dumps(params, sort_keys=True),
        )

        with self.transaction() as conn:
            self._exec_with_retry(
                conn,
                """
                INSERT INTO analysis_results(
                  result_id, run_id, created_at_utc, module,
                  window_start_utc, window_end_utc,
                  baseline_start_utc, baseline_end_utc,
                  group_by, params_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(result_id) DO UPDATE SET
                  run_id=excluded.run_id,
                  created_at_utc=excluded.created_at_utc,
                  params_json=excluded.params_json,
                  result_json=excluded.result_json
                """,
                (
                    result_id,
                    run_id,
                    utc_now_iso(),
                    module,
                    window_start_utc,
                    window_end_utc,
                    baseline_start_utc,
                    baseline_end_utc,
                    group_by,
                    json.dumps(params, ensure_ascii=False, sort_keys=True),
                    json.dumps(result, ensure_ascii=False),
                ),
            )
        return result_id


# ----------------------------
# Convenience factory
# ----------------------------
def open_db(db_path: Optional[str] = None) -> DB:
    cfg = DBConfig(path=db_path or DEFAULT_DB_PATH)
    return DB(cfg)
