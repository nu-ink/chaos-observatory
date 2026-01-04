#!/usr/bin/env python3
"""
Chaos-Observatory Normalizer (v1)

Reads raw JSONL produced by rss_collector.py and outputs normalized JSONL.

Normalization goals:
- Stable IDs (hash-based)
- Clean text fields (title/body)
- Best-effort timestamp parsing
- Preserve raw for audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dateutil import parser as dtparser  # pip install python-dateutil


WS_RE = re.compile(r"\s+")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = WS_RE.sub(" ", s).strip()
    return s


def parse_time_best_effort(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        dt = dtparser.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return None


def stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def normalize_row(raw_row: Dict[str, Any]) -> Dict[str, Any]:
    src = raw_row.get("source", {}) or {}
    item = raw_row.get("item", {}) or {}

    title = clean_text(item.get("title"))
    link = clean_text(item.get("link"))
    summary = clean_text(item.get("summary"))
    published_raw = item.get("published")

    published_utc = parse_time_best_effort(published_raw)

    doc_id = stable_id(
        src.get("id", ""),
        link or "",
        title or "",
        published_utc or "",
    )

    # “body_text” is summary/description for RSS v1; later you can fetch full article text if you want.
    body_text = summary

    return {
        "id": doc_id,
        "source_id": src.get("id"),
        "source_label": src.get("label"),
        "region": src.get("region"),
        "category": src.get("category"),
        "feed_url": src.get("feed_url"),
        "ingested_at_utc": raw_row.get("ingested_at_utc"),
        "published_at_utc": published_utc,
        "title": title,
        "body_text": body_text,
        "url": link,
        # Keep raw for audit / replay.
        "raw": raw_row,
        "normalized_at_utc": _utc_now_iso(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input raw JSONL file")
    ap.add_argument("--out", dest="out", required=True, help="Output normalized JSONL file")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    if not inp.exists():
        print(f"ERROR: input not found: {inp}", file=sys.stderr)
        return 2

    normalized_rows = (normalize_row(r) for r in read_jsonl(inp))
    written = write_jsonl(out, normalized_rows)

    print(json.dumps({"event": "normalized", "in": str(inp), "out": str(out), "rows": written}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
