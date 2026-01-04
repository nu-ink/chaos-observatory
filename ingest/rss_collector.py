#!/usr/bin/env python3
"""
Chaos-Observatory RSS Collector (v1)

- Reads ingest/sources.yaml
- Fetches enabled RSS feeds
- Writes raw items to JSONL under data/raw/YYYY-MM-DD/<source_id>.jsonl
- Each line = one raw item (minimal transformation)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import feedparser
import yaml


@dataclass
class FeedConfig:
    url: str


@dataclass
class SourceConfig:
    id: str
    label: str
    kind: str
    region: str
    category: str
    enabled: bool
    feeds: List[FeedConfig]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_sources_yaml(path: Path) -> Tuple[Dict[str, Any], List[SourceConfig]]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    defaults = cfg.get("defaults", {})
    sources_raw = cfg.get("sources", [])

    sources: List[SourceConfig] = []
    for s in sources_raw:
        if not s.get("enabled", defaults.get("enabled", True)):
            continue

        feeds = [FeedConfig(url=f["url"]) for f in s.get("feeds", [])]
        sources.append(
            SourceConfig(
                id=s["id"],
                label=s.get("label", s["id"]),
                kind=s.get("kind", "rss"),
                region=s.get("region", "unknown"),
                category=s.get("category", "unknown"),
                enabled=True,
                feeds=feeds,
            )
        )

    return cfg, sources


def safe_get(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def normalize_entry_minimal(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep this *raw-ish*. Don't overthink.
    Full normalization happens in normalize.py.
    """
    # feedparser returns objects that behave like dicts; cast to dict carefully
    title = entry.get("title")
    link = entry.get("link")
    published = entry.get("published") or entry.get("updated")
    summary = entry.get("summary") or entry.get("description")

    return {
        "title": title,
        "link": link,
        "published": published,
        "summary": summary,
        "raw": dict(entry),
    }


def fetch_feed(url: str, timeout_sec: int, user_agent: str) -> feedparser.FeedParserDict:
    # feedparser supports request headers via its internal urllib;
    # setting a global UA via feedparser.USER_AGENT is common practice.
    feedparser.USER_AGENT = user_agent
    parsed = feedparser.parse(url)
    # If you need hard timeouts per request later, we can swap to requests + feedparser.parse(data)
    return parsed


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="ingest/sources.yaml", help="Path to sources.yaml")
    ap.add_argument("--outdir", default="data/raw", help="Base output directory for raw JSONL")
    ap.add_argument("--max-items", type=int, default=None, help="Override max_items_per_feed from YAML")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep between feed fetches (seconds)")
    args = ap.parse_args()

    sources_path = Path(args.sources)
    if not sources_path.exists():
        print(f"ERROR: sources file not found: {sources_path}", file=sys.stderr)
        return 2

    cfg, sources = load_sources_yaml(sources_path)
    defaults = cfg.get("defaults", {})
    timeout_sec = int(defaults.get("fetch_timeout_sec", 20))
    user_agent = str(defaults.get("user_agent", "Chaos-Observatory/0.1"))
    max_items_default = int(defaults.get("max_items_per_feed", 50))
    max_items = args.max_items if args.max_items is not None else max_items_default

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_base = Path(args.outdir) / day

    run_meta = {
        "run_started_utc": _utc_now_iso(),
        "sources_file": str(sources_path),
        "max_items_per_feed": max_items,
    }

    print(json.dumps({"event": "collector_start", **run_meta}, ensure_ascii=False))

    total_written = 0

    for src in sources:
        if src.kind != "rss":
            continue

        for feed in src.feeds:
            try:
                parsed = fetch_feed(feed.url, timeout_sec=timeout_sec, user_agent=user_agent)

                if getattr(parsed, "bozo", False):
                    # bozo_exception is often non-fatal but indicates parsing issues
                    bozo_exc = getattr(parsed, "bozo_exception", None)
                    print(
                        json.dumps(
                            {
                                "event": "feed_parse_warning",
                                "source_id": src.id,
                                "feed_url": feed.url,
                                "warning": str(bozo_exc),
                                "ts_utc": _utc_now_iso(),
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                    )

                entries = list(parsed.get("entries", []))[:max_items]

                rows = []
                for e in entries:
                    rows.append(
                        {
                            "ingested_at_utc": _utc_now_iso(),
                            "source": {
                                "id": src.id,
                                "label": src.label,
                                "region": src.region,
                                "category": src.category,
                                "feed_url": feed.url,
                            },
                            "item": normalize_entry_minimal(e),
                        }
                    )

                out_path = out_base / f"{src.id}.jsonl"
                written = write_jsonl(out_path, rows)
                total_written += written

                print(
                    json.dumps(
                        {
                            "event": "feed_fetched",
                            "source_id": src.id,
                            "feed_url": feed.url,
                            "items": len(entries),
                            "written": written,
                            "out": str(out_path),
                            "ts_utc": _utc_now_iso(),
                        },
                        ensure_ascii=False,
                    )
                )

            except Exception as ex:
                print(
                    json.dumps(
                        {
                            "event": "feed_fetch_error",
                            "source_id": src.id,
                            "feed_url": feed.url,
                            "error": repr(ex),
                            "ts_utc": _utc_now_iso(),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )

            time.sleep(max(args.sleep, 0.0))

    print(
        json.dumps(
            {"event": "collector_done", "total_written": total_written, "ts_utc": _utc_now_iso()},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
