from __future__ import annotations

from pathlib import Path

from ingest.normalize import normalize_row
from ingest.rss_collector import load_sources_yaml


def run(project_root: Path) -> list[dict]:
    results: list[dict] = []
    sources_path = project_root / "ingest" / "sources.yaml"

    try:
        _cfg, sources = load_sources_yaml(sources_path)
        results.append(
            _result(
                "sources yaml loads", bool(sources), f"{len(sources)} enabled sources"
            )
        )
    except Exception as exc:
        results.append(_result("sources yaml loads", False, str(exc)))

    try:
        normalized = normalize_row(
            {
                "source": {
                    "id": "health",
                    "label": "Health Source",
                    "region": "global",
                    "category": "test",
                },
                "item": {
                    "title": "Health check item",
                    "link": "https://example.test/health",
                    "summary": "A minimal RSS item for normalization.",
                    "published": "2026-06-01T00:00:00Z",
                },
                "ingested_at_utc": "2026-06-01T00:01:00Z",
            }
        )
        ok = (
            normalized["source_id"] == "health"
            and normalized["title"] == "Health check item"
            and normalized["body_text"] == "A minimal RSS item for normalization."
        )
        results.append(
            _result("test RSS item normalizes", ok, normalized.get("title") or "")
        )
    except Exception as exc:
        results.append(_result("test RSS item normalizes", False, str(exc)))

    return results


def _result(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "ok": ok, "detail": detail}
