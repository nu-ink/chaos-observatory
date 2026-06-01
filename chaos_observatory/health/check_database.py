from __future__ import annotations

import sqlite3
from pathlib import Path


def run(project_root: Path) -> list[dict]:
    db_path = project_root / "storage" / "chaos.db"
    schema_path = project_root / "storage" / "schema.sql"
    results = [
        _result("schema file exists", schema_path.exists(), str(schema_path)),
        _result("database path parent exists", db_path.parent.exists(), str(db_path.parent)),
    ]

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("SELECT 1").fetchone()
            conn.execute("PRAGMA foreign_keys = ON")
        results.append(_result("database connects", True, str(db_path)))
    except Exception as exc:
        results.append(_result("database connects", False, str(exc)))

    return results


def _result(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "ok": ok, "detail": detail}
