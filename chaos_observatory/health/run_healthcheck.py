from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaos_observatory.health import (
    check_config,
    check_database,
    check_ingest,
    check_ml,
    check_reports,
)

CHECK_MODULES = [
    ("config", check_config),
    ("database", check_database),
    ("ingest", check_ingest),
    ("ml", check_ml),
    ("reports", check_reports),
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_all(root: Path) -> list[dict]:
    rows: list[dict] = []
    for group, module in CHECK_MODULES:
        for result in module.run(root):
            rows.append({"group": group, **result})
    return rows


def print_table(rows: list[dict]) -> None:
    print("Chaos Observatory Health Check")
    print("=" * 31)
    for row in rows:
        status = "PASS" if row["ok"] else "FAIL"
        detail = f" - {row['detail']}" if row.get("detail") else ""
        print(f"[{status}] {row['group']}: {row['check']}{detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Chaos Observatory pre-flight health checks."
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of a text table"
    )
    args = parser.parse_args(argv)

    rows = run_all(project_root())
    ok = all(row["ok"] for row in rows)

    if args.json:
        print(json.dumps({"ok": ok, "results": rows}, indent=2, sort_keys=True))
    else:
        print_table(rows)
        print("")
        print(f"Overall: {'PASS' if ok else 'FAIL'}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
