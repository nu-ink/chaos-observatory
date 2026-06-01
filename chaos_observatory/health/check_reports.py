from __future__ import annotations

import json
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from report.weekly_report import main as weekly_report_main


def run(project_root: Path) -> list[dict]:
    results: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="chaos-health-report-") as tmp:
        tmp_path = Path(tmp)
        normalized_dir = tmp_path / "normalized"
        report_dir = tmp_path / "reports"
        partition = normalized_dir / "2026-06-01"
        partition.mkdir(parents=True, exist_ok=True)
        sample = {
            "source_id": "health",
            "source_label": "Health Source",
            "published_at_utc": "2026-06-01T00:00:00Z",
            "title": "Health report item",
            "body_text": "Emergency teams reported stable operations and routine support.",
        }
        (partition / "health.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

        try:
            with redirect_stdout(StringIO()):
                exit_code = weekly_report_main(
                    [
                        "--normalized-dir",
                        str(normalized_dir),
                        "--outdir",
                        str(report_dir),
                        "--end-date",
                        "2026-06-01",
                        "--window-days",
                        "1",
                        "--baseline-days",
                        "0",
                    ]
                )
            report_path = report_dir / "2026-06-01" / "weekly_report.md"
            results.append(
                _result(
                    "weekly report generates",
                    exit_code == 0 and report_path.exists(),
                    str(report_path),
                )
            )
        except Exception as exc:
            results.append(_result("weekly report generates", False, str(exc)))

    return results


def _result(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "ok": ok, "detail": detail}
