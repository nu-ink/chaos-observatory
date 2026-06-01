from __future__ import annotations

from chaos_observatory.health.run_healthcheck import project_root, run_all


def test_healthcheck_passes() -> None:
    rows = run_all(project_root())

    assert rows
    assert all(row["ok"] for row in rows)
    assert any(row["check"] == "logs writable" for row in rows)
    assert any(row["check"].startswith("dry-run ml/") for row in rows)
