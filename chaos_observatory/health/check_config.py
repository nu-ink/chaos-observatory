from __future__ import annotations

from pathlib import Path

import yaml


def run(project_root: Path) -> list[dict]:
    config_path = project_root / "config" / "chaos.yaml"
    results: list[dict] = []

    if not config_path.exists():
        return [_result("config file exists", False, f"Missing {config_path}")]

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return [_result("config loads", False, str(exc))]

    results.append(_result("config loads", True, str(config_path)))

    for key in ("version", "paths", "ingest", "analyze", "report"):
        results.append(_result(f"config has {key}", key in config, key))

    paths = config.get("paths") or {}
    required_dirs = [
        paths.get("data_raw", "data/raw"),
        paths.get("data_normalized", "data/normalized"),
        paths.get("reports", "reports"),
        "logs",
        "storage",
    ]
    for rel_path in required_dirs:
        path = project_root / rel_path
        results.append(
            _result(
                f"directory present: {rel_path}",
                path.exists() and path.is_dir(),
                str(path),
            )
        )

    logs_dir = project_root / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        probe = logs_dir / ".healthcheck.tmp"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        results.append(_result("logs writable", True, str(logs_dir)))
    except Exception as exc:
        results.append(_result("logs writable", False, str(exc)))

    return results


def _result(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "ok": ok, "detail": detail}
