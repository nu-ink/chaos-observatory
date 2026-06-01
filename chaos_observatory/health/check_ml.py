from __future__ import annotations

import json
import importlib
import subprocess
import sys
from pathlib import Path


ML_MODULES = [
    "ml",
    "ml.ml_change_detection",
    "ml.ml_topic_convergence",
    "ml.ml_embeddings",
    "ml.ml_evaluation",
    "ml.ml_semantic_linker",
    "ml.ml_sentiment_shift",
    "ml.ml_similarity_thresholds",
    "ml.vector_store",
]

ML_DRY_RUNS = [
    ["ml/ml_change_detection.py", "--dry-run"],
    ["ml/ml_topic_convergence.py", "--dry-run"],
    ["ml/ml_sentiment_shift.py", "--dry-run"],
]


def run(project_root: Path) -> list[dict]:
    results: list[dict] = []

    for module_name in ML_MODULES:
        try:
            importlib.import_module(module_name)
            results.append(_result(f"import {module_name}", True, ""))
        except Exception as exc:
            results.append(_result(f"import {module_name}", False, str(exc)))

    for command in ML_DRY_RUNS:
        script = command[0]
        try:
            completed = subprocess.run(
                [sys.executable, *command],
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = (completed.stdout or completed.stderr).strip()
            detail = _summarize_output(output, completed.returncode)
            results.append(
                _result(
                    f"dry-run {script}",
                    completed.returncode == 0,
                    detail,
                )
            )
        except Exception as exc:
            results.append(_result(f"dry-run {script}", False, str(exc)))

    return results


def _result(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "ok": ok, "detail": detail}


def _summarize_output(output: str, returncode: int) -> str:
    if not output:
        return f"exit={returncode}"
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return output.splitlines()[0]
    event = payload.get("event")
    status = payload.get("status")
    if event and status:
        return f"{event} ({status})"
    return event or status or f"exit={returncode}"
