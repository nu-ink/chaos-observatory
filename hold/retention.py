#!/usr/bin/env python3
"""
Chaos-Observatory Retention (v1)

Manages disk retention for date-partitioned ingestion outputs.

Expected directory layout (UTC partitions):
  data/raw/YYYY-MM-DD/<source_id>.jsonl
  data/normalized/YYYY-MM-DD/<source_id>.jsonl

Policy model:
- HOLD WINDOW: Never delete partitions newer than --hold-days
- KEEP WINDOW: Delete (or archive) partitions older than --keep-days

Typical example:
- hold_days = 7   (protect last week for analysis)
- keep_days = 30  (retain one month on disk)
- archive_dir set (optional) to move very old partitions elsewhere

Safety:
- Defaults to dry-run (no deletes)
- Requires --apply to actually delete/move
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class PartitionAction:
    base: str                # e.g., "raw" or "normalized"
    partition: str           # YYYY-MM-DD
    path: Path               # full path to partition dir
    age_days: int
    action: str              # "skip_hold", "keep", "archive", "delete"
    reason: str


def utc_today_date() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def parse_partition_date(name: str) -> Optional[datetime]:
    if not DATE_DIR_RE.match(name):
        return None
    try:
        dt = datetime.strptime(name, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def compute_age_days(partition_date: datetime, today: datetime) -> int:
    # Age in whole days, based on midnight UTC boundaries
    delta = today - partition_date
    return int(delta.days)


def list_partitions(base_dir: Path) -> List[Tuple[str, Path, datetime]]:
    """
    Returns [(partition_name, partition_path, partition_date_utc), ...]
    """
    if not base_dir.exists():
        return []

    parts: List[Tuple[str, Path, datetime]] = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        pdt = parse_partition_date(child.name)
        if not pdt:
            continue
        parts.append((child.name, child, pdt))
    return parts


def plan_actions_for_base(
    base_label: str,
    base_dir: Path,
    hold_days: int,
    keep_days: int,
    today: datetime,
    archive_dir: Optional[Path],
) -> List[PartitionAction]:
    actions: List[PartitionAction] = []

    for name, path, pdt in list_partitions(base_dir):
        age = compute_age_days(pdt, today)

        # Protect hold window: never delete/archive these.
        if age <= hold_days:
            actions.append(
                PartitionAction(
                    base=base_label,
                    partition=name,
                    path=path,
                    age_days=age,
                    action="skip_hold",
                    reason=f"within hold window (age_days={age} <= hold_days={hold_days})",
                )
            )
            continue

        # Keep window: if not older than keep_days, keep it.
        if age <= keep_days:
            actions.append(
                PartitionAction(
                    base=base_label,
                    partition=name,
                    path=path,
                    age_days=age,
                    action="keep",
                    reason=f"within keep window (age_days={age} <= keep_days={keep_days})",
                )
            )
            continue

        # Beyond keep window: archive if archive_dir provided, else delete.
        if archive_dir:
            actions.append(
                PartitionAction(
                    base=base_label,
                    partition=name,
                    path=path,
                    age_days=age,
                    action="archive",
                    reason=f"older than keep_days (age_days={age} > keep_days={keep_days}) and archive_dir set",
                )
            )
        else:
            actions.append(
                PartitionAction(
                    base=base_label,
                    partition=name,
                    path=path,
                    age_days=age,
                    action="delete",
                    reason=f"older than keep_days (age_days={age} > keep_days={keep_days})",
                )
            )

    return actions


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def move_partition_to_archive(
    action: PartitionAction,
    archive_root: Path,
) -> Path:
    """
    Moves partition directory to:
      <archive_root>/<base>/<YYYY-MM-DD>/
    """
    dest = archive_root / action.base / action.partition
    ensure_dir(dest.parent)

    # If destination exists, avoid clobbering. Append suffix.
    if dest.exists():
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = archive_root / action.base / f"{action.partition}__{suffix}"

    shutil.move(str(action.path), str(dest))
    return dest


def delete_partition(action: PartitionAction) -> None:
    shutil.rmtree(action.path)


def emit(event: str, payload: Dict) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw", help="Base raw directory")
    ap.add_argument("--normalized-dir", default="data/normalized", help="Base normalized directory")
    ap.add_argument("--hold-days", type=int, default=7, help="Protect newest N days from deletion/archive")
    ap.add_argument("--keep-days", type=int, default=30, help="Keep newest N days on disk (older will be deleted/archived)")
    ap.add_argument("--archive-dir", default=None, help="If set, move old partitions here instead of deleting")
    ap.add_argument("--apply", action="store_true", help="Actually perform deletes/moves (otherwise dry-run)")
    ap.add_argument("--only", choices=["raw", "normalized", "both"], default="both", help="Which stores to process")
    args = ap.parse_args()

    if args.hold_days < 0 or args.keep_days < 0:
        raise SystemExit("ERROR: hold-days and keep-days must be >= 0")
    if args.keep_days < args.hold_days:
        raise SystemExit("ERROR: keep-days must be >= hold-days")

    raw_dir = Path(args.raw_dir)
    norm_dir = Path(args.normalized_dir)
    archive_dir = Path(args.archive_dir) if args.archive_dir else None

    today = utc_today_date()

    emit(
        "retention_start",
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "today_utc": today.isoformat(timespec="seconds"),
            "raw_dir": str(raw_dir),
            "normalized_dir": str(norm_dir),
            "hold_days": args.hold_days,
            "keep_days": args.keep_days,
            "archive_dir": str(archive_dir) if archive_dir else None,
            "mode": "apply" if args.apply else "dry_run",
            "only": args.only,
        },
    )

    planned: List[PartitionAction] = []

    if args.only in ("raw", "both"):
        planned.extend(
            plan_actions_for_base(
                base_label="raw",
                base_dir=raw_dir,
                hold_days=args.hold_days,
                keep_days=args.keep_days,
                today=today,
                archive_dir=archive_dir,
            )
        )

    if args.only in ("normalized", "both"):
        planned.extend(
            plan_actions_for_base(
                base_label="normalized",
                base_dir=norm_dir,
                hold_days=args.hold_days,
                keep_days=args.keep_days,
                today=today,
                archive_dir=archive_dir,
            )
        )

    # Summarize plan
    counts: Dict[str, int] = {"skip_hold": 0, "keep": 0, "archive": 0, "delete": 0}
    for a in planned:
        counts[a.action] = counts.get(a.action, 0) + 1

    emit("retention_plan", {"counts": counts, "total_partitions": len(planned)})

    # Execute
    archived = 0
    deleted = 0
    errors = 0

    for a in planned:
        emit(
            "partition_decision",
            {
                "base": a.base,
                "partition": a.partition,
                "path": str(a.path),
                "age_days": a.age_days,
                "action": a.action,
                "reason": a.reason,
            },
        )

        if not args.apply:
            continue

        try:
            if a.action == "archive":
                assert archive_dir is not None
                dest = move_partition_to_archive(a, archive_dir)
                archived += 1
                emit(
                    "partition_archived",
                    {
                        "base": a.base,
                        "partition": a.partition,
                        "from": str(a.path),
                        "to": str(dest),
                        "age_days": a.age_days,
                    },
                )
            elif a.action == "delete":
                delete_partition(a)
                deleted += 1
                emit(
                    "partition_deleted",
                    {
                        "base": a.base,
                        "partition": a.partition,
                        "path": str(a.path),
                        "age_days": a.age_days,
                    },
                )
        except Exception as ex:
            errors += 1
            emit(
                "partition_error",
                {
                    "base": a.base,
                    "partition": a.partition,
                    "path": str(a.path),
                    "action": a.action,
                    "error": repr(ex),
                },
            )

    emit(
        "retention_done",
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "archived": archived,
            "deleted": deleted,
            "errors": errors,
            "mode": "apply" if args.apply else "dry_run",
        },
    )

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
