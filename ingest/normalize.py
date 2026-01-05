#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, Dict


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_row(row: Dict) -> Dict:
    """
    Normalize a single raw RSS item into a canonical shape.
    Adjust fields here as your schema evolves.
    """
    return {
        "source_id": row.get("source_id"),
        "title": row.get("title"),
        "url": row.get("link") or row.get("url"),
        "published_ts": row.get("published"),
        "summary": row.get("summary") or row.get("description"),
        "text": row.get("content") or row.get("summary"),
    }


def normalize_file(inp: Path, out: Path) -> int:
    normalized_rows = (normalize_row(r) for r in read_jsonl(inp))
    written = write_jsonl(out, normalized_rows)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="Input JSONL file or directory")
    parser.add_argument("--out", dest="out", required=True, help="Output JSONL file or directory")
    args = parser.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    if not inp.exists():
        raise SystemExit(f"Input path does not exist: {inp}")

    # ---- FILE → FILE or DIR
    if inp.is_file():
        if out.is_dir():
            out_file = out / inp.name
        else:
            out_file = out

        written = normalize_file(inp, out_file)
        print(f"[normalize] {inp} -> {out_file} ({written} rows)")
        return 0

    # ---- DIR → DIR
    if inp.is_dir():
        if out.exists() and out.is_file():
            raise SystemExit("--out must be a directory when --in is a directory")

        out.mkdir(parents=True, exist_ok=True)

        files = sorted(inp.glob("*.jsonl"))
        if not files:
            print(f"[normalize] No JSONL files found in {inp}")
            return 0

        total = 0
        for f in files:
            out_file = out / f.name
            written = normalize_file(f, out_file)
            total += written
            print(f"[normalize] {f.name} -> {out_file.name} ({written} rows)")

        print(f"[normalize] DONE — {len(files)} files, {total} rows")
        return 0

    raise SystemExit("Unsupported input type")


if __name__ == "__main__":
    raise SystemExit(main())
