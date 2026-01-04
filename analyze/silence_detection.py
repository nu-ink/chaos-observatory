#!/usr/bin/env python3
"""
Chaos-Observatory: Silence Detection Analyzer (robust v1)

Purpose:
  Identify meaningful topics/terms that were present in a baseline window but become rare
  or absent in the current window (a "silence" signal).

Input:
  Normalized JSONL partitions:
    data/normalized/YYYY-MM-DD/*.jsonl

Silence signals:
  1) Global silence (all docs)
  2) Per-group silence (by region or source)
  3) Group/source dropouts (volume collapse)

Methods:
  - Tokenization + (optional) ngrams
  - Baseline term presence measured by:
      * baseline_count (token count)
      * baseline_df (document frequency)
  - Current presence measured similarly
  - Scoring uses:
      * drop_ratio = (cur_rate + eps) / (base_rate + eps)
      * silence_score = -log(drop_ratio)  (bigger is "more silent")
    where rate = count / total_tokens

Outputs:
  JSON to stdout; optional Markdown summary.

Design:
  Deterministic, explainable. No embeddings required.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ----------------------------
# Stopwords (extend later)
# ----------------------------
STOPWORDS: Set[str] = {
    "a","an","and","are","as","at","be","been","but","by","can","could","did","do","does",
    "for","from","had","has","have","he","her","his","how","i","if","in","into","is","it",
    "its","just","may","might","more","most","must","not","of","on","or","our","out","over",
    "s","said","she","should","so","some","than","that","the","their","them","then","there",
    "these","they","this","to","too","under","up","was","we","were","what","when","where",
    "which","who","will","with","would","you","your",
}

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{1,}")


@dataclass(frozen=True)
class Doc:
    group: str
    source_label: str
    region: str
    title: str
    body_text: str


def utc_midnight_today() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def parse_ymd(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def day_to_partition(day: datetime) -> str:
    return day.strftime("%Y-%m-%d")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_docs(base_dir: Path, start_day: datetime, end_day: datetime, group_by: str) -> List[Doc]:
    docs: List[Doc] = []
    cur = start_day
    while cur <= end_day:
        part = base_dir / day_to_partition(cur)
        if part.exists() and part.is_dir():
            for f in part.glob("*.jsonl"):
                for row in read_jsonl(f):
                    source_label = row.get("source_label") or row.get("source_id") or "unknown"
                    region = row.get("region") or "unknown"
                    if group_by == "region":
                        group = region
                    elif group_by == "source":
                        group = source_label
                    else:
                        group = "unknown"

                    docs.append(
                        Doc(
                            group=group,
                            source_label=source_label,
                            region=region,
                            title=row.get("title") or "",
                            body_text=row.get("body_text") or "",
                        )
                    )
        cur += timedelta(days=1)
    return docs


def tokenize(text: str, stopwords: Set[str], min_len: int) -> List[str]:
    text = text.lower()
    toks = TOKEN_RE.findall(text)
    toks = [t for t in toks if len(t) >= min_len and t not in stopwords]
    return toks


def ngrams(tokens: Sequence[str], n: int) -> List[str]:
    if n <= 1:
        return list(tokens)
    out: List[str] = []
    for i in range(0, len(tokens) - n + 1):
        out.append(" ".join(tokens[i:i+n]))
    return out


def parse_ngrams_arg(s: str) -> List[int]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1 or n > 4:
            raise ValueError("ngrams must be between 1 and 4")
        vals.append(n)
    return sorted(set(vals))


def build_counts(
    docs: List[Doc],
    ngram_list: List[int],
    stopwords: Set[str],
    min_token_len: int,
) -> Tuple[Counter, Counter, int]:
    """
    Returns:
      counts: token counts
      df: document frequency (# docs containing term)
      total_terms: total token instances across docs (sum counts)
    """
    counts = Counter()
    df = Counter()
    total_terms = 0

    for d in docs:
        toks = tokenize(d.title + " " + d.body_text, stopwords=stopwords, min_len=min_token_len)
        terms_in_doc: Set[str] = set()

        for n in ngram_list:
            grams = ngrams(toks, n)
            counts.update(grams)
            total_terms += len(grams)
            terms_in_doc.update(grams)

        df.update(terms_in_doc)

    return counts, df, total_terms


def silence_rank(
    base_counts: Counter,
    base_df: Counter,
    base_total: int,
    cur_counts: Counter,
    cur_df: Counter,
    cur_total: int,
    *,
    min_base_count: int,
    min_base_df: int,
    max_cur_count: int,
    max_cur_df: int,
    eps: float = 1e-12,
    top: int = 50,
) -> List[dict]:
    """
    Rank terms that were meaningful in baseline but rare in current.

    Conditions:
      baseline_count >= min_base_count
      baseline_df >= min_base_df
      current_count <= max_cur_count
      current_df <= max_cur_df

    Score:
      base_rate = base_count / base_total
      cur_rate = cur_count / cur_total
      drop_ratio = (cur_rate + eps) / (base_rate + eps)
      silence_score = -log(drop_ratio)
    """
    if base_total <= 0 or cur_total <= 0:
        return []

    rows: List[dict] = []
    for term, bcnt in base_counts.items():
        bdf = int(base_df.get(term, 0))
        if bcnt < min_base_count or bdf < min_base_df:
            continue

        ccnt = int(cur_counts.get(term, 0))
        cdf = int(cur_df.get(term, 0))
        if ccnt > max_cur_count or cdf > max_cur_df:
            continue

        base_rate = bcnt / base_total
        cur_rate = ccnt / cur_total
        drop_ratio = (cur_rate + eps) / (base_rate + eps)
        silence_score = -math.log(drop_ratio)

        rows.append(
            {
                "term": term,
                "silence_score": float(silence_score),
                "drop_ratio": float(drop_ratio),
                "base_count": int(bcnt),
                "base_df": int(bdf),
                "cur_count": int(ccnt),
                "cur_df": int(cdf),
                "base_rate": float(base_rate),
                "cur_rate": float(cur_rate),
            }
        )

    rows.sort(key=lambda r: r["silence_score"], reverse=True)
    return rows[:top]


def volume_dropouts(
    base_docs: List[Doc],
    cur_docs: List[Doc],
    group_by: str,
    min_base_docs: int,
    drop_ratio_threshold: float,
    top: int,
) -> List[dict]:
    """
    Identify groups whose doc volume drops sharply from baseline to current.
    """
    base_by = Counter(d.group for d in base_docs) if group_by else Counter()
    cur_by = Counter(d.group for d in cur_docs) if group_by else Counter()

    rows: List[dict] = []
    groups = set(base_by.keys()) | set(cur_by.keys())
    for g in groups:
        b = int(base_by.get(g, 0))
        c = int(cur_by.get(g, 0))
        if b < min_base_docs:
            continue
        ratio = (c + 1e-9) / (b + 1e-9)
        if ratio <= drop_ratio_threshold:
            rows.append(
                {
                    "group": g,
                    "baseline_docs": b,
                    "current_docs": c,
                    "doc_ratio": float(ratio),
                    "drop_factor": float((b + 1e-9) / (c + 1e-9)),
                }
            )
    rows.sort(key=lambda r: r["doc_ratio"])
    return rows[:top]


def write_markdown(out_path: Path, result: dict) -> None:
    meta = result["meta"]
    lines: List[str] = []
    lines.append("# Chaos Observatory — Silence Detection")
    lines.append("")
    lines.append(f"**Generated (UTC):** {meta['generated_at_utc']}")
    lines.append(f"**Current window (UTC):** {meta['current_start']} → {meta['current_end']}  ")
    lines.append(f"**Baseline window (UTC):** {meta['baseline_start']} → {meta['baseline_end']}")
    lines.append(f"**Group-by:** {meta['group_by']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Docs (current) | {meta['docs_current']} |")
    lines.append(f"| Docs (baseline) | {meta['docs_baseline']} |")
    lines.append(f"| Tokens (current) | {meta['tokens_current']} |")
    lines.append(f"| Tokens (baseline) | {meta['tokens_baseline']} |")
    lines.append(f"| N-grams | {', '.join(map(str, meta['ngrams']))} |")
    lines.append("")

    # Global silence
    lines.append("## Global Silence Signals")
    lines.append("")
    lines.append("| Term | Silence Score | Drop Ratio | Base Cnt | Cur Cnt | Base DF | Cur DF |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in result["global_silence"][:40]:
        term = r["term"].replace("|", "\\|")
        lines.append(
            f"| {term} | {r['silence_score']:.3f} | {r['drop_ratio']:.6f} | "
            f"{r['base_count']} | {r['cur_count']} | {r['base_df']} | {r['cur_df']} |"
        )
    lines.append("")

    # Volume dropouts
    lines.append("## Group Volume Dropouts")
    lines.append("")
    lines.append("> Groups with large baseline volume that collapsed in the current window.")
    lines.append("")
    lines.append("| Group | Baseline Docs | Current Docs | Doc Ratio | Drop Factor |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in result["volume_dropouts"][:30]:
        g = str(r["group"]).replace("|", "\\|")
        lines.append(
            f"| {g} | {r['baseline_docs']} | {r['current_docs']} | {r['doc_ratio']:.4f} | {r['drop_factor']:.2f} |"
        )
    lines.append("")

    # Per-group silence (optional)
    per_group = result.get("per_group_silence", {})
    if per_group:
        lines.append("## Per-Group Silence (Top)")
        lines.append("")
        for g, payload in list(per_group.items())[:10]:
            lines.append(f"### {g}")
            lines.append("")
            lines.append("| Term | Silence Score | Base Cnt | Cur Cnt | Base DF | Cur DF |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for r in payload["silence"][:15]:
                term = r["term"].replace("|", "\\|")
                lines.append(
                    f"| {term} | {r['silence_score']:.3f} | {r['base_count']} | {r['cur_count']} | {r['base_df']} | {r['cur_df']} |"
                )
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized-dir", default="data/normalized", help="Base normalized partition dir")
    ap.add_argument("--end-date", default=None, help="UTC end date YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--window-days", type=int, default=7, help="Current window length in days")
    ap.add_argument("--baseline-days", type=int, default=7, help="Baseline window length in days")
    ap.add_argument("--group-by", choices=["region", "source"], default="region", help="Group docs by region or source")
    ap.add_argument("--ngrams", default="1,2", help="Comma list: 1,2 or 1 (max 4)")
    ap.add_argument("--min-token-len", type=int, default=3, help="Minimum token length for unigrams")
    ap.add_argument("--min-docs", type=int, default=10, help="Require at least this many docs per window (global)")
    ap.add_argument("--min-base-count", type=int, default=18, help="Minimum baseline token count for candidate term")
    ap.add_argument("--min-base-df", type=int, default=6, help="Minimum baseline doc frequency for candidate term")
    ap.add_argument("--max-cur-count", type=int, default=0, help="Max current token count for 'silenced' term (0=vanished)")
    ap.add_argument("--max-cur-df", type=int, default=0, help="Max current doc frequency for 'silenced' term")
    ap.add_argument("--top", type=int, default=60, help="Top global silence terms to return")
    ap.add_argument("--per-group", action="store_true", help="Also compute silence per group")
    ap.add_argument("--min-docs-per-group", type=int, default=10, help="Min docs per group per window for per-group silence")
    ap.add_argument("--dropout-min-base-docs", type=int, default=10, help="Min baseline docs for group dropout detection")
    ap.add_argument("--dropout-ratio", type=float, default=0.30, help="Flag groups with current/baseline doc ratio <= this")
    ap.add_argument("--md-out", default=None, help="Optional markdown output path")
    args = ap.parse_args()

    if args.window_days <= 0 or args.baseline_days <= 0:
        raise SystemExit("ERROR: window-days and baseline-days must be > 0")
    if args.keep_days if False else False:
        pass

    base_dir = Path(args.normalized_dir)
    end_day = parse_ymd(args.end_date) if args.end_date else utc_midnight_today()
    cur_start = end_day - timedelta(days=args.window_days - 1)

    base_end = cur_start - timedelta(days=1)
    base_start = base_end - timedelta(days=args.baseline_days - 1)

    ngram_list = parse_ngrams_arg(args.ngrams)

    docs_cur = load_docs(base_dir, cur_start, end_day, group_by=args.group_by)
    docs_base = load_docs(base_dir, base_start, base_end, group_by=args.group_by)

    if len(docs_cur) < args.min_docs or len(docs_base) < args.min_docs:
        out = {
            "event": "silence_detection",
            "error": "insufficient_docs",
            "docs_current": len(docs_cur),
            "docs_baseline": len(docs_base),
            "min_docs": args.min_docs,
            "current_window": {"start": day_to_partition(cur_start), "end": day_to_partition(end_day)},
            "baseline_window": {"start": day_to_partition(base_start), "end": day_to_partition(base_end)},
        }
        print(json.dumps(out, ensure_ascii=False))
        return 2

    # Global counts
    cur_counts, cur_df, cur_total = build_counts(docs_cur, ngram_list, STOPWORDS, args.min_token_len)
    base_counts, base_df, base_total = build_counts(docs_base, ngram_list, STOPWORDS, args.min_token_len)

    global_silence = silence_rank(
        base_counts=base_counts,
        base_df=base_df,
        base_total=base_total,
        cur_counts=cur_counts,
        cur_df=cur_df,
        cur_total=cur_total,
        min_base_count=args.min_base_count,
        min_base_df=args.min_base_df,
        max_cur_count=args.max_cur_count,
        max_cur_df=args.max_cur_df,
        top=args.top,
    )

    # Group dropouts by volume
    dropouts = volume_dropouts(
        base_docs=docs_base,
        cur_docs=docs_cur,
        group_by=args.group_by,
        min_base_docs=args.dropout_min_base_docs,
        drop_ratio_threshold=args.dropout_ratio,
        top=50,
    )

    result = {
        "event": "silence_detection",
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "current_start": day_to_partition(cur_start),
            "current_end": day_to_partition(end_day),
            "baseline_start": day_to_partition(base_start),
            "baseline_end": day_to_partition(base_end),
            "group_by": args.group_by,
            "docs_current": len(docs_cur),
            "docs_baseline": len(docs_base),
            "tokens_current": cur_total,
            "tokens_baseline": base_total,
            "ngrams": ngram_list,
            "thresholds": {
                "min_docs": args.min_docs,
                "min_base_count": args.min_base_count,
                "min_base_df": args.min_base_df,
                "max_cur_count": args.max_cur_count,
                "max_cur_df": args.max_cur_df,
            },
        },
        "global_silence": global_silence,
        "volume_dropouts": dropouts,
    }

    # Per-group silence (optional)
    if args.per_group:
        # Organize docs by group for each window
        cur_by: Dict[str, List[Doc]] = defaultdict(list)
        base_by: Dict[str, List[Doc]] = defaultdict(list)
        for d in docs_cur:
            cur_by[d.group].append(d)
        for d in docs_base:
            base_by[d.group].append(d)

        per_group_silence: Dict[str, dict] = {}
        for g in sorted(set(cur_by.keys()) & set(base_by.keys())):
            cur_docs_g = cur_by[g]
            base_docs_g = base_by[g]
            if len(cur_docs_g) < args.min_docs_per_group or len(base_docs_g) < args.min_docs_per_group:
                continue

            g_cur_counts, g_cur_df, g_cur_total = build_counts(cur_docs_g, ngram_list, STOPWORDS, args.min_token_len)
            g_base_counts, g_base_df, g_base_total = build_counts(base_docs_g, ngram_list, STOPWORDS, args.min_token_len)

            g_silence = silence_rank(
                base_counts=g_base_counts,
                base_df=g_base_df,
                base_total=g_base_total,
                cur_counts=g_cur_counts,
                cur_df=g_cur_df,
                cur_total=g_cur_total,
                min_base_count=max(6, args.min_base_count // 2),
                min_base_df=max(3, args.min_base_df // 2),
                max_cur_count=args.max_cur_count,
                max_cur_df=args.max_cur_df,
                top=min(25, args.top),
            )

            if g_silence:
                per_group_silence[g] = {
                    "docs_current": len(cur_docs_g),
                    "docs_baseline": len(base_docs_g),
                    "silence": g_silence,
                }

        result["per_group_silence"] = per_group_silence

    # Markdown output
    if args.md_out:
        md_path = Path(args.md_out)
        write_markdown(md_path, result)
        result["markdown_written_to"] = str(md_path)

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
