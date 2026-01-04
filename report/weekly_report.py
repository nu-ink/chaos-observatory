#!/usr/bin/env python3
"""
Chaos-Observatory Weekly Report (v1)

Reads normalized JSONL partitions and generates a Markdown report.

Expected input layout:
  data/normalized/YYYY-MM-DD/*.jsonl

Outputs:
  reports/YYYY-MM-DD/weekly_report.md

Core report sections:
- Volume summary (total + per source)
- Top terms (current window)
- Drift terms (up vs baseline)
- Sentiment compression index (simple lexicon-based)
- Silence indicators (terms present in baseline but missing now)

Notes:
- This is deterministic and explainable by design.
- No embeddings or heavy NLP required.
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
from typing import Dict, Iterable, List, Optional, Tuple

# ----------------------------
# Minimal stopwords (v1)
# ----------------------------
STOPWORDS = {
    "a","an","and","are","as","at","be","been","but","by","can","could","did","do","does","for","from",
    "had","has","have","he","her","his","how","i","if","in","into","is","it","its","just","may","might",
    "more","most","must","not","of","on","or","our","out","over","s","said","she","should","so","some",
    "than","that","the","their","them","then","there","these","they","this","to","too","under","up",
    "was","we","were","what","when","where","which","who","will","with","would","you","your",
}

# ----------------------------
# Simple sentiment lexicon (v1)
# Explainable, not “truth”.
# ----------------------------
POS_WORDS = {
    "agree","aid","calm","ceasefire","cooperate","cooperation","deal","decline","decrease","deescalate",
    "growth","improve","improvement","progress","recover","recovery","relief","rescue","stabilize",
    "support","truce",
}
NEG_WORDS = {
    "attack","bomb","crisis","dead","death","decline","disaster","emergency","escalate","explosion",
    "famine","fear","fighting","flood","hostage","inflation","injury","killed","missile","outbreak",
    "pandemic","protest","raid","risk","sanction","shortage","strike","threat","tension","war",
}


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{1,}")  # tokens length >=2, allow hyphen


@dataclass
class Doc:
    published_at_utc: Optional[str]
    source_id: str
    source_label: str
    title: str
    body_text: str


def utc_today() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def parse_ymd(s: str) -> datetime:
    dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def daterange_days(end_inclusive: datetime, days: int) -> List[datetime]:
    # end_inclusive is midnight UTC for that day
    out = []
    for i in range(days):
        out.append(end_inclusive - timedelta(days=i))
    return list(reversed(out))


def day_to_partition(day: datetime) -> str:
    return day.strftime("%Y-%m-%d")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_docs_from_partitions(base_dir: Path, start_day: datetime, end_day: datetime) -> List[Doc]:
    docs: List[Doc] = []
    cur = start_day
    while cur <= end_day:
        p = base_dir / day_to_partition(cur)
        if p.exists() and p.is_dir():
            for f in p.glob("*.jsonl"):
                for row in read_jsonl(f):
                    docs.append(
                        Doc(
                            published_at_utc=row.get("published_at_utc"),
                            source_id=row.get("source_id") or "unknown",
                            source_label=row.get("source_label") or row.get("source_id") or "unknown",
                            title=row.get("title") or "",
                            body_text=row.get("body_text") or "",
                        )
                    )
        cur += timedelta(days=1)
    return docs


def tokenize(text: str) -> List[str]:
    text = text.lower()
    toks = TOKEN_RE.findall(text)
    toks = [t for t in toks if t not in STOPWORDS and len(t) >= 3]
    return toks


def term_counts(docs: List[Doc]) -> Counter:
    c = Counter()
    for d in docs:
        toks = tokenize(d.title + " " + d.body_text)
        c.update(toks)
    return c


def volume_by_source(docs: List[Doc]) -> Dict[str, int]:
    c: Dict[str, int] = defaultdict(int)
    for d in docs:
        c[d.source_label] += 1
    return dict(sorted(c.items(), key=lambda kv: kv[1], reverse=True))


def sentiment_index(docs: List[Doc]) -> Dict[str, float]:
    """
    Simple compression index:
      - pos_rate = pos_hits / tokens
      - neg_rate = neg_hits / tokens
      - certainty_rate = certainty_hits / tokens
      - compression = (neg_rate + certainty_rate) - pos_rate

    Interpret carefully: it's a consistent indicator, not “truth”.
    """
    certainty_words = {"must","will","cannot","never","always","inevitable","urgent","critical"}

    pos_hits = 0
    neg_hits = 0
    cert_hits = 0
    tok_total = 0

    for d in docs:
        toks = tokenize(d.title + " " + d.body_text)
        tok_total += len(toks)
        pos_hits += sum(1 for t in toks if t in POS_WORDS)
        neg_hits += sum(1 for t in toks if t in NEG_WORDS)
        cert_hits += sum(1 for t in toks if t in certainty_words)

    if tok_total == 0:
        return {"pos_rate": 0.0, "neg_rate": 0.0, "certainty_rate": 0.0, "compression": 0.0, "tokens": 0}

    pos_rate = pos_hits / tok_total
    neg_rate = neg_hits / tok_total
    cert_rate = cert_hits / tok_total
    compression = (neg_rate + cert_rate) - pos_rate

    return {
        "pos_rate": pos_rate,
        "neg_rate": neg_rate,
        "certainty_rate": cert_rate,
        "compression": compression,
        "tokens": tok_total,
    }


def drift_terms(current: Counter, baseline: Counter, top_n: int = 25, min_current: int = 8) -> List[Tuple[str, float, int, int]]:
    """
    Drift score: log1p(current) - log1p(baseline)
    Returns list of (term, score, current_count, baseline_count)
    """
    rows = []
    all_terms = set(current.keys()) | set(baseline.keys())
    for t in all_terms:
        c = current.get(t, 0)
        b = baseline.get(t, 0)
        if c < min_current:
            continue
        score = math.log1p(c) - math.log1p(b)
        rows.append((t, score, c, b))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def silence_terms(current: Counter, baseline: Counter, top_n: int = 15, min_baseline: int = 12) -> List[Tuple[str, int]]:
    """
    Terms that were frequent in baseline but effectively absent now.
    """
    rows = []
    for t, b in baseline.most_common():
        if b < min_baseline:
            break
        if current.get(t, 0) == 0:
            rows.append((t, b))
        if len(rows) >= top_n:
            break
    return rows


def md_table_kv(rows: List[Tuple[str, str]], header_left: str = "Metric", header_right: str = "Value") -> str:
    out = [f"| {header_left} | {header_right} |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out)


def md_table_counts(rows: List[Tuple[str, int]], header_left: str = "Item", header_right: str = "Count") -> str:
    out = [f"| {header_left} | {header_right} |", "|---|---:|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out)


def fmt_pct(x: float) -> str:
    return f"{x*100:.3f}%"


def build_report(
    end_day: datetime,
    window_days: int,
    baseline_days: int,
    docs_current: List[Doc],
    docs_baseline: List[Doc],
) -> str:
    start_day = end_day - timedelta(days=window_days - 1)
    baseline_start = start_day - timedelta(days=baseline_days)
    baseline_end = start_day - timedelta(days=1)

    cur_vol = len(docs_current)
    base_vol = len(docs_baseline)

    cur_by_src = volume_by_source(docs_current)
    cur_terms = term_counts(docs_current)

    base_terms = term_counts(docs_baseline) if docs_baseline else Counter()

    top_terms = cur_terms.most_common(30)
    drift = drift_terms(cur_terms, base_terms, top_n=25) if docs_baseline else []
    silence = silence_terms(cur_terms, base_terms, top_n=15) if docs_baseline else []

    sent_cur = sentiment_index(docs_current)
    sent_base = sentiment_index(docs_baseline) if docs_baseline else {"compression": 0.0, "tokens": 0}

    # Markdown
    lines: List[str] = []

    lines.append("# Chaos Observatory — Weekly Drift Report")
    lines.append("")
    lines.append(f"**Report window (UTC):** {start_day.strftime('%Y-%m-%d')} → {end_day.strftime('%Y-%m-%d')}  ")
    if docs_baseline:
        lines.append(f"**Baseline window (UTC):** {baseline_start.strftime('%Y-%m-%d')} → {baseline_end.strftime('%Y-%m-%d')}")
    else:
        lines.append("**Baseline window (UTC):** *(disabled)*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Volume summary
    lines.append("## 1) Volume Summary")
    lines.append("")
    lines.append(
        md_table_kv(
            [
                ("Documents (current)", str(cur_vol)),
                ("Documents (baseline)", str(base_vol) if docs_baseline else "n/a"),
                ("Unique terms (current)", str(len(cur_terms))),
                ("Generated at (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")),
            ]
        )
    )
    lines.append("")
    lines.append("### Sources (Current Window)")
    lines.append("")
    lines.append(md_table_counts(list(cur_by_src.items())[:25], header_left="Source", header_right="Docs"))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Top terms
    lines.append("## 2) Top Terms (Current Window)")
    lines.append("")
    lines.append(
        "> These are the most common non-stopword tokens across titles + bodies (RSS summaries). "
        "This is a high-level signal, not a conclusion."
    )
    lines.append("")
    lines.append(md_table_counts(top_terms, header_left="Term", header_right="Count"))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Drift
    lines.append("## 3) Term Drift (Rise vs Baseline)")
    lines.append("")
    if not docs_baseline:
        lines.append("_Baseline disabled — drift section unavailable._")
    elif not drift:
        lines.append("_No drift terms met thresholds._")
    else:
        lines.append(
            "> Drift score = log1p(current_count) − log1p(baseline_count). "
            "Higher means stronger rise in the current window."
        )
        lines.append("")
        lines.append("| Term | Drift Score | Current | Baseline |")
        lines.append("|---|---:|---:|---:|")
        for term, score, c, b in drift:
            lines.append(f"| {term} | {score:.4f} | {c} | {b} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Sentiment compression
    lines.append("## 4) Sentiment Compression Index (Explainable)")
    lines.append("")
    lines.append(
        "> A simple, deterministic index based on small lexicons:\n"
        "> - **neg_rate**: share of negative/instability terms\n"
        "> - **certainty_rate**: share of certainty/urgency terms\n"
        "> - **pos_rate**: share of positive/de-escalation terms\n"
        ">\n"
        "> **compression = (neg_rate + certainty_rate) − pos_rate**\n"
        "> Higher values suggest more compressed/urgent language overall."
    )
    lines.append("")
    cur_rows = [
        ("Tokens analyzed (current)", str(sent_cur["tokens"])),
        ("pos_rate (current)", fmt_pct(sent_cur["pos_rate"])),
        ("neg_rate (current)", fmt_pct(sent_cur["neg_rate"])),
        ("certainty_rate (current)", fmt_pct(sent_cur["certainty_rate"])),
        ("compression (current)", f"{sent_cur['compression']:.6f}"),
    ]
    if docs_baseline:
        cur_rows += [
            ("Tokens analyzed (baseline)", str(sent_base["tokens"])),
            ("compression (baseline)", f"{sent_base['compression']:.6f}"),
            ("compression delta", f"{(sent_cur['compression'] - sent_base['compression']):.6f}"),
        ]
    lines.append(md_table_kv(cur_rows))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Silence
    lines.append("## 5) Silence Indicators (Present in Baseline, Missing Now)")
    lines.append("")
    if not docs_baseline:
        lines.append("_Baseline disabled — silence section unavailable._")
    elif not silence:
        lines.append("_No strong baseline terms fully disappeared in the current window._")
    else:
        lines.append(
            "> These terms appeared frequently in the baseline window but did not appear at all in the current window. "
            "This can indicate de-escalation, narrative replacement, reporting gaps, or shifting priorities."
        )
        lines.append("")
        lines.append("| Term | Baseline Count |")
        lines.append("|---|---:|")
        for term, b in silence:
            lines.append(f"| {term} | {b} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Notes
    lines.append("## Notes")
    lines.append("")
    lines.append("- This report is **observational** and based on text frequency signals.")
    lines.append("- RSS summaries vary by publisher; treat cross-source comparisons carefully.")
    lines.append("- Next upgrades: entity extraction, phrase/bigram drift, semantic clustering, and source weighting.")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized-dir", default="data/normalized", help="Base directory for normalized partitions")
    ap.add_argument("--outdir", default="reports", help="Base output directory for reports")
    ap.add_argument("--end-date", default=None, help="End date (UTC) YYYY-MM-DD. Default: today (UTC)")
    ap.add_argument("--window-days", type=int, default=7, help="Report window length (days)")
    ap.add_argument("--baseline-days", type=int, default=7, help="Baseline window length (days). Set 0 to disable.")
    args = ap.parse_args()

    if args.window_days <= 0:
        raise SystemExit("ERROR: --window-days must be > 0")
    if args.baseline_days < 0:
        raise SystemExit("ERROR: --baseline-days must be >= 0")

    end_day = parse_ymd(args.end_date) if args.end_date else utc_today()
    start_day = end_day - timedelta(days=args.window_days - 1)

    base_dir = Path(args.normalized_dir)
    outdir = Path(args.outdir) / day_to_partition(end_day)
    outdir.mkdir(parents=True, exist_ok=True)

    docs_current = load_docs_from_partitions(base_dir, start_day, end_day)

    docs_baseline: List[Doc] = []
    if args.baseline_days > 0:
        baseline_end = start_day - timedelta(days=1)
        baseline_start = start_day - timedelta(days=args.baseline_days)
        if baseline_end >= baseline_start:
            docs_baseline = load_docs_from_partitions(base_dir, baseline_start, baseline_end)

    report_md = build_report(
        end_day=end_day,
        window_days=args.window_days,
        baseline_days=args.baseline_days,
        docs_current=docs_current,
        docs_baseline=docs_baseline,
    )

    out_path = outdir / "weekly_report.md"
    out_path.write_text(report_md, encoding="utf-8")

    print(json.dumps({"event": "weekly_report_written", "out": str(out_path), "docs_current": len(docs_current), "docs_baseline": len(docs_baseline)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
