#!/usr/bin/env python3
"""
Chaos-Observatory: Frequency Drift Analyzer (robust v1)

Goal:
  Identify terms/phrases whose relative frequency increases in a "current" window
  compared to a "baseline" window.

Input:
  Normalized JSONL docs in date partitions:
    data/normalized/YYYY-MM-DD/*.jsonl

Output:
  JSON (default) + optional Markdown report.

Scoring:
  - log_odds_dirichlet: Monroe et al.-style log-odds with informative Dirichlet prior.
    This is robust to rare counts and different corpus sizes.
  - log_ratio: log((c_cur + s)/(N_cur + sV)) - log((c_base + s)/(N_base + sV))
    (simple, useful sanity check)

Notes:
  - Deterministic & explainable by design.
  - Uses title+body_text tokens.
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
# Stopwords (v1; extend later)
# ----------------------------
STOPWORDS: Set[str] = {
    "a","an","and","are","as","at","be","been","but","by","can","could","did","do","does",
    "for","from","had","has","have","he","her","his","how","i","if","in","into","is","it",
    "its","just","may","might","more","most","must","not","of","on","or","our","out","over",
    "s","said","she","should","so","some","than","that","the","their","them","then","there",
    "these","they","this","to","too","under","up","was","we","were","what","when","where",
    "which","who","will","with","would","you","your",
}

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{1,}")  # allow digits/hyphen after first alpha


@dataclass(frozen=True)
class Doc:
    source_label: str
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


def load_docs(base_dir: Path, start_day: datetime, end_day: datetime) -> List[Doc]:
    docs: List[Doc] = []
    cur = start_day
    while cur <= end_day:
        part = base_dir / day_to_partition(cur)
        if part.exists() and part.is_dir():
            for f in part.glob("*.jsonl"):
                for row in read_jsonl(f):
                    docs.append(
                        Doc(
                            source_label=row.get("source_label") or row.get("source_id") or "unknown",
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
      - term_counts: total token counts (across docs)
      - doc_freq: number of docs containing term at least once
      - total_terms: total token instances across docs (sum counts)
    """
    term_counts = Counter()
    doc_freq = Counter()

    total_terms = 0
    for d in docs:
        toks = tokenize(d.title + " " + d.body_text, stopwords=stopwords, min_len=min_token_len)
        doc_terms_in_doc: Set[str] = set()

        for n in ngram_list:
            grams = ngrams(toks, n)
            term_counts.update(grams)
            total_terms += len(grams)
            doc_terms_in_doc.update(grams)

        doc_freq.update(doc_terms_in_doc)

    return term_counts, doc_freq, total_terms


def log_odds_dirichlet(
    cur_counts: Counter,
    base_counts: Counter,
    prior_counts: Counter,
    total_cur: int,
    total_base: int,
    total_prior: int,
) -> Dict[str, float]:
    """
    Returns z-scores for each term using log-odds with informative Dirichlet prior.

    For term w:
      delta = log((c1 + a) / (N1 - c1 + A - a)) - log((c2 + a) / (N2 - c2 + A - a))
      var = 1/(c1+a) + 1/(c2+a) + 1/(N1 - c1 + A - a) + 1/(N2 - c2 + A - a)
      z = delta / sqrt(var)

    Where:
      c1 = current count, c2 = baseline count
      N1, N2 are total token instances (within chosen ngram set)
      a is prior count for w, A is total prior count
    """
    z: Dict[str, float] = {}
    A = float(total_prior)

    # union of terms (but score only those present anywhere)
    terms = set(cur_counts.keys()) | set(base_counts.keys()) | set(prior_counts.keys())

    for w in terms:
        c1 = float(cur_counts.get(w, 0))
        c2 = float(base_counts.get(w, 0))
        a = float(prior_counts.get(w, 0))

        # To avoid pathological cases where A-a == 0, enforce tiny epsilon
        eps = 1e-12
        denom1 = (total_cur - c1) + (A - a)
        denom2 = (total_base - c2) + (A - a)

        # Skip if both corpora have no tokens (shouldn’t happen)
        if total_cur <= 0 or total_base <= 0:
            continue

        denom1 = max(denom1, eps)
        denom2 = max(denom2, eps)

        num1 = c1 + a
        num2 = c2 + a
        num1 = max(num1, eps)
        num2 = max(num2, eps)

        delta = math.log(num1 / denom1) - math.log(num2 / denom2)

        var = (1.0 / num1) + (1.0 / num2) + (1.0 / denom1) + (1.0 / denom2)
        z[w] = delta / math.sqrt(var)

    return z


def log_ratio_score(
    c_cur: int,
    c_base: int,
    total_cur: int,
    total_base: int,
    smooth: float,
    vocab_size: int,
) -> float:
    """
    Simple smoothed log probability ratio.
    """
    # add smooth mass distributed across vocab to stabilize
    p_cur = (c_cur + smooth) / (total_cur + smooth * vocab_size)
    p_base = (c_base + smooth) / (total_base + smooth * vocab_size)
    return math.log(p_cur) - math.log(p_base)


def md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def write_markdown(
    out_path: Path,
    meta: dict,
    top_risers: List[dict],
    silence: List[dict],
) -> None:
    lines: List[str] = []
    lines.append("# Chaos Observatory — Frequency Drift")
    lines.append("")
    lines.append(f"**Generated (UTC):** {meta['generated_at_utc']}")
    lines.append(f"**Current window (UTC):** {meta['current_start']} → {meta['current_end']}  ")
    lines.append(f"**Baseline window (UTC):** {meta['baseline_start']} → {meta['baseline_end']}")
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
    lines.append("## Top Risers (by log-odds z-score)")
    lines.append("")
    lines.append("| Term | z | cur_count | base_count | cur_df | base_df | log_ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in top_risers:
        lines.append(
            f"| {md_escape(r['term'])} | {r['z']:.3f} | {r['cur_count']} | {r['base_count']} | "
            f"{r['cur_df']} | {r['base_df']} | {r['log_ratio']:.4f} |"
        )
    lines.append("")
    lines.append("## Silence Indicators")
    lines.append("")
    lines.append("> Terms with meaningful baseline presence that are absent in the current window.")
    lines.append("")
    lines.append("| Term | base_count | base_df |")
    lines.append("|---|---:|---:|")
    for s in silence:
        lines.append(f"| {md_escape(s['term'])} | {s['base_count']} | {s['base_df']} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized-dir", default="data/normalized", help="Base normalized partition dir")
    ap.add_argument("--end-date", default=None, help="UTC end date YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--window-days", type=int, default=7, help="Current window length in days")
    ap.add_argument("--baseline-days", type=int, default=7, help="Baseline window length in days")
    ap.add_argument("--ngrams", default="1,2", help="Comma list: 1,2 or 1 or 2 (max 4)")
    ap.add_argument("--min-token-len", type=int, default=3, help="Minimum token length for unigrams")
    ap.add_argument("--min-cur", type=int, default=8, help="Minimum current count for a term to be ranked")
    ap.add_argument("--min-df", type=int, default=3, help="Minimum current document frequency for ranking")
    ap.add_argument("--min-docs", type=int, default=10, help="Require at least this many docs per window")
    ap.add_argument("--top", type=int, default=40, help="How many risers to return")
    ap.add_argument("--silence-top", type=int, default=20, help="How many silence indicators to return")
    ap.add_argument("--silence-min-base", type=int, default=12, help="Minimum baseline count to consider for silence")
    ap.add_argument("--smooth", type=float, default=0.5, help="Smoothing mass used in log_ratio")
    ap.add_argument("--per-source", action="store_true", help="Also compute drift per source_label")
    ap.add_argument("--md-out", default=None, help="Optional markdown output path")
    args = ap.parse_args()

    if args.window_days <= 0 or args.baseline_days <= 0:
        raise SystemExit("ERROR: window-days and baseline-days must be > 0")
    if args.min_doc <= 0 if False else False:
        pass

    base_dir = Path(args.normalized_dir)
    end_day = parse_ymd(args.end_date) if args.end_date else utc_midnight_today()
    cur_start = end_day - timedelta(days=args.window_days - 1)

    base_end = cur_start - timedelta(days=1)
    base_start = base_end - timedelta(days=args.baseline_days - 1)

    ngram_list = parse_ngrams_arg(args.ngrams)

    docs_cur = load_docs(base_dir, cur_start, end_day)
    docs_base = load_docs(base_dir, base_start, base_end)

    if len(docs_cur) < args.min_docs or len(docs_base) < args.min_docs:
        out = {
            "event": "frequency_drift",
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
    cur_counts, cur_df, total_cur = build_counts(docs_cur, ngram_list, STOPWORDS, args.min_token_len)
    base_counts, base_df, total_base = build_counts(docs_base, ngram_list, STOPWORDS, args.min_token_len)

    # Prior: pooled counts as informative prior (standard, robust)
    prior_counts = cur_counts + base_counts
    total_prior = total_cur + total_base

    zscores = log_odds_dirichlet(
        cur_counts=cur_counts,
        base_counts=base_counts,
        prior_counts=prior_counts,
        total_cur=total_cur,
        total_base=total_base,
        total_prior=total_prior,
    )

    vocab = set(cur_counts.keys()) | set(base_counts.keys())
    vocab_size = max(len(vocab), 1)

    candidates = []
    for term, z in zscores.items():
        ccur = int(cur_counts.get(term, 0))
        cbase = int(base_counts.get(term, 0))
        dfcur = int(cur_df.get(term, 0))
        dfbase = int(base_df.get(term, 0))

        if ccur < args.min_cur:
            continue
        if dfcur < args.min_df:
            continue

        lr = log_ratio_score(
            c_cur=ccur,
            c_base=cbase,
            total_cur=total_cur,
            total_base=total_base,
            smooth=args.smooth,
            vocab_size=vocab_size,
        )

        candidates.append(
            {
                "term": term,
                "z": float(z),
                "cur_count": ccur,
                "base_count": cbase,
                "cur_df": dfcur,
                "base_df": dfbase,
                "log_ratio": float(lr),
            }
        )

    candidates.sort(key=lambda r: r["z"], reverse=True)
    top_risers = candidates[: args.top]

    # Silence terms: strong in baseline, absent now
    silence = []
    for term, bcnt in base_counts.most_common():
        if bcnt < args.silence_min_base:
            break
        if cur_counts.get(term, 0) == 0:
            silence.append(
                {
                    "term": term,
                    "base_count": int(bcnt),
                    "base_df": int(base_df.get(term, 0)),
                }
            )
        if len(silence) >= args.silence_top:
            break

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "current_start": day_to_partition(cur_start),
        "current_end": day_to_partition(end_day),
        "baseline_start": day_to_partition(base_start),
        "baseline_end": day_to_partition(base_end),
        "docs_current": len(docs_cur),
        "docs_baseline": len(docs_base),
        "tokens_current": total_cur,
        "tokens_baseline": total_base,
        "ngrams": ngram_list,
        "thresholds": {
            "min_cur": args.min_cur,
            "min_df": args.min_df,
            "min_docs": args.min_docs,
        },
        "scoring": {
            "method": "log_odds_dirichlet_z",
            "log_ratio_smooth": args.smooth,
        },
    }

    result = {
        "event": "frequency_drift",
        "meta": meta,
        "top_risers": top_risers,
        "silence": silence,
    }

    # Optional per-source drift
    if args.per_source:
        by_source_cur: Dict[str, List[Doc]] = defaultdict(list)
        by_source_base: Dict[str, List[Doc]] = defaultdict(list)

        for d in docs_cur:
            by_source_cur[d.source_label].append(d)
        for d in docs_base:
            by_source_base[d.source_label].append(d)

        per_source = {}
        for src, cur_docs in by_source_cur.items():
            base_docs = by_source_base.get(src, [])
            if len(cur_docs) < args.min_docs or len(base_docs) < args.min_docs:
                continue

            s_cur_counts, s_cur_df, s_total_cur = build_counts(cur_docs, ngram_list, STOPWORDS, args.min_token_len)
            s_base_counts, s_base_df, s_total_base = build_counts(base_docs, ngram_list, STOPWORDS, args.min_token_len)

            s_prior = s_cur_counts + s_base_counts
            s_total_prior = s_total_cur + s_total_base

            s_z = log_odds_dirichlet(s_cur_counts, s_base_counts, s_prior, s_total_cur, s_total_base, s_total_prior)
            s_vocab = set(s_cur_counts.keys()) | set(s_base_counts.keys())
            s_vocab_size = max(len(s_vocab), 1)

            s_candidates = []
            for term, z in s_z.items():
                ccur = int(s_cur_counts.get(term, 0))
                cbase = int(s_base_counts.get(term, 0))
                dfcur = int(s_cur_df.get(term, 0))
                if ccur < args.min_cur or dfcur < args.min_df:
                    continue
                lr = log_ratio_score(ccur, cbase, s_total_cur, s_total_base, args.smooth, s_vocab_size)
                s_candidates.append(
                    {
                        "term": term,
                        "z": float(z),
                        "cur_count": ccur,
                        "base_count": cbase,
                        "cur_df": int(dfcur),
                        "base_df": int(s_base_df.get(term, 0)),
                        "log_ratio": float(lr),
                    }
                )
            s_candidates.sort(key=lambda r: r["z"], reverse=True)
            per_source[src] = {
                "docs_current": len(cur_docs),
                "docs_baseline": len(base_docs),
                "top_risers": s_candidates[: min(args.top, 25)],
            }

        result["per_source"] = per_source

    # Markdown output if requested
    if args.md_out:
        md_path = Path(args.md_out)
        write_markdown(md_path, meta=meta, top_risers=top_risers, silence=silence)
        result["markdown_written_to"] = str(md_path)

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
