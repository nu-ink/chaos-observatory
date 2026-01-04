#!/usr/bin/env python3
"""
Chaos-Observatory: Topic Convergence Analyzer (robust v1)

Purpose:
  Detect whether multiple groups (regions or sources) are converging on similar topics.

Approach (explainable, deterministic-ish):
  - Use TF-IDF over documents in a time window
  - Represent each group as the mean TF-IDF vector of its docs
  - Compute cosine similarity between group vectors
  - Identify "convergent terms" that show up in many groups' top TF-IDF terms

Input:
  Normalized JSONL docs in date partitions:
    data/normalized/YYYY-MM-DD/*.jsonl

Output:
  JSON (default) + optional Markdown report.

Notes:
  - This is not semantic embeddings; it's lexical topic alignment.
  - Works well for institutional/news language shifts and shared narratives.

Dependencies:
  - scikit-learn
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

# ---------- Stopwords (extend later) ----------
STOPWORDS = {
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


def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_group_corpus(docs: List[Doc]) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Returns:
      groups: group names aligned with texts
      texts: one doc per original doc (not aggregated) so group mean is meaningful
      group_index: map group->index for later use
    """
    groups: List[str] = []
    texts: List[str] = []
    for d in docs:
        groups.append(d.group)
        texts.append(clean_text(f"{d.title} {d.body_text}"))
    group_index: Dict[str, int] = {g: i for i, g in enumerate(sorted(set(groups)))}
    return groups, texts, group_index


def cosine_similarity_dense(a, b) -> float:
    # a, b are 1D arrays
    na = math.sqrt(float((a * a).sum()))
    nb = math.sqrt(float((b * b).sum()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float((a @ b) / (na * nb))


def make_markdown_report(
    out_path: Path,
    meta: dict,
    group_stats: List[dict],
    pair_rank: List[dict],
    convergent_terms: List[dict],
) -> None:
    lines: List[str] = []
    lines.append("# Chaos Observatory — Topic Convergence")
    lines.append("")
    lines.append(f"**Generated (UTC):** {meta['generated_at_utc']}")
    lines.append(f"**Window (UTC):** {meta['window_start']} → {meta['window_end']}")
    lines.append(f"**Group-by:** {meta['group_by']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Groups analyzed | {meta['groups_analyzed']} |")
    lines.append(f"| Docs analyzed | {meta['docs_analyzed']} |")
    lines.append(f"| Convergence score (avg cosine) | {meta['convergence_score']:.4f} |")
    lines.append("")
    lines.append("## Group Stats")
    lines.append("")
    lines.append("| Group | Docs |")
    lines.append("|---|---:|")
    for gs in group_stats:
        lines.append(f"| {gs['group']} | {gs['docs']} |")
    lines.append("")
    lines.append("## Most Similar Group Pairs")
    lines.append("")
    lines.append("| Rank | Group A | Group B | Cosine Similarity |")
    lines.append("|---:|---|---|---:|")
    for i, pr in enumerate(pair_rank[:25], start=1):
        lines.append(f"| {i} | {pr['a']} | {pr['b']} | {pr['cosine']:.4f} |")
    lines.append("")
    lines.append("## Top Convergent Terms/Phrases")
    lines.append("")
    lines.append("> Terms that appear in many groups' top terms (lexical convergence).")
    lines.append("")
    lines.append("| Term | Coverage (groups) | Avg TF-IDF |")
    lines.append("|---|---:|---:|")
    for t in convergent_terms[:30]:
        lines.append(f"| {t['term'].replace('|','\\\\|')} | {t['coverage']} | {t['avg_tfidf']:.6f} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized-dir", default="data/normalized", help="Base normalized partition dir")
    ap.add_argument("--end-date", default=None, help="UTC end date YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--window-days", type=int, default=7, help="Window length in days")
    ap.add_argument("--group-by", choices=["region", "source"], default="region", help="Group docs by region or source")
    ap.add_argument("--min-docs-per-group", type=int, default=10, help="Minimum docs per group to include")
    ap.add_argument("--max-groups", type=int, default=25, help="If too many groups, keep top-N by doc count")
    ap.add_argument("--top-terms-per-group", type=int, default=25, help="Top TF-IDF terms used for convergence term coverage")
    ap.add_argument("--min-term-coverage", type=int, default=3, help="Minimum number of groups a term must appear in")
    ap.add_argument("--ngram-max", type=int, default=2, help="Use 1..ngram-max (1=unigram, 2=unigram+bigrams)")
    ap.add_argument("--min-df", type=int, default=2, help="TF-IDF min_df")
    ap.add_argument("--max-df", type=float, default=0.90, help="TF-IDF max_df (float fraction)")
    ap.add_argument("--md-out", default=None, help="Optional markdown output path")
    args = ap.parse_args()

    if args.window_days <= 0:
        raise SystemExit("ERROR: --window-days must be > 0")
    if args.ngram_max < 1 or args.ngram_max > 4:
        raise SystemExit("ERROR: --ngram-max must be between 1 and 4")

    # Import sklearn lazily (clear error if missing)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as ex:
        print(json.dumps({"event": "topic_convergence", "error": "missing_dependency", "detail": repr(ex)}))
        print("Install: pip install scikit-learn", flush=True)
        return 2

    base_dir = Path(args.normalized_dir)
    end_day = parse_ymd(args.end_date) if args.end_date else utc_midnight_today()
    start_day = end_day - timedelta(days=args.window_days - 1)

    docs = load_docs(base_dir, start_day, end_day, group_by=args.group_by)

    # Group docs and filter by size
    docs_by_group: Dict[str, List[Doc]] = defaultdict(list)
    for d in docs:
        docs_by_group[d.group].append(d)

    group_counts = sorted(((g, len(v)) for g, v in docs_by_group.items()), key=lambda x: x[1], reverse=True)

    # Keep only groups that meet threshold, then top-N by doc count
    kept = [(g, n) for g, n in group_counts if n >= args.min_docs_per_group]
    kept = kept[: args.max_groups]
    kept_groups = {g for g, _ in kept}

    filtered_docs = [d for d in docs if d.group in kept_groups]

    if len(kept_groups) < 2:
        out = {
            "event": "topic_convergence",
            "error": "insufficient_groups",
            "groups_found": len(docs_by_group),
            "groups_kept": len(kept_groups),
            "min_docs_per_group": args.min_docs_per_group,
            "window": {"start": day_to_partition(start_day), "end": day_to_partition(end_day)},
        }
        print(json.dumps(out, ensure_ascii=False))
        return 2

    groups_list, texts, _ = build_group_corpus(filtered_docs)

    # Vectorize documents
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=STOPWORDS,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
    )
    X = vectorizer.fit_transform(texts)  # shape: (docs, terms)
    terms = vectorizer.get_feature_names_out()

    # Compute group mean vectors
    group_to_rows: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(groups_list):
        group_to_rows[g].append(i)

    group_vectors: Dict[str, "any"] = {}
    group_doc_counts: Dict[str, int] = {}
    for g, rows in group_to_rows.items():
        group_doc_counts[g] = len(rows)
        # Mean TF-IDF vector for group
        gv = X[rows].mean(axis=0)  # 1 x terms (matrix)
        group_vectors[g] = gv

    # Convert to dense arrays for cosine calc safely (groups are small)
    import numpy as np

    dense_vecs: Dict[str, np.ndarray] = {}
    for g, gv in group_vectors.items():
        dense_vecs[g] = np.asarray(gv).ravel()

    groups_sorted = sorted(dense_vecs.keys(), key=lambda g: group_doc_counts[g], reverse=True)

    # Pairwise cosine similarities
    pair_rank: List[dict] = []
    sims: List[float] = []
    for i in range(len(groups_sorted)):
        for j in range(i + 1, len(groups_sorted)):
            a = groups_sorted[i]
            b = groups_sorted[j]
            cos = cosine_similarity_dense(dense_vecs[a], dense_vecs[b])
            sims.append(cos)
            pair_rank.append({"a": a, "b": b, "cosine": cos})

    pair_rank.sort(key=lambda r: r["cosine"], reverse=True)
    convergence_score = float(sum(sims) / len(sims)) if sims else 0.0

    # Convergent terms:
    # For each group, take its top-K terms by TF-IDF weight, then count term coverage across groups.
    term_coverage = Counter()
    term_strength = Counter()

    for g in groups_sorted:
        v = dense_vecs[g]
        # top K indices
        K = min(args.top_terms_per_group, len(v))
        if K <= 0:
            continue
        top_idx = np.argpartition(v, -K)[-K:]
        # sort descending
        top_idx = top_idx[np.argsort(v[top_idx])[::-1]]

        seen_terms = set()
        for idx in top_idx:
            t = terms[idx]
            if t in seen_terms:
                continue
            seen_terms.add(t)
            term_coverage[t] += 1
            term_strength[t] += float(v[idx])

    convergent_terms: List[dict] = []
    for t, cov in term_coverage.most_common():
        if cov < args.min_term_coverage:
            break
        avg_strength = float(term_strength[t] / cov) if cov else 0.0
        convergent_terms.append({"term": t, "coverage": int(cov), "avg_tfidf": avg_strength})

    # Group stats
    group_stats = [{"group": g, "docs": group_doc_counts[g]} for g in groups_sorted]

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_start": day_to_partition(start_day),
        "window_end": day_to_partition(end_day),
        "group_by": args.group_by,
        "groups_analyzed": len(groups_sorted),
        "docs_analyzed": len(filtered_docs),
        "convergence_score": convergence_score,
        "settings": {
            "min_docs_per_group": args.min_docs_per_group,
            "max_groups": args.max_groups,
            "top_terms_per_group": args.top_terms_per_group,
            "min_term_coverage": args.min_term_coverage,
            "ngram_max": args.ngram_max,
            "min_df": args.min_df,
            "max_df": args.max_df,
        },
    }

    result = {
        "event": "topic_convergence",
        "meta": meta,
        "group_stats": group_stats,
        "pair_rank": pair_rank[: min(200, len(pair_rank))],  # cap
        "convergent_terms": convergent_terms[:200],
    }

    if args.md_out:
        md_path = Path(args.md_out)
        make_markdown_report(md_path, meta, group_stats, pair_rank, convergent_terms)
        result["markdown_written_to"] = str(md_path)

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
