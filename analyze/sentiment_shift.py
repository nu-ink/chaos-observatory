#!/usr/bin/env python3
"""
Chaos-Observatory: Sentiment Shift Analyzer (robust v1)

Purpose:
  Measure shifts in tone over time windows in a deterministic, explainable way.
  This is NOT a true sentiment model; it's a consistent lexical indicator.

Input:
  Normalized JSONL partitions:
    data/normalized/YYYY-MM-DD/*.jsonl

Windows:
  - Current window: last N days
  - Baseline window: previous M days immediately before current window

Signals (lexicon-based):
  - pos_rate: positive / de-escalation / recovery terms
  - neg_rate: negative / conflict / disaster terms
  - urgency_rate: urgent / critical / emergency terms
  - certainty_rate: must / will / cannot / inevitable style terms
  - compression: (neg_rate + urgency_rate + certainty_rate) - pos_rate
  - volatility: Jensen–Shannon distance between word distributions (optional, lexical)

Outputs:
  JSON to stdout, optional Markdown.

Design:
  Explainable, stable, tunable thresholds.
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
    "its","just","may","might","more","most","not","of","on","or","our","out","over",
    "s","said","she","should","so","some","than","that","the","their","them","then","there",
    "these","they","this","to","too","under","up","was","we","were","what","when","where",
    "which","who","with","would","you","your",
}

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]{1,}")


# ----------------------------
# Lexicons (small, explainable)
# Tune over time.
# ----------------------------
POS_WORDS = {
    "agree","aid","calm","ceasefire","cooperate","cooperation","deal","decrease","deescalate",
    "growth","improve","improvement","progress","recover","recovery","relief","rescue",
    "stabilize","support","truce","peace","reopen","rebuild",
}

NEG_WORDS = {
    "attack","bomb","crisis","dead","death","disaster","emergency","escalate","explosion",
    "famine","fear","fighting","flood","hostage","inflation","injury","killed","missile",
    "outbreak","pandemic","protest","raid","risk","sanction","shortage","strike","threat",
    "tension","war","violence","collapse",
}

URGENCY_WORDS = {
    "urgent","critically","critical","emergency","immediately","warning","alert","evacuate",
    "evacuation","deadline","severe","rapid","surge","spike","escalation",
}

CERTAINTY_WORDS = {
    "must","will","cannot","never","always","inevitable","certain","clearly","undeniable",
    "required","demand","order","ban",
}

# (Optional) "uncertainty/hedging" can also be tracked; sometimes rising hedging is signal.
HEDGE_WORDS = {
    "may","might","could","possible","possibly","reportedly","alleged","allegedly","suggest",
    "suggests","unclear","unknown","likely","unlikely",
}


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
                        group = "all"

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


def rates_from_tokens(tokens: Sequence[str]) -> Dict[str, float]:
    n = len(tokens)
    if n == 0:
        return {
            "tokens": 0,
            "pos_rate": 0.0,
            "neg_rate": 0.0,
            "urgency_rate": 0.0,
            "certainty_rate": 0.0,
            "hedge_rate": 0.0,
            "compression": 0.0,
        }

    pos = sum(1 for t in tokens if t in POS_WORDS)
    neg = sum(1 for t in tokens if t in NEG_WORDS)
    urg = sum(1 for t in tokens if t in URGENCY_WORDS)
    cert = sum(1 for t in tokens if t in CERTAINTY_WORDS)
    hedge = sum(1 for t in tokens if t in HEDGE_WORDS)

    pos_rate = pos / n
    neg_rate = neg / n
    urg_rate = urg / n
    cert_rate = cert / n
    hedge_rate = hedge / n

    compression = (neg_rate + urg_rate + cert_rate) - pos_rate

    return {
        "tokens": n,
        "pos_rate": pos_rate,
        "neg_rate": neg_rate,
        "urgency_rate": urg_rate,
        "certainty_rate": cert_rate,
        "hedge_rate": hedge_rate,
        "compression": compression,
    }


def analyze_docs(docs: List[Doc], min_token_len: int) -> Tuple[Dict[str, float], Counter]:
    """
    Returns:
      - rates (lexical indicators)
      - token_counts (for optional distribution shift)
    """
    all_tokens: List[str] = []
    token_counts = Counter()

    for d in docs:
        toks = tokenize(d.title + " " + d.body_text, STOPWORDS, min_token_len)
        all_tokens.extend(toks)
        token_counts.update(toks)

    rates = rates_from_tokens(all_tokens)
    return rates, token_counts


def kl_div(p: Dict[str, float], q: Dict[str, float]) -> float:
    eps = 1e-12
    s = 0.0
    for k, pv in p.items():
        qv = q.get(k, eps)
        s += pv * math.log((pv + eps) / (qv + eps))
    return s


def jensen_shannon_distance(counts_a: Counter, counts_b: Counter, top_vocab: int = 5000) -> float:
    """
    JSD between token distributions, capped to top_vocab terms to keep it stable.
    Returns sqrt(JSD) in [0, 1-ish].
    """
    # Build capped vocab
    vocab = set([t for t, _ in counts_a.most_common(top_vocab)]) | set([t for t, _ in counts_b.most_common(top_vocab)])
    if not vocab:
        return 0.0

    total_a = sum(counts_a[t] for t in vocab)
    total_b = sum(counts_b[t] for t in vocab)
    if total_a <= 0 or total_b <= 0:
        return 0.0

    p = {t: counts_a[t] / total_a for t in vocab}
    q = {t: counts_b[t] / total_b for t in vocab}
    m = {t: 0.5 * (p[t] + q[t]) for t in vocab}

    jsd = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    return math.sqrt(max(jsd, 0.0))


def delta(a: Dict[str, float], b: Dict[str, float], key: str) -> float:
    return float(a.get(key, 0.0) - b.get(key, 0.0))


def write_markdown(out_path: Path, result: dict) -> None:
    meta = result["meta"]
    cur = result["current"]
    base = result["baseline"]
    dlt = result["delta"]

    lines: List[str] = []
    lines.append("# Chaos Observatory — Sentiment Shift")
    lines.append("")
    lines.append(f"**Generated (UTC):** {meta['generated_at_utc']}")
    lines.append(f"**Current window (UTC):** {meta['current_start']} → {meta['current_end']}  ")
    lines.append(f"**Baseline window (UTC):** {meta['baseline_start']} → {meta['baseline_end']}")
    lines.append(f"**Group-by:** {meta['group_by']}")
    lines.append("")
    lines.append("## Global Shift Summary")
    lines.append("")
    lines.append("| Metric | Current | Baseline | Delta |")
    lines.append("|---|---:|---:|---:|")
    for k in ["pos_rate","neg_rate","urgency_rate","certainty_rate","hedge_rate","compression"]:
        lines.append(f"| {k} | {cur[k]:.6f} | {base[k]:.6f} | {dlt[k]:.6f} |")
    lines.append("")
    lines.append("| Docs/Token Metric | Current | Baseline |")
    lines.append("|---|---:|---:|")
    lines.append(f"| docs | {meta['docs_current']} | {meta['docs_baseline']} |")
    lines.append(f"| tokens | {cur['tokens']} | {base['tokens']} |")
    if "lexical_js_distance" in result:
        lines.append("")
        lines.append(f"**Lexical distribution shift (JSD distance):** {result['lexical_js_distance']:.6f}")

    # Per-group
    per_group = result.get("per_group", {})
    if per_group:
        lines.append("")
        lines.append("## Per-Group Shift (Top by |compression delta|)")
        lines.append("")
        lines.append("| Group | Docs(cur/base) | compression Δ | neg Δ | urgency Δ | certainty Δ | pos Δ | hedge Δ |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for g, row in list(per_group.items())[:25]:
            lines.append(
                f"| {g.replace('|','\\\\|')} | {row['docs_current']}/{row['docs_baseline']} | "
                f"{row['delta']['compression']:.6f} | {row['delta']['neg_rate']:.6f} | "
                f"{row['delta']['urgency_rate']:.6f} | {row['delta']['certainty_rate']:.6f} | "
                f"{row['delta']['pos_rate']:.6f} | {row['delta']['hedge_rate']:.6f} |"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized-dir", default="data/normalized", help="Base normalized partition dir")
    ap.add_argument("--end-date", default=None, help="UTC end date YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--window-days", type=int, default=7, help="Current window length (days)")
    ap.add_argument("--baseline-days", type=int, default=7, help="Baseline window length (days)")
    ap.add_argument("--group-by", choices=["region", "source"], default="region", help="Group docs by region or source")
    ap.add_argument("--min-docs", type=int, default=10, help="Minimum docs per window required (global)")
    ap.add_argument("--min-docs-per-group", type=int, default=10, help="Minimum docs per group per window")
    ap.add_argument("--min-token-len", type=int, default=3, help="Minimum token length")
    ap.add_argument("--per-group", action="store_true", help="Also compute per-group shifts")
    ap.add_argument("--jsd", action="store_true", help="Compute lexical Jensen–Shannon distance between windows")
    ap.add_argument("--md-out", default=None, help="Optional markdown output path")
    args = ap.parse_args()

    if args.window_days <= 0 or args.baseline_days <= 0:
        raise SystemExit("ERROR: window-days and baseline-days must be > 0")

    base_dir = Path(args.normalized_dir)
    end_day = parse_ymd(args.end_date) if args.end_date else utc_midnight_today()
    cur_start = end_day - timedelta(days=args.window_days - 1)

    base_end = cur_start - timedelta(days=1)
    base_start = base_end - timedelta(days=args.baseline_days - 1)

    docs_cur = load_docs(base_dir, cur_start, end_day, group_by=args.group_by)
    docs_base = load_docs(base_dir, base_start, base_end, group_by=args.group_by)

    if len(docs_cur) < args.min_docs or len(docs_base) < args.min_docs:
        out = {
            "event": "sentiment_shift",
            "error": "insufficient_docs",
            "docs_current": len(docs_cur),
            "docs_baseline": len(docs_base),
            "min_docs": args.min_docs,
            "current_window": {"start": day_to_partition(cur_start), "end": day_to_partition(end_day)},
            "baseline_window": {"start": day_to_partition(base_start), "end": day_to_partition(base_end)},
        }
        print(json.dumps(out, ensure_ascii=False))
        return 2

    cur_rates, cur_counts = analyze_docs(docs_cur, args.min_token_len)
    base_rates, base_counts = analyze_docs(docs_base, args.min_token_len)

    dlt = {
        "pos_rate": delta(cur_rates, base_rates, "pos_rate"),
        "neg_rate": delta(cur_rates, base_rates, "neg_rate"),
        "urgency_rate": delta(cur_rates, base_rates, "urgency_rate"),
        "certainty_rate": delta(cur_rates, base_rates, "certainty_rate"),
        "hedge_rate": delta(cur_rates, base_rates, "hedge_rate"),
        "compression": delta(cur_rates, base_rates, "compression"),
    }

    result = {
        "event": "sentiment_shift",
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "current_start": day_to_partition(cur_start),
            "current_end": day_to_partition(end_day),
            "baseline_start": day_to_partition(base_start),
            "baseline_end": day_to_partition(base_end),
            "group_by": args.group_by,
            "docs_current": len(docs_cur),
            "docs_baseline": len(docs_base),
            "lexicons": {
                "pos_words": len(POS_WORDS),
                "neg_words": len(NEG_WORDS),
                "urgency_words": len(URGENCY_WORDS),
                "certainty_words": len(CERTAINTY_WORDS),
                "hedge_words": len(HEDGE_WORDS),
            },
        },
        "current": cur_rates,
        "baseline": base_rates,
        "delta": dlt,
    }

    if args.jsd:
        result["lexical_js_distance"] = jensen_shannon_distance(cur_counts, base_counts)

    # Per-group shifts
    if args.per_group:
        cur_by: Dict[str, List[Doc]] = defaultdict(list)
        base_by: Dict[str, List[Doc]] = defaultdict(list)
        for d in docs_cur:
            cur_by[d.group].append(d)
        for d in docs_base:
            base_by[d.group].append(d)

        per_group_rows: List[Tuple[str, dict]] = []
        for g in sorted(set(cur_by.keys()) & set(base_by.keys())):
            cur_docs_g = cur_by[g]
            base_docs_g = base_by[g]
            if len(cur_docs_g) < args.min_docs_per_group or len(base_docs_g) < args.min_docs_per_group:
                continue

            g_cur, _ = analyze_docs(cur_docs_g, args.min_token_len)
            g_base, _ = analyze_docs(base_docs_g, args.min_token_len)

            g_delta = {
                "pos_rate": delta(g_cur, g_base, "pos_rate"),
                "neg_rate": delta(g_cur, g_base, "neg_rate"),
                "urgency_rate": delta(g_cur, g_base, "urgency_rate"),
                "certainty_rate": delta(g_cur, g_base, "certainty_rate"),
                "hedge_rate": delta(g_cur, g_base, "hedge_rate"),
                "compression": delta(g_cur, g_base, "compression"),
            }

            per_group_rows.append(
                (
                    g,
                    {
                        "docs_current": len(cur_docs_g),
                        "docs_baseline": len(base_docs_g),
                        "current": g_cur,
                        "baseline": g_base,
                        "delta": g_delta,
                    },
                )
            )

        # Sort by biggest absolute compression delta
        per_group_rows.sort(key=lambda kv: abs(kv[1]["delta"]["compression"]), reverse=True)
        result["per_group"] = {g: row for g, row in per_group_rows}

    if args.md_out:
        md_path = Path(args.md_out)
        write_markdown(md_path, result)
        result["markdown_written_to"] = str(md_path)

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
