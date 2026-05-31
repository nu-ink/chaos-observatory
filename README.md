# Chaos Observatory

Chaos Observatory is a text-only global signal observatory. It collects public RSS feeds, stores date-partitioned raw JSONL, normalizes those items into analysis-ready JSONL, and runs lightweight explainable analysis over recent windows.

**Chaos-Observatory** is an experimental global-signal intelligence platform designed to observe, correlate, and reason about cascading events across the world, from natural disasters to economic shocks, social unrest, and geopolitical consequences.

The current build is intentionally small and auditable. It is not a prediction engine; it is a pipeline for watching how public signals change across sources, regions, and topics.

Examples of real-world signal chains:

- Earthquake -> Tsunami alert -> Port shutdowns -> Supply chain disruption
- Cyberattack -> Media silence -> Financial volatility
- Political unrest -> Currency pressure -> Capital flight
- Disease outbreak -> News saturation -> Public fatigue -> Silence risk

## Current Build

Implemented today:

- RSS ingestion from `ingest/sources.yaml`
- UTC date-partitioned raw output under `data/raw/YYYY-MM-DD/`
- JSONL normalization with file or directory input
- Analysis modules for frequency drift, topic convergence, silence detection, and sentiment shift
- Markdown weekly report generation
- Dry-run-first retention management for raw and normalized partitions
- SQLite schema and helper wrapper in `storage/`
- A hardened systemd service template in `systemd/`
- Basic ingest/normalization tests
- A working `run_pipeline.py` orchestrator for ingest, normalize, analysis, and report generation

In progress or rough:

- `run_15min.sh` is a lightweight shell helper for ingest, normalize, and frequency drift.
- The ML folder is a placeholder.

## System Architecture

```text
        Sources
   (RSS, Feeds, Alerts)
          |
          v
     Ingest Layer
 (`ingest/rss_collector.py`)
          |
          v
     Normalization
 (`ingest/normalize.py`)
          |
          v
     Storage Layer
 (`storage/db.py`, SQLite)
          |
          v
     Analysis Engines
       (`analyze/`)
          |
          v
     Reports & Signals
```

## Repository Layout

```text
chaos-observatory/
├── analyze/                  # Analysis engines
│   ├── frequency_drift.py
│   ├── topic_convergence.py
│   ├── silence_detection.py
│   └── sentiment_shift.py
├── config/
│   └── chaos.yaml            # Current configuration reference
├── hold/
│   └── retention.py          # Retention planning and apply mode
├── ingest/
│   ├── sources.yaml          # RSS source registry
│   ├── rss_collector.py      # Public RSS ingestion
│   └── normalize.py          # JSONL normalization
├── ml/                       # Interpretable ML monitoring experiments
│   ├── ml_change_detection.py
│   └── ml_topic_convergence.py
├── report/
│   └── weekly_report.py      # Markdown weekly report
├── storage/
│   ├── schema.sql            # SQLite-first schema
│   ├── db.py                 # SQLite helper wrapper
│   └── chaos.db              # Local database artifact, if present
├── systemd/
│   └── chaos-observatory.service.ini
├── tests/
│   └── test_ingest.py
├── run_pipeline.py           # Pipeline orchestrator
├── run_15min.sh              # Lightweight shell helper
└── requirements.txt
```

Runtime data is written to `data/` and `reports/`. Those directories may not exist until the first run and are not committed to the repository.

## Requirements

- Python 3.11+
- pip and venv
- Network access for RSS feeds
- SQLite for local storage helpers

Install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Source Configuration

RSS feeds live in `ingest/sources.yaml`.

The current registry includes:

- BBC News - World
- US Federal Reserve press releases
- European Central Bank press releases
- ReliefWeb updates

Each source has an `id`, display label, region, category, enabled flag, and one or more RSS feed URLs.

## Pipeline Run

Run a full cycle from the repository root with the virtual environment activated:

```bash
python run_pipeline.py
```

The runner collects RSS items, normalizes the current UTC day, runs analyzers, and writes a weekly report. Some analyzers may report `insufficient_docs` until enough baseline partitions exist.

## Manual Run

Set the UTC day used by the date-partitioned pipeline:

```bash
DAY=$(date -u +%F)
```

Collect raw RSS items:

```bash
python ingest/rss_collector.py \
  --sources ingest/sources.yaml \
  --outdir data/raw
```

Normalize the current day:

```bash
mkdir -p "data/normalized/${DAY}"
python ingest/normalize.py \
  --in "data/raw/${DAY}" \
  --out "data/normalized/${DAY}"
```

Generate a weekly report:

```bash
python report/weekly_report.py \
  --normalized-dir data/normalized \
  --outdir reports \
  --window-days 7 \
  --baseline-days 7
```

The report is written to:

```text
reports/YYYY-MM-DD/weekly_report.md
```

## Analysis Commands

Each analyzer can be run independently against `data/normalized`.

```bash
python analyze/frequency_drift.py --normalized-dir data/normalized
python analyze/topic_convergence.py --normalized-dir data/normalized
python analyze/silence_detection.py --normalized-dir data/normalized
python analyze/sentiment_shift.py --normalized-dir data/normalized
```

Most analyzers support:

- `--end-date YYYY-MM-DD`
- `--window-days N`
- `--baseline-days N`
- `--md-out path/to/output.md`

Use `--help` on any analyzer for the full option list.

## ML Commands

ML modules are experimental monitoring algorithms and do not replace the
deterministic analyzers under `analyze/`.

```bash
python ml/ml_change_detection.py --normalized-dir data/normalized
python ml/ml_topic_convergence.py --normalized-dir data/normalized
```

## Retention

Retention is dry-run by default.

Preview retention decisions:

```bash
python hold/retention.py \
  --raw-dir data/raw \
  --normalized-dir data/normalized \
  --hold-days 7 \
  --keep-days 30
```

Apply retention:

```bash
python hold/retention.py \
  --raw-dir data/raw \
  --normalized-dir data/normalized \
  --hold-days 7 \
  --keep-days 30 \
  --apply
```

Archive instead of deleting old partitions:

```bash
python hold/retention.py \
  --raw-dir data/raw \
  --normalized-dir data/normalized \
  --hold-days 7 \
  --keep-days 30 \
  --archive-dir /path/to/archive \
  --apply
```

## Database Layer

`storage/schema.sql` defines a SQLite-first schema for:

- pipeline runs
- sources
- raw items
- normalized documents
- analysis results
- optional FTS5 document search

`storage/db.py` provides a minimal SQLite wrapper with schema initialization, run tracking, source upserts, raw item inserts, document storage, and analysis result storage.

The JSONL pipeline is currently the main runtime path. The database layer is present for the next build stage.

## systemd

`systemd/chaos-observatory.service.ini` contains a hardened oneshot service template. It runs the manual pipeline shape directly:

1. collect RSS
2. normalize current-day JSONL files
3. generate a weekly report
4. run retention in dry-run mode

Before installing it, update paths such as `WorkingDirectory`, `VIRTUAL_ENV`, and `ReadWritePaths` for the target machine.

## Tests

Run the current test suite with:

```bash
pytest
```

The tests currently focus on source loading, RSS item shaping, and normalization behavior.

## Design Notes

Chaos Observatory favors explainable, inspectable signals over opaque scoring. Current analysis is based on counts, term drift, TF-IDF convergence, source/region group comparisons, silence/dropout checks, and simple lexicon sentiment. Treat results as observational leads for human review, not conclusions.
