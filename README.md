# Chaos Observatory

Chaos Observatory is a text-only global signal observatory. It collects public RSS feeds, stores raw and normalized records by UTC day, runs explainable analysis over recent windows, and keeps an experimental ML layer for semantic linking, anomaly detection, clustering, and tone monitoring.

The project is intentionally small and inspectable. It is not a prediction engine. The current build is designed to answer:

- What public signals changed?
- Which sources, regions, or topics are moving together?
- Which signals went quiet?
- Which tone or volume shifts deserve human review?
- Which differently worded items appear to describe the same underlying event?

Example signal chains the system is meant to observe:

- Earthquake -> tsunami alert -> port shutdowns -> supply chain disruption
- Cyberattack -> media silence -> financial volatility
- Political unrest -> currency pressure -> capital flight
- Disease outbreak -> news saturation -> public fatigue -> silence risk

## Current Build

The working build has five main layers:

1. **Collection and normalization**
   - RSS ingestion from `ingest/sources.yaml`
   - Retry/source expansion support in `ingest/sources_retry.yaml`
   - Raw JSONL output under `data/raw/YYYY-MM-DD/`
   - Normalized JSONL output under `data/normalized/YYYY-MM-DD/`

2. **Deterministic analysis**
   - Frequency drift
   - Topic convergence
   - Silence detection
   - Sentiment shift
   - Markdown weekly reports

3. **Storage and operations**
   - SQLite schema in `storage/schema.sql`
   - SQLite helper wrapper in `storage/db.py`
   - Normalized JSONL -> SQLite import in `scripts/import_normalized_to_db.py`
   - Retention planning in `hold/retention.py`
   - `systemd` service template
   - `run_pipeline.py` orchestration

4. **Health, safety, and validation**
   - Pre-flight checks under `chaos_observatory/health/`
   - JSON health output via `scripts/print_health.py`
   - Pre-commit configuration in `.pre-commit-config.yaml`
   - Security scan artifacts from Bandit, pip-audit, Safety, and TruffleHog

5. **ML experiments**
   - Change-point detection with `ruptures`
   - Semantic embeddings with `sentence-transformers`
   - FAISS vector-store helpers
   - Semantic article linking
   - ML-ready tone/sentiment shift script
   - Topic clustering and convergence over SQLite documents
   - Evaluation and threshold helpers

The deterministic `analyze/` layer remains the main observatory path. The ML layer is functional enough for validation and review workflows, but should still be treated as experimental unless a workflow has explicit tests, thresholds, and human review.

## Architecture

```text
      Public Sources
   (RSS feeds, alerts)
            |
            v
       Ingest Layer
  ingest/rss_collector.py
            |
            v
      Normalization
    ingest/normalize.py
            |
            v
   Date-Partitioned JSONL
 data/raw + data/normalized
            |
            +----------------------+
            |                      |
            v                      v
   Explainable Analyzers      SQLite Import
        analyze/          scripts/import_normalized_to_db.py
            |                      |
            v                      v
    Reports and Signals       ML Helpers
        report/                  ml/
                                   |
                                   v
                         semantic links, clusters,
                         anomaly alerts, tone shifts,
                         review artifacts
```

## Repository Structure

```text
chaos-observatory/
├── analyze/                     # Deterministic, explainable analyzers
├── chaos_observatory/health/     # Pre-flight health checks
├── config/
│   └── chaos.yaml               # Runtime and analysis configuration
├── hold/
│   └── retention.py             # Dry-run-first retention planning
├── ingest/
│   ├── sources.yaml             # RSS source registry
│   ├── sources_retry.yaml       # Alternate/retry source list
│   ├── rss_collector.py         # Feed collection
│   └── normalize.py             # Raw JSONL -> normalized JSONL
├── ml/                          # Experimental ML monitoring components
├── report/
│   └── weekly_report.py         # Markdown report generation
├── scripts/
│   ├── import_normalized_to_db.py
│   ├── print_health.py
│   └── run_semantic_linker_demo.py
├── storage/
│   ├── schema.sql               # SQLite-first schema
│   ├── db.py                    # SQLite helper wrapper
│   └── chaos.db                 # Local SQLite artifact, if present
├── systemd/
│   └── chaos-observatory.service.ini
├── tests/                       # Pytest coverage for ingest, health, ML, and analyzers
├── run_pipeline.py              # Full ingest -> normalize -> analyze -> report runner
├── run_15min.sh                 # Lightweight shell helper
├── requirements.txt             # Runtime dependencies
└── pyproject.toml               # Project/test metadata
```

Runtime data is written under `data/`, `reports/`, `logs/`, `storage/`, and selected ML output paths. Treat those paths as operational artifacts rather than core source code.

## Requirements

- Python 3.11+
- pip and venv
- Network access for RSS ingestion
- Network access for first-time model/dependency downloads
- SQLite

Install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Important runtime and ML dependencies include:

- `requests`, `feedparser`, `PyYAML`, `python-dateutil`
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `ruptures`
- `sentence-transformers`
- `faiss-cpu`
- `SQLAlchemy`
- `vaderSentiment`, `textblob`

## Source Configuration

RSS feeds live in `ingest/sources.yaml`.

The current registry has 60 enabled sources across global news, central-bank updates, humanitarian feeds, business/startup feeds, technology feeds, and selected high-volume public aggregators. Each source declares an ID, label, region, category, enabled flag, and feed URLs.

Project-level settings live in `config/chaos.yaml`, including:

- data paths
- analysis windows
- analyzer thresholds
- report settings
- semantic similarity settings
- retention defaults
- scheduling-oriented defaults

## Running The Pipeline

Run a full cycle from the repository root:

```bash
python run_pipeline.py
```

The runner performs:

1. RSS ingest
2. normalization for the current UTC day
3. deterministic analysis
4. weekly report generation

Some analyzers may return `insufficient_docs` until enough baseline partitions exist. The pipeline treats that as a warning for analyzer steps that explicitly report incomplete data.

## Manual Pipeline

Set the UTC day:

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

## Analysis Commands

Run deterministic analyzers independently:

```bash
python analyze/frequency_drift.py --normalized-dir data/normalized
python analyze/topic_convergence.py --normalized-dir data/normalized
python analyze/silence_detection.py --normalized-dir data/normalized
python analyze/sentiment_shift.py --normalized-dir data/normalized
```

Common options include:

- `--end-date YYYY-MM-DD`
- `--window-days N`
- `--baseline-days N`
- `--md-out path/to/report.md`

Use `--help` on any analyzer for exact options.

## Health Checks

Run pre-flight health checks:

```bash
python -m chaos_observatory.health.run_healthcheck
```

Emit health checks as JSON:

```bash
python -m chaos_observatory.health.run_healthcheck --json
python scripts/print_health.py
```

The health layer checks expected configuration, database readiness, ingest paths, ML importability, and report paths.

## Database Layer

`storage/schema.sql` defines a SQLite-first schema for:

- pipeline runs
- sources
- raw items
- normalized documents
- analysis results
- optional FTS5 document search

`storage/db.py` provides helper methods for schema initialization, run tracking, source upserts, raw item storage, normalized document storage, document-window queries, and analysis result storage.

Import normalized JSONL partitions into SQLite for ML workflows:

```bash
python scripts/import_normalized_to_db.py \
  --in data/normalized \
  --db storage/chaos.db
```

The JSONL pipeline is still the main runtime path. SQLite is available for persistence, document-window queries, and ML workflows that need stable document storage.

## ML Components

The `ml/` directory is a connected experimental layer. It does not replace `analyze/`; it adds monitoring and semantic tools that can mature into a later ML pipeline.

### Change Detection

Reads normalized JSONL partitions and detects source/topic count changes.

```bash
python ml/ml_change_detection.py \
  --normalized-dir data/normalized \
  --end-date 2026-05-30 \
  --csv-out /tmp/ml_change_alerts.csv
```

### Topic Convergence

Reads SQLite records from the `documents`, `articles`, or `normalized_items` table, embeds text, clusters documents, and writes a Markdown report.

```bash
python ml/ml_topic_convergence.py \
  --db storage/chaos.db \
  --since 2026-05-30 \
  --output /tmp/ml_topic_report.md
```

If `storage/chaos.db` has no documents, the command exits cleanly with a no-records message.

### Semantic Linking

The semantic linker combines:

- `ml/ml_embeddings.py`
- `ml/vector_store.py`
- `ml/ml_similarity_thresholds.py`
- `ml/ml_semantic_linker.py`

It is designed to embed articles, store vectors in FAISS, classify similarity scores, and export human review CSVs.

Demo script:

```bash
python scripts/run_semantic_linker_demo.py
```

The first run may download a sentence-transformer model and write FAISS/database artifacts under configured paths.

### Sentiment / Tone Shift

The ML-ready sentiment shift script reads CSV or JSON, scores tone with domain lexicons, detects source/topic tone shifts, and writes JSON plus Markdown output.

```bash
python ml/ml_sentiment_shift.py \
  --input path/to/articles.csv \
  --scores-output /tmp/sentiment_scores.json \
  --alerts-output /tmp/tone_shift_alerts.json \
  --report-output /tmp/sentiment_shift_report.md
```

Expected input fields:

- `article_id`
- `source`
- `topic`
- `title`
- `text`
- `published_at`

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

Archive instead of deleting:

```bash
python hold/retention.py \
  --raw-dir data/raw \
  --normalized-dir data/normalized \
  --hold-days 7 \
  --keep-days 30 \
  --archive-dir /path/to/archive \
  --apply
```

## Tests And Validation

Run the full test suite:

```bash
pytest
```

Current coverage includes:

- source loading and RSS item shaping
- normalization
- health checks
- database document-window queries
- ML change detection helpers
- ML topic convergence database loading
- embedding text construction
- vector-store add/search
- similarity threshold classification
- evaluation metrics
- ML sentiment shift scoring and CLI output

Useful validation commands:

```bash
python -m py_compile analyze/*.py ingest/*.py ml/*.py report/*.py storage/*.py
python -m pytest
python -c "import ml; import ml.ml_change_detection; import ml.ml_topic_convergence; import ml.ml_embeddings; import ml.ml_evaluation; import ml.ml_semantic_linker; import ml.ml_sentiment_shift; import ml.ml_similarity_thresholds; import ml.vector_store; print('all-ml-imports-ok')"
```

## Security And Pre-Commit

The repository includes pre-commit and scan artifacts:

- `.pre-commit-config.yaml`
- `.secrets.baseline`
- `scan-bandit.json`
- `scan-pip-audit.json`
- `scan-safety.json`
- `scan-trufflehog.json`
- `trufflehog-results.json`

The current pre-commit file points at local virtual-environment executables. Check those paths before relying on hooks across operating systems.

## Operational Notes

`systemd/chaos-observatory.service.ini` is a hardened oneshot service template. Before installing it, update machine-specific paths such as:

- `WorkingDirectory`
- `VIRTUAL_ENV`
- `ReadWritePaths`

`run_15min.sh` is a lightweight helper for short-interval local runs.

## Design Direction

Chaos Observatory favors explainable, inspectable signals before opaque modeling.

Near-term direction:

- Keep deterministic analyzers stable and auditable.
- Use ML for monitoring, similarity, clustering, and review assistance.
- Keep humans in the loop for labels, thresholds, and validation.
- Promote ML modules into the main pipeline only after they have reliable data contracts and tests.

The current shape is best understood as:

```text
deterministic observatory first
ML-assisted review second
prediction claims never implied
```
