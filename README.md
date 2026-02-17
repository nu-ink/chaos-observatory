

Chaos-Observatory

**Chaos-Observatory** is an experimental global-signal intelligence platform designed to observe, correlate, and reason about cascading events across the world— from natural disasters to economic shocks, social unrest, and geopolitical consequences.

It ingests heterogeneous data streams (news, RSS, seismic feeds, alerts), normalizes them into a unified timeline, and applies analytical models to detect **patterns of escalation, convergence, silence, and drift** across domains.

> **Mission**
> Detect early signals of chaos, track how events propagate across systems, and surface actionable insight before consequences fully unfold.


What Chaos-Observatory Can Detect

Examples of real-world signal chains:

* Earthquake → Tsunami alert → Port shutdowns → Supply chain disruption
* Cyberattack → Media silence → Financial volatility
* Political unrest → Currency pressure → Capital flight
* Disease outbreak → News saturation → Public fatigue → Silence risk

This system is *not predictive AI* (yet) — it is a signal observatory, designed for humans-in-the-loop analysis.



System Architecture (High Level)

```text
        ┌────────────┐
        │   Sources  │  (RSS, Feeds, Alerts)
        └─────┬──────┘
              │
              ▼
       ┌──────────────┐
       │  Ingest Layer│  (`ingest/rss_collector.py`)
       └─────┬────────┘
         ▼
         ┌─────────────────┐
         │ Normalization   │  (`ingest/normalize.py`)
         │ & Canonical Data│
         └─────┬───────────┘
           ▼
         ┌─────────────────┐
         │  Storage Layer  │  (SQLite / Postgres via `storage/db.py`)
         └─────┬───────────┘
           ▼
       ┌───────────────────────┐
       │ Analysis Engines      │  (in `analyze/`)
       │ • Frequency Drift     │  (`analyze/frequency_drift.py`)
       │ • Topic Convergence   │  (`analyze/topic_convergence.py`)
       │ • Sentiment Shift     │  (`analyze/sentiment_shift.py`)
       │ • Silence Detection   │  (`analyze/silence_detection.py`)
       │ • Retention Decay     │  (`hold/retention.py`)
       └─────┬─────────────────┘
         ▼
   ┌───────────────────────┐
   │ Reports & Signals     │
   │ (weekly, alerts, diffs)│
   └───────────────────────┘
```



**Project Structure**

```text
chaos-observatory/
├── ingest/                  # rss_collector.py, normalize.py, sources.yaml
├── analyze/                 # analysis engines (frequency, topic, sentiment, silence)
├── report/                  # weekly_report.py
├── storage/                 # schema.sql, db.py
├── config/                  # chaos.yaml
├── hold/                    # retention and retention review notes
├── systemd/                 # service unit (packaged, not installed)
├── requirements.txt
├── Pipfile
├── README.md
└── .venv/
```



## ⚙️ Requirements

### System

* **OS**: Ubuntu 22.04+ (24.04 LTS recommended)
* **CPU**: x86_64 (ARM compatible with adjustments)
* **RAM**: 8 GB minimum (16+ GB recommended)
* **Disk**: 20 GB+ free

### Software

- **Python**: 3.11+ (3.12 also tested in dev environment)
* **Database**:

  * SQLite (default, local)
  * PostgreSQL 15+ (recommended for scale)



Python Environment Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



Configuration

### `config/chaos.yaml`

```yaml
storage:
  backend: sqlite
  path: storage/chaos.db

ingest:
  poll_interval_minutes: 15

analysis:
  sentiment_enabled: true
  silence_threshold_hours: 48

reporting:
  weekly_day: Sunday
```



## Data Ingestion

### Define Sources (`ingest/sources.yaml`)

```yaml
- name: USGS Earthquakes
  type: rss
  url: https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.atom

- name: Global News
  type: rss
  url: https://feeds.bbci.co.uk/news/world/rss.xml
```

### Run Ingestion

```bash
python ingest/rss_collector.py
```

By convention raw/normalized output is organized by date (for example `data/raw/YYYY-MM-DD` and `data/normalized/YYYY-MM-DD`). These `data/` directories are not committed to the repository — create them as needed or configure `config/chaos.yaml` to point to your storage paths.



## Normalization

Normalize **directories or individual files**:

```bash
python ingest/normalize.py \
  --in data/raw/2026-01-05 \
  --out data/normalized/2026-01-05
```

Outputs **JSONL canonical events**.



## Analysis Engines

### Frequency Drift

Detects sudden increases or decreases in event volume.

### Topic Convergence

Identifies when unrelated domains begin sharing vocabulary.

### Sentiment Shift

Tracks emotional movement (neutral → alarm → panic).

### Silence Detection

Flags **dangerous absence of expected signals**.

### Retention Decay

Models how long topics persist before fading from discourse.

Run all analysis (scripts live under `analyze/`, retention lives under `hold/`):

```bash
python analyze/frequency_drift.py
python analyze/topic_convergence.py
python analyze/sentiment_shift.py
python analyze/silence_detection.py
python hold/retention.py
```



## Reporting

Generate a weekly intelligence summary:

```bash
python report/weekly_report.py
```

Output includes:

* Notable escalations
* Cross-domain convergence
* Silence warnings
* Emerging global narratives



## Run as a Service (systemd)

```bash
# Copy and enable service (packaged unit file is in systemd/)
sudo cp systemd/chaos-observatory.service.ini /etc/systemd/system/chaos-observatory.service
sudo systemctl daemon-reexec
sudo systemctl enable chaos-observatory
sudo systemctl start chaos-observatory
```



## Design Principles

* **Human-in-the-Loop** (not autonomous decision-making)
* **Explainable signals**
* **Composable analytics**
* **Low dependency surface**
* **Data locality**



## Roadmap

* [ ] Event graph linking (cause → effect)
* [ ] Geo-temporal clustering
* [ ] LLM-assisted summarization (offline capable)
* [ ] Alert thresholds & notifications
* [ ] Multi-language ingestion
* [ ] Dashboard UI



## Disclaimer

Chaos-Observatory is an **experimental research system**.
It does not provide predictions, financial advice, or emergency alerts.

Use responsibly.



##  Philosophy

> *“Chaos is rarely random. It only appears so when viewed in isolation.”*





