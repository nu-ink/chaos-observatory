#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/keystone/chaos-observatory/chaos-observatory"
VENV="$BASE_DIR/.venv"
TODAY=$(date +%F)

cd "$BASE_DIR"

source "$VENV/bin/activate"

# Ingest
python ingest/rss_collector.py

# Normalize
mkdir -p "data/normalized/$TODAY"
python ingest/normalize.py \
  --in "data/raw/$TODAY" \
  --out "data/normalized/$TODAY"

# Optional: lightweight analysis
python analysis/frequency_drift.py
