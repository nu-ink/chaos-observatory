#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/home/keystone/chaos-observatory/chaos-observatory}"
VENV="$BASE_DIR/.venv"
TODAY="$(date -u +%F)"

cd "$BASE_DIR"

source "$VENV/bin/activate"

python ingest/rss_collector.py \
  --sources ingest/sources.yaml \
  --outdir data/raw

mkdir -p "data/normalized/$TODAY"
python ingest/normalize.py \
  --in "data/raw/$TODAY" \
  --out "data/normalized/$TODAY"

python analyze/frequency_drift.py \
  --normalized-dir data/normalized \
  --end-date "$TODAY"
