-- Chaos-Observatory schema (v1)
-- SQLite-first. Designed to be portable to Postgres with minimal changes.

PRAGMA foreign_keys = ON;

-- ----------------------------
-- Runs: capture each pipeline run
-- ----------------------------
CREATE TABLE IF NOT EXISTS runs (
  run_id            TEXT PRIMARY KEY,
  started_at_utc    TEXT NOT NULL,
  finished_at_utc   TEXT,
  status            TEXT NOT NULL,          -- ok | error | partial
  mode              TEXT NOT NULL,          -- ingest | normalize | analyze | report | daily
  config_path       TEXT,
  notes             TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at_utc);

-- ----------------------------
-- Sources registry (optional but useful)
-- ----------------------------
CREATE TABLE IF NOT EXISTS sources (
  source_id     TEXT PRIMARY KEY,
  label         TEXT,
  region        TEXT,
  category      TEXT,
  feed_url      TEXT,
  enabled       INTEGER NOT NULL DEFAULT 1,
  created_at_utc TEXT NOT NULL,
  updated_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_region ON sources(region);
CREATE INDEX IF NOT EXISTS idx_sources_category ON sources(category);

-- ----------------------------
-- Raw items (as collected)
-- Store as JSON; keep minimal extracted fields for indexing.
-- ----------------------------
CREATE TABLE IF NOT EXISTS raw_items (
  raw_id           TEXT PRIMARY KEY,      -- stable hash (source_id + url + published + title)
  run_id           TEXT,                  -- link to runs
  ingested_at_utc  TEXT NOT NULL,
  source_id        TEXT NOT NULL,
  feed_url         TEXT,
  title            TEXT,
  url              TEXT,
  published_raw    TEXT,                  -- as provided in feed
  raw_json         TEXT NOT NULL,          -- full raw record as JSON string
  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE SET NULL,
  FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_raw_items_ingested ON raw_items(ingested_at_utc);
CREATE INDEX IF NOT EXISTS idx_raw_items_source ON raw_items(source_id);
CREATE INDEX IF NOT EXISTS idx_raw_items_published_raw ON raw_items(published_raw);
CREATE INDEX IF NOT EXISTS idx_raw_items_url ON raw_items(url);

-- ----------------------------
-- Normalized documents
-- One row per normalized doc id (hash-based).
-- ----------------------------
CREATE TABLE IF NOT EXISTS documents (
  doc_id             TEXT PRIMARY KEY,
  run_id             TEXT,                  -- run that wrote/updated this record
  normalized_at_utc  TEXT NOT NULL,
  ingested_at_utc    TEXT,                  -- from raw pipeline
  published_at_utc   TEXT,                  -- best-effort parsed
  source_id          TEXT NOT NULL,
  source_label       TEXT,
  region             TEXT,
  category           TEXT,
  feed_url           TEXT,
  url                TEXT,
  title              TEXT,
  body_text          TEXT,
  raw_json           TEXT,                  -- optional audit copy (can be null)
  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE SET NULL,
  FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_docs_published ON documents(published_at_utc);
CREATE INDEX IF NOT EXISTS idx_docs_source ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_docs_region ON documents(region);
CREATE INDEX IF NOT EXISTS idx_docs_category ON documents(category);

-- ----------------------------
-- Analysis results
-- Store module outputs as JSON blobs keyed by window.
-- ----------------------------
CREATE TABLE IF NOT EXISTS analysis_results (
  result_id         TEXT PRIMARY KEY,      -- stable hash (module + window + group_by + params)
  run_id            TEXT,
  created_at_utc    TEXT NOT NULL,
  module            TEXT NOT NULL,          -- frequency_drift | topic_convergence | silence_detection | sentiment_shift
  window_start_utc  TEXT NOT NULL,          -- YYYY-MM-DD
  window_end_utc    TEXT NOT NULL,          -- YYYY-MM-DD
  baseline_start_utc TEXT,                 -- YYYY-MM-DD (nullable)
  baseline_end_utc   TEXT,                 -- YYYY-MM-DD (nullable)
  group_by          TEXT,                  -- region | source | null
  params_json       TEXT NOT NULL,
  result_json       TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_results_module ON analysis_results(module);
CREATE INDEX IF NOT EXISTS idx_results_window ON analysis_results(window_start_utc, window_end_utc);

-- ----------------------------
-- Lightweight full-text search (SQLite FTS5)
-- Optional: if FTS5 not available, you can skip creating this.
-- ----------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
USING fts5(
  doc_id UNINDEXED,
  title,
  body_text,
  source_label,
  region,
  category,
  content='',
  tokenize='porter'
);
