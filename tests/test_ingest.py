#!/usr/bin/env python3
"""
Tests for Chaos-Observatory ingest module.

Tests both rss_collector.py and normalize.py functionality.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import modules under test
import sys
import importlib.util

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import modules directly using importlib (works without __init__.py)
spec_normalize = importlib.util.spec_from_file_location(
    "normalize", parent_dir / "ingest" / "normalize.py"
)
normalize = importlib.util.module_from_spec(spec_normalize)
spec_normalize.loader.exec_module(normalize)

spec_rss = importlib.util.spec_from_file_location(
    "rss_collector", parent_dir / "ingest" / "rss_collector.py"
)
rss_collector = importlib.util.module_from_spec(spec_rss)
spec_rss.loader.exec_module(rss_collector)


# ============================================================================
# Tests for rss_collector.py
# ============================================================================

class TestLoadSourcesYAML:
    """Test load_sources_yaml() function."""

    def test_load_basic_sources(self, tmp_path):
        """Test loading a basic sources.yaml file."""
        yaml_content = """
defaults:
  enabled: true
  fetch_timeout_sec: 20
  user_agent: "Test-Agent/1.0"
  max_items_per_feed: 50

sources:
  - id: test_source
    label: "Test Source"
    kind: rss
    region: test_region
    category: test_category
    enabled: true
    feeds:
      - url: "https://example.com/feed.xml"
"""
        yaml_file = tmp_path / "sources.yaml"
        yaml_file.write_text(yaml_content)

        cfg, sources = rss_collector.load_sources_yaml(yaml_file)

        assert isinstance(cfg, dict)
        assert len(sources) == 1
        assert sources[0].id == "test_source"
        assert sources[0].label == "Test Source"
        assert sources[0].kind == "rss"
        assert sources[0].region == "test_region"
        assert sources[0].category == "test_category"
        assert sources[0].enabled is True
        assert len(sources[0].feeds) == 1
        assert sources[0].feeds[0].url == "https://example.com/feed.xml"

    def test_load_with_defaults(self, tmp_path):
        """Test that defaults are applied correctly."""
        yaml_content = """
defaults:
  enabled: true
  fetch_timeout_sec: 30

sources:
  - id: source1
    feeds:
      - url: "https://example.com/feed.xml"
"""
        yaml_file = tmp_path / "sources.yaml"
        yaml_file.write_text(yaml_content)

        cfg, sources = rss_collector.load_sources_yaml(yaml_file)

        assert len(sources) == 1
        assert sources[0].label == "source1"  # Uses id as default label
        assert sources[0].kind == "rss"  # Default kind
        assert sources[0].region == "unknown"  # Default region
        assert sources[0].category == "unknown"  # Default category

    def test_disabled_source_filtered(self, tmp_path):
        """Test that disabled sources are filtered out."""
        yaml_content = """
defaults:
  enabled: true

sources:
  - id: enabled_source
    enabled: true
    feeds:
      - url: "https://example.com/feed1.xml"
  - id: disabled_source
    enabled: false
    feeds:
      - url: "https://example.com/feed2.xml"
"""
        yaml_file = tmp_path / "sources.yaml"
        yaml_file.write_text(yaml_content)

        cfg, sources = rss_collector.load_sources_yaml(yaml_file)

        assert len(sources) == 1
        assert sources[0].id == "enabled_source"

    def test_default_enabled_false(self, tmp_path):
        """Test when default enabled is false."""
        yaml_content = """
defaults:
  enabled: false

sources:
  - id: source1
    enabled: true  # Explicitly enabled
    feeds:
      - url: "https://example.com/feed.xml"
  - id: source2
    # Not explicitly enabled, uses default (false)
    feeds:
      - url: "https://example.com/feed2.xml"
"""
        yaml_file = tmp_path / "sources.yaml"
        yaml_file.write_text(yaml_content)

        cfg, sources = rss_collector.load_sources_yaml(yaml_file)

        assert len(sources) == 1
        assert sources[0].id == "source1"

    def test_multiple_feeds_per_source(self, tmp_path):
        """Test source with multiple feeds."""
        yaml_content = """
sources:
  - id: multi_feed_source
    feeds:
      - url: "https://example.com/feed1.xml"
      - url: "https://example.com/feed2.xml"
      - url: "https://example.com/feed3.xml"
"""
        yaml_file = tmp_path / "sources.yaml"
        yaml_file.write_text(yaml_content)

        cfg, sources = rss_collector.load_sources_yaml(yaml_file)

        assert len(sources) == 1
        assert len(sources[0].feeds) == 3


class TestNormalizeEntryMinimal:
    """Test normalize_entry_minimal() function."""

    def test_basic_entry(self):
        """Test normalizing a basic feed entry."""
        entry = {
            "title": "Test Title",
            "link": "https://example.com/article",
            "published": "2024-01-01T12:00:00Z",
            "summary": "Test summary text"
        }

        result = rss_collector.normalize_entry_minimal(entry)

        assert result["title"] == "Test Title"
        assert result["link"] == "https://example.com/article"
        assert result["published"] == "2024-01-01T12:00:00Z"
        assert result["summary"] == "Test summary text"
        assert "raw" in result
        assert result["raw"] == entry

    def test_missing_fields(self):
        """Test entry with missing optional fields."""
        entry = {
            "title": "Only Title"
        }

        result = rss_collector.normalize_entry_minimal(entry)

        assert result["title"] == "Only Title"
        assert result["link"] is None
        assert result["published"] is None
        assert result["summary"] is None

    def test_updated_fallback(self):
        """Test that 'updated' field is used as fallback for 'published'."""
        entry = {
            "title": "Test",
            "updated": "2024-01-01T12:00:00Z"
        }

        result = rss_collector.normalize_entry_minimal(entry)

        assert result["published"] == "2024-01-01T12:00:00Z"

    def test_description_fallback(self):
        """Test that 'description' field is used as fallback for 'summary'."""
        entry = {
            "title": "Test",
            "description": "Description text"
        }

        result = rss_collector.normalize_entry_minimal(entry)

        assert result["summary"] == "Description text"


class TestSafeGet:
    """Test safe_get() function."""

    def test_nested_access(self):
        """Test accessing nested dictionary keys."""
        d = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }

        result = rss_collector.safe_get(d, "level1", "level2", "level3")
        assert result == "value"

    def test_missing_key(self):
        """Test accessing missing key returns None."""
        d = {"level1": {"level2": "value"}}

        result = rss_collector.safe_get(d, "level1", "missing")
        assert result is None

    def test_non_dict_intermediate(self):
        """Test when intermediate value is not a dict."""
        d = {"level1": "not_a_dict"}

        result = rss_collector.safe_get(d, "level1", "level2")
        assert result is None

    def test_empty_path(self):
        """Test with empty key path."""
        d = {"key": "value"}

        result = rss_collector.safe_get(d)
        assert result == d


class TestWriteJSONL:
    """Test write_jsonl() function."""

    def test_write_single_row(self, tmp_path):
        """Test writing a single row."""
        output_file = tmp_path / "output.jsonl"
        rows = [{"id": 1, "text": "test"}]

        count = rss_collector.write_jsonl(output_file, rows)

        assert count == 1
        assert output_file.exists()
        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"id": 1, "text": "test"}

    def test_write_multiple_rows(self, tmp_path):
        """Test writing multiple rows."""
        output_file = tmp_path / "output.jsonl"
        rows = [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"},
            {"id": 3, "text": "third"}
        ]

        count = rss_collector.write_jsonl(output_file, rows)

        assert count == 3
        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_append_mode(self, tmp_path):
        """Test that write_jsonl appends to existing file."""
        output_file = tmp_path / "output.jsonl"
        rows1 = [{"id": 1}]
        rows2 = [{"id": 2}]

        rss_collector.write_jsonl(output_file, rows1)
        count = rss_collector.write_jsonl(output_file, rows2)

        assert count == 1
        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        output_file = tmp_path / "nested" / "deep" / "output.jsonl"
        rows = [{"id": 1}]

        rss_collector.write_jsonl(output_file, rows)

        assert output_file.exists()
        assert output_file.parent.exists()


class TestFetchFeed:
    """Test fetch_feed() function."""

    @patch("ingest.rss_collector.feedparser")
    def test_fetch_feed_sets_user_agent(self, mock_feedparser):
        """Test that user agent is set before parsing."""
        mock_feedparser.parse.return_value = {"entries": []}

        rss_collector.fetch_feed("https://example.com/feed.xml", timeout_sec=20, user_agent="Test-Agent/1.0")

        assert mock_feedparser.USER_AGENT == "Test-Agent/1.0"
        mock_feedparser.parse.assert_called_once_with("https://example.com/feed.xml")

    @patch("ingest.rss_collector.feedparser")
    def test_fetch_feed_returns_parsed(self, mock_feedparser):
        """Test that parsed feed is returned."""
        mock_result = {"entries": [{"title": "Test"}]}
        mock_feedparser.parse.return_value = mock_result

        result = rss_collector.fetch_feed("https://example.com/feed.xml", timeout_sec=20, user_agent="Test")

        assert result == mock_result


# ============================================================================
# Tests for normalize.py
# ============================================================================

class TestCleanText:
    """Test clean_text() function."""

    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        text = "  Hello   World  "
        result = normalize.clean_text(text)
        assert result == "Hello World"

    def test_nbsp_replacement(self):
        """Test that non-breaking spaces are replaced."""
        text = "Hello\u00a0World"
        result = normalize.clean_text(text)
        assert result == "Hello World"

    def test_multiple_spaces(self):
        """Test that multiple spaces are collapsed."""
        text = "Hello    World    Test"
        result = normalize.clean_text(text)
        assert result == "Hello World Test"

    def test_none_input(self):
        """Test that None input returns empty string."""
        result = normalize.clean_text(None)
        assert result == ""

    def test_empty_string(self):
        """Test that empty string returns empty string."""
        result = normalize.clean_text("")
        assert result == ""

    def test_newlines_and_tabs(self):
        """Test that newlines and tabs are normalized."""
        text = "Hello\n\tWorld"
        result = normalize.clean_text(text)
        assert result == "Hello World"


class TestParseTimeBestEffort:
    """Test parse_time_best_effort() function."""

    def test_iso_format(self):
        """Test parsing ISO format timestamp."""
        result = normalize.parse_time_best_effort("2024-01-01T12:00:00Z")
        assert result is not None
        assert result.startswith("2024-01-01T12:00:00")

    def test_rfc2822_format(self):
        """Test parsing RFC2822 format."""
        result = normalize.parse_time_best_effort("Mon, 01 Jan 2024 12:00:00 +0000")
        assert result is not None
        assert "2024-01-01" in result

    def test_naive_datetime(self):
        """Test that naive datetime gets UTC timezone."""
        result = normalize.parse_time_best_effort("2024-01-01 12:00:00")
        assert result is not None
        assert result.endswith("+00:00") or "Z" in result

    def test_none_input(self):
        """Test that None input returns None."""
        result = normalize.parse_time_best_effort(None)
        assert result is None

    def test_invalid_format(self):
        """Test that invalid format returns None."""
        result = normalize.parse_time_best_effort("not a date")
        assert result is None

    def test_empty_string(self):
        """Test that empty string returns None."""
        result = normalize.parse_time_best_effort("")
        assert result is None


class TestStableID:
    """Test stable_id() function."""

    def test_deterministic(self):
        """Test that same inputs produce same ID."""
        id1 = normalize.stable_id("source1", "url1", "title1", "2024-01-01")
        id2 = normalize.stable_id("source1", "url1", "title1", "2024-01-01")
        assert id1 == id2

    def test_different_inputs(self):
        """Test that different inputs produce different IDs."""
        id1 = normalize.stable_id("source1", "url1", "title1", "2024-01-01")
        id2 = normalize.stable_id("source2", "url1", "title1", "2024-01-01")
        assert id1 != id2

    def test_order_matters(self):
        """Test that order of arguments matters."""
        id1 = normalize.stable_id("a", "b", "c")
        id2 = normalize.stable_id("c", "b", "a")
        assert id1 != id2

    def test_sha256_format(self):
        """Test that ID is a valid SHA256 hex string."""
        doc_id = normalize.stable_id("test")
        assert len(doc_id) == 64  # SHA256 hex is 64 chars
        assert all(c in "0123456789abcdef" for c in doc_id)


class TestReadJSONL:
    """Test read_jsonl() function."""

    def test_read_single_line(self, tmp_path):
        """Test reading a single JSON line."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"id": 1, "text": "test"}\n')

        rows = list(normalize.read_jsonl(jsonl_file))

        assert len(rows) == 1
        assert rows[0] == {"id": 1, "text": "test"}

    def test_read_multiple_lines(self, tmp_path):
        """Test reading multiple JSON lines."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"id": 1}\n'
            '{"id": 2}\n'
            '{"id": 3}\n'
        )

        rows = list(normalize.read_jsonl(jsonl_file))

        assert len(rows) == 3
        assert rows[0]["id"] == 1
        assert rows[1]["id"] == 2
        assert rows[2]["id"] == 3

    def test_skip_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"id": 1}\n'
            '\n'
            '{"id": 2}\n'
            '   \n'
            '{"id": 3}\n'
        )

        rows = list(normalize.read_jsonl(jsonl_file))

        assert len(rows) == 3

    def test_empty_file(self, tmp_path):
        """Test reading empty file."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text("")

        rows = list(normalize.read_jsonl(jsonl_file))

        assert len(rows) == 0


class TestWriteJSONLNormalize:
    """Test write_jsonl() function from normalize module."""

    def test_write_single_row(self, tmp_path):
        """Test writing a single row."""
        output_file = tmp_path / "output.jsonl"
        rows = [{"id": 1, "text": "test"}]

        count = normalize.write_jsonl(output_file, rows)

        assert count == 1
        assert output_file.exists()
        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"id": 1, "text": "test"}

    def test_write_mode_overwrites(self, tmp_path):
        """Test that normalize.write_jsonl overwrites existing file."""
        output_file = tmp_path / "output.jsonl"
        rows1 = [{"id": 1}]
        rows2 = [{"id": 2}]

        normalize.write_jsonl(output_file, rows1)
        count = normalize.write_jsonl(output_file, rows2)

        assert count == 1
        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == 2


class TestNormalizeRow:
    """Test normalize_row() function."""

    def test_basic_normalization(self):
        """Test basic row normalization."""
        raw_row = {
            "ingested_at_utc": "2024-01-01T10:00:00Z",
            "source": {
                "id": "test_source",
                "label": "Test Source",
                "region": "test_region",
                "category": "test_category",
                "feed_url": "https://example.com/feed.xml"
            },
            "item": {
                "title": "Test Title",
                "link": "https://example.com/article",
                "published": "2024-01-01T12:00:00Z",
                "summary": "Test summary"
            }
        }

        result = normalize.normalize_row(raw_row)

        assert "id" in result
        assert result["source_id"] == "test_source"
        assert result["source_label"] == "Test Source"
        assert result["region"] == "test_region"
        assert result["category"] == "test_category"
        assert result["feed_url"] == "https://example.com/feed.xml"
        assert result["title"] == "Test Title"
        assert result["url"] == "https://example.com/article"
        assert result["body_text"] == "Test summary"
        assert result["published_at_utc"] is not None
        assert result["ingested_at_utc"] == "2024-01-01T10:00:00Z"
        assert "normalized_at_utc" in result
        assert "raw" in result
        assert result["raw"] == raw_row

    def test_missing_fields(self):
        """Test normalization with missing optional fields."""
        raw_row = {
            "source": {"id": "test"},
            "item": {"title": "Test"}
        }

        result = normalize.normalize_row(raw_row)

        assert result["source_id"] == "test"
        assert result["source_label"] is None
        assert result["region"] is None
        assert result["category"] is None
        assert result["title"] == "Test"
        assert result["url"] is None
        assert result["body_text"] == ""
        assert result["published_at_utc"] is None

    def test_stable_id_generation(self):
        """Test that stable ID is generated correctly."""
        raw_row = {
            "source": {"id": "source1"},
            "item": {
                "title": "Title",
                "link": "https://example.com",
                "published": "2024-01-01T12:00:00Z"
            }
        }

        result1 = normalize.normalize_row(raw_row)
        result2 = normalize.normalize_row(raw_row)

        # Same input should produce same ID
        assert result1["id"] == result2["id"]

    def test_text_cleaning(self):
        """Test that text fields are cleaned."""
        raw_row = {
            "source": {"id": "test"},
            "item": {
                "title": "  Title  with  spaces  ",
                "summary": "Summary\u00a0with\u00a0nbsp"
            }
        }

        result = normalize.normalize_row(raw_row)

        assert result["title"] == "Title with spaces"
        assert result["body_text"] == "Summary with nbsp"

    def test_empty_source_and_item(self):
        """Test handling of empty source and item dicts."""
        raw_row = {
            "source": {},
            "item": {}
        }

        result = normalize.normalize_row(raw_row)

        assert result["source_id"] is None
        assert result["title"] == ""
        assert "id" in result  # ID should still be generated


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full ingest pipeline."""

    def test_full_pipeline(self, tmp_path):
        """Test the full pipeline from raw to normalized."""
        # Create a raw JSONL file
        raw_file = tmp_path / "raw.jsonl"
        raw_data = {
            "ingested_at_utc": "2024-01-01T10:00:00Z",
            "source": {
                "id": "test_source",
                "label": "Test Source",
                "region": "test",
                "category": "test",
                "feed_url": "https://example.com/feed.xml"
            },
            "item": {
                "title": "Test Article",
                "link": "https://example.com/article",
                "published": "2024-01-01T12:00:00Z",
                "summary": "Article summary"
            }
        }
        raw_file.write_text(json.dumps(raw_data, ensure_ascii=False) + "\n")

        # Normalize it
        normalized_file = tmp_path / "normalized.jsonl"
        rows = list(normalize.read_jsonl(raw_file))
        normalized_rows = [normalize.normalize_row(r) for r in rows]
        normalize.write_jsonl(normalized_file, normalized_rows)

        # Verify output
        assert normalized_file.exists()
        output_rows = list(normalize.read_jsonl(normalized_file))
        assert len(output_rows) == 1
        assert output_rows[0]["source_id"] == "test_source"
        assert output_rows[0]["title"] == "Test Article"
        assert output_rows[0]["id"] is not None


# ============================================================================
# Fixtures
# ============================================================================
# Note: pytest provides tmp_path fixture automatically, no need to define it

