#!/usr/bin/env python3
"""
Tests for Chaos-Observatory ingest module.

Tests both rss_collector.py and normalize.py functionality.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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

    @patch.object(rss_collector, "feedparser")
    def test_fetch_feed_sets_user_agent(self, mock_feedparser):
        """Test that user agent is set before parsing."""
        mock_feedparser.parse.return_value = {"entries": []}

        rss_collector.fetch_feed("https://example.com/feed.xml", timeout_sec=20, user_agent="Test-Agent/1.0")

        assert mock_feedparser.USER_AGENT == "Test-Agent/1.0"
        mock_feedparser.parse.assert_called_once_with("https://example.com/feed.xml")

    @patch.object(rss_collector, "feedparser")
    def test_fetch_feed_returns_parsed(self, mock_feedparser):
        """Test that parsed feed is returned."""
        mock_result = {"entries": [{"title": "Test"}]}
        mock_feedparser.parse.return_value = mock_result

        result = rss_collector.fetch_feed("https://example.com/feed.xml", timeout_sec=20, user_agent="Test")

        assert result == mock_result


# ============================================================================
# Tests for normalize.py
# ============================================================================


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
            "source_id": "test_source",
            "title": "Test Title",
            "link": "https://example.com/article",
            "published": "2024-01-01T12:00:00Z",
            "summary": "Test summary"
        }

        result = normalize.normalize_row(raw_row)

        assert result["source_id"] == "test_source"
        assert result["title"] == "Test Title"
        assert result["url"] == "https://example.com/article"
        assert result["published_ts"] == "2024-01-01T12:00:00Z"
        assert result["summary"] == "Test summary"
        assert result["text"] == "Test summary"

    def test_url_fallback(self):
        """Test that 'url' field is used as fallback for 'link'."""
        raw_row = {
            "source_id": "test",
            "title": "Test",
            "url": "https://example.com/article"
        }

        result = normalize.normalize_row(raw_row)

        assert result["url"] == "https://example.com/article"

    def test_link_precedence_over_url(self):
        """Test that 'link' takes precedence over 'url'."""
        raw_row = {
            "source_id": "test",
            "title": "Test",
            "link": "https://example.com/link",
            "url": "https://example.com/url"
        }

        result = normalize.normalize_row(raw_row)

        assert result["url"] == "https://example.com/link"

    def test_description_fallback(self):
        """Test that 'description' field is used as fallback for 'summary'."""
        raw_row = {
            "source_id": "test",
            "title": "Test",
            "description": "Description text"
        }

        result = normalize.normalize_row(raw_row)

        assert result["summary"] == "Description text"
        assert result["text"] == "Description text"

    def test_content_precedence_over_summary(self):
        """Test that 'content' takes precedence over 'summary' for 'text' field."""
        raw_row = {
            "source_id": "test",
            "title": "Test",
            "content": "Content text",
            "summary": "Summary text"
        }

        result = normalize.normalize_row(raw_row)

        assert result["text"] == "Content text"
        assert result["summary"] == "Summary text"

    def test_missing_fields(self):
        """Test normalization with missing optional fields."""
        raw_row = {
            "title": "Test"
        }

        result = normalize.normalize_row(raw_row)

        assert result["source_id"] is None
        assert result["title"] == "Test"
        assert result["url"] is None
        assert result["published_ts"] is None
        assert result["summary"] is None
        assert result["text"] is None

    def test_empty_dict(self):
        """Test handling of empty dict."""
        raw_row = {}

        result = normalize.normalize_row(raw_row)

        assert result["source_id"] is None
        assert result["title"] is None
        assert result["url"] is None
        assert result["published_ts"] is None
        assert result["summary"] is None
        assert result["text"] is None


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full ingest pipeline."""

    def test_full_pipeline(self, tmp_path):
        """Test the full pipeline from raw to normalized."""
        # Create a raw JSONL file (flat structure as expected by normalize_row)
        raw_file = tmp_path / "raw.jsonl"
        raw_data = {
            "source_id": "test_source",
            "title": "Test Article",
            "link": "https://example.com/article",
            "published": "2024-01-01T12:00:00Z",
            "summary": "Article summary"
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
        assert output_rows[0]["url"] == "https://example.com/article"
        assert output_rows[0]["published_ts"] == "2024-01-01T12:00:00Z"
        assert output_rows[0]["summary"] == "Article summary"
        assert output_rows[0]["text"] == "Article summary"


# ============================================================================
# Fixtures
# ============================================================================
# Note: pytest provides tmp_path fixture automatically, no need to define it

