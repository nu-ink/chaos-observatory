# Test Review: test_ingest.py

## Overall Assessment
The test suite provides good coverage of core functionality with clear organization and descriptive test names. However, there are several critical issues and gaps that need attention.

## ✅ Strengths

1. **Well-organized**: Clear test classes grouped by functionality
2. **Good coverage**: Most core functions are tested
3. **Effective use of fixtures**: Proper use of `tmp_path` for file operations
4. **Descriptive names**: Test names clearly describe what they're testing
5. **Edge cases**: Some edge cases are covered (empty files, missing fields, etc.)

## ❌ Critical Issues

### 1. **Data Structure Mismatch** ⚠️ CRITICAL

**Problem**: The tests for `normalize_row()` use flat data structures, but `rss_collector.py` actually produces nested structures.

**Actual output from rss_collector** (lines 185-196):
```python
{
    "ingested_at_utc": "...",
    "source": {
        "id": src.id,
        "label": src.label,
        "region": src.region,
        "category": src.category,
        "feed_url": feed.url,
    },
    "item": {
        "title": ...,
        "link": ...,
        "published": ...,
        "summary": ...,
        "raw": ...
    }
}
```

**What normalize_row() expects** (based on tests):
```python
{
    "source_id": "...",
    "title": "...",
    "link": "...",
    "published": "...",
    "summary": "..."
}
```

**Impact**: The integration test and all `normalize_row()` tests may not reflect real-world usage.

**Recommendation**: 
- Either add a transformation step between rss_collector and normalize
- Or update `normalize_row()` to handle nested structures
- Or update tests to match actual data flow

### 2. **Missing Error Handling Tests**

No tests for:
- Invalid YAML files
- Missing required fields in YAML (e.g., missing `id`)
- File not found errors
- Invalid JSON in JSONL files
- Network errors in `fetch_feed()`
- Timeout handling in `fetch_feed()`

### 3. **Incomplete `fetch_feed()` Testing**

Current test only mocks `feedparser.parse()` but doesn't:
- Test timeout behavior (timeout parameter is accepted but not tested)
- Test error handling
- Verify that timeout is actually used

### 4. **Missing Function Coverage**

Not tested:
- `normalize_file()` - processes entire files/directories
- `main()` functions for both modules
- `_utc_now_iso()` helper function

### 5. **Integration Test Doesn't Match Reality**

The integration test (line 544) creates flat data, but the actual pipeline would have nested data from `rss_collector`. This test may pass but doesn't validate the real data flow.

## 🔍 Specific Test Issues

### TestNormalizeRow
- Tests assume flat input structure, but real data is nested
- Missing test for nested structure from rss_collector output
- Missing test for `item.title`, `item.link`, etc. access patterns

### TestFetchFeed
- `test_fetch_feed_sets_user_agent`: Good, but should also verify timeout handling
- Missing: Test for feedparser errors/exceptions
- Missing: Test for bozo parsing warnings

### TestLoadSourcesYAML
- Missing: Test for invalid YAML syntax
- Missing: Test for missing `id` field (required)
- Missing: Test for empty `feeds` list
- Missing: Test for invalid feed URL format

### TestWriteJSONL (rss_collector)
- Good coverage, but missing:
  - Test for encoding issues (non-ASCII characters)
  - Test for very large files

### TestReadJSONL
- Missing: Test for invalid JSON lines (should handle gracefully or fail?)
- Missing: Test for encoding issues
- Missing: Test for very large files

## 📋 Recommendations

### High Priority
1. **Fix data structure mismatch**: Determine correct data flow and update tests/code accordingly
2. **Add error handling tests**: Test invalid inputs, missing files, network errors
3. **Test `normalize_file()`**: This is a key function that processes files/directories
4. **Update integration test**: Make it reflect actual data structures

### Medium Priority
5. **Add tests for `main()` functions**: At least basic smoke tests
6. **Test timeout behavior**: Verify `fetch_feed()` timeout parameter works
7. **Add edge case tests**: Empty feeds, malformed entries, etc.

### Low Priority
8. **Test encoding handling**: Non-ASCII characters in various fields
9. **Performance tests**: Large files, many feeds
10. **Test `_utc_now_iso()`**: If it's used in critical paths

## 🧪 Suggested Additional Tests

```python
# Error handling
def test_load_sources_yaml_invalid_syntax(tmp_path):
    """Test handling of invalid YAML syntax."""
    
def test_load_sources_yaml_missing_id(tmp_path):
    """Test handling of missing required 'id' field."""
    
def test_read_jsonl_invalid_json(tmp_path):
    """Test handling of invalid JSON lines."""
    
def test_fetch_feed_timeout():
    """Test that timeout parameter is respected."""
    
def test_fetch_feed_network_error():
    """Test handling of network errors."""
    
def test_normalize_file_single_file(tmp_path):
    """Test normalize_file() with single file."""
    
def test_normalize_file_directory(tmp_path):
    """Test normalize_file() with directory."""
    
def test_normalize_row_nested_structure():
    """Test normalize_row() with actual rss_collector output structure."""
```

## 📊 Coverage Estimate

- **Functions tested**: ~70%
- **Error paths tested**: ~10%
- **Edge cases tested**: ~40%
- **Integration scenarios**: ~30%

## 🎯 Priority Actions

1. **Immediate**: Resolve data structure mismatch
2. **Short-term**: Add error handling tests
3. **Medium-term**: Add missing function tests
4. **Long-term**: Improve edge case coverage

