# Code Review: rss_collector.py

## Overall Assessment
**Grade: B+** - Solid, production-ready code with good practices, but several improvements would make it more robust.

## Strengths ✅

1. **Clean Architecture**: Well-structured with clear separation of concerns
2. **Type Hints**: Comprehensive type annotations improve maintainability
3. **Error Handling**: Good JSON-structured logging for errors and warnings
4. **Documentation**: Clear docstrings and comments
5. **Best Practices**: Uses dataclasses, pathlib, context managers

## Critical Issues 🔴

### 1. Timeout Parameter Not Actually Used
**Location**: `fetch_feed()` function (line 104-110)
**Issue**: `timeout_sec` parameter is accepted but never used. `feedparser.parse()` doesn't support timeout.
**Impact**: No protection against hanging requests
**Recommendation**: Use `requests` library with timeout:
```python
import requests

def fetch_feed(url: str, timeout_sec: int, user_agent: str) -> feedparser.FeedParserDict:
    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers, timeout=timeout_sec)
    response.raise_for_status()
    return feedparser.parse(response.content)
```

### 2. Missing Input Validation
**Location**: `load_sources_yaml()` function (line 47-72)
**Issue**: No validation that required fields like `id` exist in source entries
**Impact**: Could crash with cryptic `KeyError` at runtime
**Recommendation**: Add validation:
```python
for s in sources_raw:
    if "id" not in s:
        print(f"WARNING: Skipping source missing 'id' field: {s}", file=sys.stderr)
        continue
    # ... rest of code
```

## Medium Priority Issues 🟡

### 3. Unused Function
**Location**: `safe_get()` function (line 75-81)
**Issue**: Function is defined but never called
**Recommendation**: Remove if not needed, or use it to improve code safety

### 4. Global State Modification
**Location**: Line 107 (`feedparser.USER_AGENT = user_agent`)
**Issue**: Modifies global state, could cause issues in concurrent scenarios
**Recommendation**: Use requests library approach (see issue #1) or accept this limitation with a comment

### 5. Append Mode Behavior
**Location**: `write_jsonl()` function (line 113-120)
**Issue**: Uses append mode (`"a"`), so multiple runs in same day create duplicates
**Impact**: Data duplication if script runs multiple times per day
**Recommendation**: 
- Document this behavior in docstring, OR
- Add deduplication logic, OR
- Use write mode (`"w"`) if overwriting is acceptable

### 6. Empty Feeds List
**Location**: Main loop (line 160)
**Issue**: No check if `src.feeds` is empty
**Recommendation**: Add check:
```python
if not src.feeds:
    print(json.dumps({
        "event": "source_skipped",
        "source_id": src.id,
        "reason": "no feeds configured",
        "ts_utc": _utc_now_iso(),
    }), file=sys.stderr)
    continue
```

### 7. FeedParserDict to dict Conversion
**Location**: `normalize_entry_minimal()` line 100
**Issue**: `dict(entry)` conversion may not preserve all attributes
**Impact**: Potential data loss in raw field
**Recommendation**: Test thoroughly or use `vars(entry)` or explicit field extraction

## Minor Issues / Suggestions 🟢

### 8. Error Context
Consider including more context in error messages (e.g., which source/feed failed)

### 9. Return Code Consistency
Error codes: 2 for file not found, 0 for success. Consider 1 for runtime errors (currently exceptions go to stderr but return 0 implicitly if they don't crash)

### 10. Dependencies
Consider pinning dependency versions in requirements.txt (e.g., `feedparser==6.0.0`)

### 11. JSON Serialization
The `dict(entry)` on line 100 might fail if entry contains non-serializable objects. Consider error handling or explicit field extraction.

## Performance Considerations

- ✅ Good: Sleep between requests to avoid hammering servers
- ✅ Good: Limiting max items per feed
- 💡 Consider: Adding retry logic with exponential backoff for transient failures
- 💡 Consider: Parallel fetching (with rate limiting) for better performance with many feeds

## Security Considerations

- ✅ Good: User-Agent is configurable
- ✅ Good: No eval() or unsafe operations
- 💡 Consider: Validate URLs before fetching (basic sanity check)
- 💡 Consider: Rate limiting to avoid being blocked

## Testing Recommendations

1. Test with malformed YAML
2. Test with missing required fields
3. Test with empty feeds list
4. Test with invalid URLs
5. Test timeout behavior (if implemented)
6. Test append mode behavior (multiple runs)

## Suggested Priority Fixes

1. **High**: Implement actual timeout (#1)
2. **High**: Add input validation (#2)
3. **Medium**: Remove unused function or use it (#3)
4. **Medium**: Document or fix append mode behavior (#5)
5. **Low**: Handle empty feeds list (#6)

