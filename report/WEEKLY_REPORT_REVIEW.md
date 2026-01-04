# Code Review: weekly_report.py

## Overall Assessment
**Grade: B+** - Well-structured, production-ready code with excellent documentation and type hints. Several improvements needed for robustness, error handling, and data quality.

## Strengths ✅

1. **Excellent Documentation**: Clear docstrings, inline comments, and module-level documentation
2. **Type Hints**: Comprehensive type annotations throughout
3. **Clean Architecture**: Well-organized functions with clear separation of concerns
4. **Deterministic Design**: Explicitly designed to be explainable and deterministic (as documented)
5. **Best Practices**: Uses dataclasses, pathlib, type hints, and modern Python patterns
6. **Clear Output Format**: Well-structured Markdown report generation

## Critical Issues 🔴

### 1. Duplicate Word in Sentiment Lexicons
**Location**: Lines 53 and 58
**Issue**: The word "decline" appears in both `POS_WORDS` (line 53) and `NEG_WORDS` (line 58)
**Impact**: Ambiguous sentiment classification - "decline" will be counted as both positive and negative, leading to incorrect sentiment metrics
**Recommendation**: Remove "decline" from `POS_WORDS` (it belongs in negative as "decrease" is already in positive):
```python
POS_WORDS = {
    "agree","aid","calm","ceasefire","cooperate","cooperation","deal","decrease","deescalate",
    "growth","improve","improvement","progress","recover","recovery","relief","rescue","stabilize",
    "support","truce",
}
```

### 2. No Error Handling for JSON Parsing
**Location**: `read_jsonl()` function (lines 97-103)
**Issue**: `json.loads(line)` can raise `json.JSONDecodeError` on malformed JSON lines
**Impact**: Script crashes on any malformed JSON line, making it fragile with real-world data
**Recommendation**: Add error handling with logging:
```python
def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(json.dumps({
                    "event": "json_parse_error",
                    "file": str(path),
                    "line": line_num,
                    "error": str(e)
                }), file=sys.stderr)
                continue
```

### 3. No Error Handling for File I/O Operations
**Location**: `load_docs_from_partitions()` (lines 106-124) and `main()` (line 436)
**Issue**: File operations can fail (permission errors, disk full, etc.) but exceptions are unhandled
**Impact**: Script crashes with cryptic errors instead of graceful failure
**Recommendation**: Add try/except blocks with structured error logging:
```python
def load_docs_from_partitions(base_dir: Path, start_day: datetime, end_day: datetime) -> List[Doc]:
    docs: List[Doc] = []
    cur = start_day
    while cur <= end_day:
        p = base_dir / day_to_partition(cur)
        if p.exists() and p.is_dir():
            for f in p.glob("*.jsonl"):
                try:
                    for row in read_jsonl(f):
                        # ... existing code ...
                except (OSError, PermissionError) as e:
                    print(json.dumps({
                        "event": "file_read_error",
                        "file": str(f),
                        "error": str(e)
                    }), file=sys.stderr)
                    continue
        cur += timedelta(days=1)
    return docs
```

## Medium Priority Issues 🟡

### 4. Missing Import for sys.stderr
**Location**: Throughout (error handling recommendations require `sys`)
**Issue**: `sys` is not imported, so structured error logging to stderr isn't possible without adding the import
**Recommendation**: Add `import sys` at the top

### 5. Memory Usage with Large Datasets
**Location**: `load_docs_from_partitions()` (line 107), `term_counts()` (line 134)
**Issue**: All documents are loaded into memory as a list, then processed. For very large date ranges, this could cause memory issues
**Impact**: Potential memory exhaustion with large datasets
**Recommendation**: Consider streaming/chunked processing for very large windows, or document memory requirements:
```python
# Add to docstring:
# Memory Usage: This script loads all documents for the window into memory.
# For windows > 30 days or high-volume sources, consider chunked processing.
```

### 6. No Validation for Empty Results
**Location**: `main()` function (lines 418-426)
**Issue**: No warning if `docs_current` is empty (no data for the window)
**Impact**: Report will be generated with zeros/empty tables, which might be misleading
**Recommendation**: Add a warning:
```python
docs_current = load_docs_from_partitions(base_dir, start_day, end_day)
if not docs_current:
    print(json.dumps({
        "event": "warning",
        "message": "No documents found for current window",
        "window": f"{start_day} to {end_day}"
    }), file=sys.stderr)
```

### 7. Baseline Window Logic Could Be Clearer
**Location**: `main()` function (lines 420-425)
**Issue**: The condition `if baseline_end >= baseline_start` is correct but the logic could be more explicit
**Impact**: Minor - code works but could be more readable
**Recommendation**: The current logic is fine, but consider a comment:
```python
docs_baseline: List[Doc] = []
if args.baseline_days > 0:
    baseline_end = start_day - timedelta(days=1)
    baseline_start = start_day - timedelta(days=args.baseline_days)
    # Only load baseline if we have a valid date range (baseline_days > 0)
    if baseline_end >= baseline_start:
        docs_baseline = load_docs_from_partitions(base_dir, baseline_start, baseline_end)
```

### 8. Token Length Filtering Inconsistency
**Location**: `tokenize()` function (line 130)
**Issue**: Regex allows tokens of length >= 2 (`[a-zA-Z\-]{1,}` = 1+ chars after first), but then filters to `len(t) >= 3`. The regex could be optimized
**Impact**: Minor performance issue - regex matches tokens that are then discarded
**Recommendation**: Either adjust regex to `[a-zA-Z\-]{2,}` (for min length 3 total) or document why the two-step filter exists

### 9. No Validation of Date Arguments
**Location**: `main()` function (line 411)
**Issue**: No validation that `--end-date` is not in the future (if that's desired), or that dates are reasonable
**Impact**: Could generate reports for future dates (probably fine, but worth considering)
**Recommendation**: Add optional validation or document behavior:
```python
end_day = parse_ymd(args.end_date) if args.end_date else utc_today()
if end_day > utc_today():
    print(json.dumps({
        "event": "warning",
        "message": "end_date is in the future",
        "end_date": args.end_date
    }), file=sys.stderr)
```

## Minor Issues / Suggestions 🟢

### 10. Magic Numbers
**Location**: Various thresholds (e.g., `min_current: int = 8`, `min_baseline: int = 12`)
**Issue**: Hard-coded thresholds scattered throughout
**Recommendation**: Consider making these configurable constants or command-line arguments:
```python
DRIFT_MIN_CURRENT = 8
DRIFT_TOP_N = 25
SILENCE_MIN_BASELINE = 12
SILENCE_TOP_N = 15
TOP_TERMS_COUNT = 30
```

### 11. Output Directory Permissions
**Location**: `main()` function (line 416)
**Issue**: `mkdir(parents=True, exist_ok=True)` can fail on permission errors, but error isn't caught
**Impact**: Script crashes with cryptic error
**Recommendation**: Handle `PermissionError` explicitly

### 12. Sentiment Lexicon Size
**Location**: Lines 52-61
**Issue**: Very small lexicons (20 positive, 20 negative words) - may not be representative
**Impact**: Sentiment index may not be accurate for many domains
**Recommendation**: Document this as a known limitation (already somewhat done in docstrings). Consider expanding lexicons or making them configurable.

### 13. Hyphen Handling in Tokens
**Location**: `TOKEN_RE` regex (line 64)
**Issue**: Allows hyphens in tokens (e.g., "well-known" becomes one token)
**Impact**: May or may not be desired behavior
**Recommendation**: Document this choice, or consider splitting on hyphens if compound words should be separate tokens

### 14. Return Code Consistency
**Location**: `main()` function
**Issue**: Returns 0 on success, but exceptions (if unhandled) would exit with code 1
**Recommendation**: Consider consistent error codes (0=success, 1=runtime error, 2=usage error) - though current behavior is acceptable

### 15. Missing Dependency Check
**Location**: Module level
**Issue**: No validation that required directories exist or are accessible
**Recommendation**: Consider early validation in `main()`:
```python
base_dir = Path(args.normalized_dir)
if not base_dir.exists():
    print(json.dumps({
        "event": "error",
        "message": "normalized_dir does not exist",
        "path": str(base_dir)
    }), file=sys.stderr)
    return 1
```

## Performance Considerations

- ✅ Good: Efficient use of Counter for term counting
- ✅ Good: Single pass through documents for term extraction
- 💡 Consider: For very large datasets, streaming/chunked processing might be needed
- 💡 Consider: The `silence_terms()` function iterates through all baseline terms - for very large vocabularies, this could be optimized with a set intersection

## Security Considerations

- ✅ Good: No eval() or unsafe operations
- ✅ Good: Path handling uses pathlib (helps prevent path traversal)
- 💡 Consider: Validate that input paths are within expected directories (if security is a concern)

## Testing Recommendations

1. Test with malformed JSON lines (should handle gracefully)
2. Test with empty partitions (no JSONL files)
3. Test with empty JSONL files
4. Test with missing required fields in JSON (should use defaults)
5. Test with very large date ranges (memory usage)
6. Test sentiment calculation with known inputs
7. Test drift calculation edge cases (empty baseline, no overlap, etc.)
8. Test with baseline_days=0 (baseline disabled)
9. Test with future end dates
10. Test with permission errors on input/output directories

## Suggested Priority Fixes

1. **High**: Fix duplicate "decline" in sentiment lexicons (#1)
2. **High**: Add error handling for JSON parsing (#2)
3. **High**: Add error handling for file I/O (#3)
4. **Medium**: Add sys import and structured error logging (#4)
5. **Medium**: Add warning for empty document sets (#6)
6. **Medium**: Document memory requirements or add chunked processing (#5)
7. **Low**: Extract magic numbers to constants (#10)
8. **Low**: Add input validation for directories (#15)

## Code Quality Notes

- Excellent type hints and documentation
- Clean function signatures and separation of concerns
- Good use of dataclasses and modern Python features
- Deterministic and explainable design (as intended)
- The code follows good practices overall, with the main gaps being error handling and the sentiment lexicon bug

