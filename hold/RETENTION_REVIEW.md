# Code Review: retention.py

## Overall Assessment
**Grade: A-** - Well-structured, production-ready code with excellent safety features. A few edge cases and validations could be improved.

## Strengths ✅

1. **Excellent Safety Design**: Default dry-run mode with explicit `--apply` flag
2. **Clear Architecture**: Well-separated functions with single responsibilities
3. **Comprehensive Type Hints**: Full type annotations throughout
4. **Structured Logging**: JSON-structured event emission for observability
5. **Good Documentation**: Clear docstrings and module-level documentation
6. **Proper Date Handling**: Uses UTC consistently
7. **Safety Validation**: Checks `hold_days <= keep_days` relationship

## Critical Issues 🔴

### 1. Future Date Partitions Not Handled
**Location**: `compute_age_days()` (line 65-68) and `plan_actions_for_base()` (line 89-154)
**Issue**: If a partition date is in the future (e.g., due to clock skew or manual creation), `age_days` will be negative. The logic `if age <= hold_days` will incorrectly classify future partitions as within the hold window.
**Impact**: Future partitions could be protected when they shouldn't be, or could cause confusion
**Recommendation**: Add explicit handling:
```python
def compute_age_days(partition_date: datetime, today: datetime) -> int:
    delta = today - partition_date
    age = int(delta.days)
    # Handle future dates explicitly
    if age < 0:
        # Future partition - treat as very old (will be kept/archived based on policy)
        # Or raise an error/warning
        return -1  # or handle differently
    return age
```

Or add a check in `plan_actions_for_base`:
```python
age = compute_age_days(pdt, today)
if age < 0:
    actions.append(
        PartitionAction(
            base=base_label,
            partition=name,
            path=path,
            age_days=age,
            action="skip_hold",  # or "keep" or raise error
            reason=f"future partition (age_days={age} < 0)",
        )
    )
    continue
```

### 2. Missing Archive Directory Validation
**Location**: `main()` (line 207) and `move_partition_to_archive()` (line 161-178)
**Issue**: No validation that `archive_dir` is not the same as or a subdirectory of `raw_dir` or `normalized_dir`. This could cause data loss or infinite loops.
**Impact**: Could accidentally move data into itself, causing corruption or loss
**Recommendation**: Add validation:
```python
if archive_dir:
    archive_path = Path(archive_dir)
    # Check for circular references
    if archive_path.resolve() == raw_dir.resolve() or archive_path.resolve() == norm_dir.resolve():
        raise SystemExit("ERROR: archive_dir cannot be the same as raw_dir or normalized_dir")
    # Check if archive_dir is a subdirectory
    try:
        raw_dir.resolve().relative_to(archive_path.resolve())
        raise SystemExit("ERROR: archive_dir cannot be a parent of raw_dir or normalized_dir")
    except ValueError:
        pass  # Good, not a subdirectory
    try:
        norm_dir.resolve().relative_to(archive_path.resolve())
        raise SystemExit("ERROR: archive_dir cannot be a parent of raw_dir or normalized_dir")
    except ValueError:
        pass
```

## Medium Priority Issues 🟡

### 3. No Input Directory Validation
**Location**: `main()` (lines 205-207)
**Issue**: No check that `raw_dir` and `normalized_dir` are actually directories (not files). If they're files, `list_partitions()` will return empty, but no error is raised.
**Impact**: Silent failure - script runs but does nothing
**Recommendation**: Add validation:
```python
if raw_dir.exists() and not raw_dir.is_dir():
    raise SystemExit(f"ERROR: raw_dir exists but is not a directory: {raw_dir}")
if norm_dir.exists() and not norm_dir.is_dir():
    raise SystemExit(f"ERROR: normalized_dir exists but is not a directory: {norm_dir}")
```

### 4. Race Condition in Archive Collision Handling
**Location**: `move_partition_to_archive()` (lines 173-175)
**Issue**: Between checking `dest.exists()` and creating the new name, another process could create the same timestamped directory. Very unlikely but possible.
**Impact**: Potential `FileExistsError` or data loss in concurrent scenarios
**Recommendation**: Use a loop with retry or use `tempfile` for atomic operations:
```python
import tempfile

def move_partition_to_archive(
    action: PartitionAction,
    archive_root: Path,
) -> Path:
    base_dest = archive_root / action.base / action.partition
    ensure_dir(base_dest.parent)
    
    # Atomic move with collision handling
    if base_dest.exists():
        # Use a more unique suffix (timestamp + random)
        import random
        suffix = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{random.randint(1000, 9999)}"
        dest = archive_root / action.base / f"{action.partition}__{suffix}"
    else:
        dest = base_dest
    
    # Retry logic for race conditions
    max_retries = 5
    for attempt in range(max_retries):
        try:
            shutil.move(str(action.path), str(dest))
            return dest
        except FileExistsError:
            if attempt < max_retries - 1:
                # Generate new name and retry
                suffix = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{random.randint(1000, 9999)}"
                dest = archive_root / action.base / f"{action.partition}__{suffix}"
            else:
                raise
```

### 5. Empty Partition Directories
**Location**: `list_partitions()` (line 71-86)
**Issue**: No check if partition directories are empty. Empty directories will still be archived/deleted.
**Impact**: Probably fine, but might want to skip empty partitions or log them separately
**Recommendation**: Consider adding a check (if empty partitions should be handled differently):
```python
def list_partitions(base_dir: Path) -> List[Tuple[str, Path, datetime]]:
    # ... existing code ...
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        # Optional: skip empty directories
        # if not any(child.iterdir()):
        #     continue
        pdt = parse_partition_date(child.name)
        if not pdt:
            continue
        parts.append((child.name, child, pdt))
    return parts
```

### 6. Error Handling Could Be More Specific
**Location**: Exception handling (lines 307-318)
**Issue**: Catches all `Exception` types, which could hide programming errors
**Impact**: Bugs might be masked as "partition errors"
**Recommendation**: Catch more specific exceptions:
```python
except (OSError, shutil.Error, PermissionError) as ex:
    # File system errors
    errors += 1
    emit(...)
except Exception as ex:
    # Unexpected errors - re-raise or log differently
    errors += 1
    emit(
        "partition_error_unexpected",
        {
            "base": a.base,
            "partition": a.partition,
            "path": str(a.path),
            "action": a.action,
            "error": repr(ex),
            "error_type": type(ex).__name__,
        },
    )
```

## Minor Issues / Suggestions 🟢

### 7. Return Code on Validation Errors
**Location**: `main()` (lines 200-203)
**Issue**: Uses `raise SystemExit()` for validation errors, which exits with code 1. This is fine, but could be more explicit.
**Recommendation**: Consider using `argparse`'s built-in error handling or be explicit:
```python
if args.hold_days < 0 or args.keep_days < 0:
    ap.error("hold-days and keep-days must be >= 0")
if args.keep_days < args.hold_days:
    ap.error("keep-days must be >= hold-days")
```

### 8. Missing Type for `counts` Dictionary
**Location**: Line 253
**Issue**: `counts: Dict[str, int]` is fine, but could use `Counter` from collections for cleaner code
**Recommendation**: 
```python
from collections import Counter

counts = Counter(a.action for a in planned)
```

### 9. Potential Issue with `ensure_dir` on Archive
**Location**: `move_partition_to_archive()` (line 170)
**Issue**: `ensure_dir(dest.parent)` is called, but if `archive_root` itself doesn't exist, this will create it. Might want to validate `archive_root` exists earlier.
**Recommendation**: Validate in `main()`:
```python
if archive_dir:
    archive_path = Path(archive_dir)
    if not archive_path.exists():
        # Option 1: Create it
        ensure_dir(archive_path)
        # Option 2: Require it to exist
        # raise SystemExit(f"ERROR: archive_dir does not exist: {archive_dir}")
```

### 10. No Progress Indication for Large Operations
**Location**: Execution loop (lines 264-318)
**Issue**: For large numbers of partitions, no progress indication
**Recommendation**: Consider emitting progress events periodically:
```python
total = len([a for a in planned if a.action in ("archive", "delete")])
processed = 0
for a in planned:
    # ... existing code ...
    if a.action in ("archive", "delete") and args.apply:
        processed += 1
        if processed % 10 == 0 or processed == total:
            emit("progress", {"processed": processed, "total": total})
```

### 11. Documentation Could Mention Edge Cases
**Location**: Module docstring (lines 1-23)
**Suggestion**: Add notes about:
- Future partitions behavior
- Empty directories
- Concurrent execution safety

## Performance Considerations

- ✅ Good: Uses `Path` objects (efficient)
- ✅ Good: Sorted iteration for predictable ordering
- 💡 Consider: For very large directory trees, `iterdir()` could be slow. Consider using `os.scandir()` for better performance.
- 💡 Consider: Batch operations if moving many partitions

## Security Considerations

- ✅ Good: Requires explicit `--apply` flag
- ✅ Good: No eval() or unsafe operations
- ✅ Good: Uses `shutil` for safe file operations
- 💡 Consider: Validate that paths are within expected boundaries (prevent directory traversal if paths come from untrusted input, though this seems to be internal tooling)

## Testing Recommendations

1. Test with future-dated partitions
2. Test with `archive_dir` as subdirectory of source dirs
3. Test with non-existent directories
4. Test with empty partition directories
5. Test concurrent execution (if applicable)
6. Test with `hold_days == keep_days` (edge case)
7. Test with `hold_days == 0` and `keep_days == 0`
8. Test archive collision scenarios
9. Test with read-only directories (permission errors)
10. Test with very large numbers of partitions

## Suggested Priority Fixes

1. **High**: Handle future-dated partitions (#1)
2. **High**: Validate archive_dir is not circular (#2)
3. **Medium**: Add input directory validation (#3)
4. **Medium**: Improve error handling specificity (#6)
5. **Low**: Add progress indication (#10)
6. **Low**: Use Counter for counts (#8)

## Code Quality Notes

- Excellent use of dataclasses
- Good separation of concerns
- Clear function names
- Consistent error handling pattern
- Well-structured event emission

Overall, this is very solid code with good practices. The main concerns are edge case handling (future dates, circular archive paths) and some defensive validations that would make it more robust in production environments.

