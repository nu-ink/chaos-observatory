# utils/lock.py
"""
File locking utility for pipeline execution.

Prevents concurrent pipeline runs by using a file lock.
"""
import sys
import logging
from pathlib import Path
from filelock import FileLock, Timeout

log = logging.getLogger(__name__)

# Default lock timeout: 10 minutes (pipeline should complete well before this)
DEFAULT_LOCK_TIMEOUT = 600  # seconds

# Lock file location (can be overridden)
DEFAULT_LOCK_FILE = Path("/tmp/chaos_observatory.lock")


def run_with_lock(lock_file: Path = None, timeout: int = None):
    """
    Run the pipeline with file locking to prevent concurrent execution.
    
    Args:
        lock_file: Path to lock file (defaults to /tmp/chaos_observatory.lock)
        timeout: Lock acquisition timeout in seconds (defaults to 600)
    
    Returns:
        Exit code from pipeline execution
    """
    if lock_file is None:
        lock_file = DEFAULT_LOCK_FILE
    if timeout is None:
        timeout = DEFAULT_LOCK_TIMEOUT
    
    lock = FileLock(lock_file, timeout=timeout)
    
    try:
        with lock:
            log.info(f"Acquired lock: {lock_file}")
            from run_pipeline import main as run_pipeline_main
            return run_pipeline_main()
    except Timeout:
        log.error(f"Could not acquire lock {lock_file} within {timeout}s. Another instance may be running.")
        return 1
    except Exception as e:
        log.exception(f"Error during locked pipeline execution: {e}")
        return 1


if __name__ == "__main__":
    # Allow running as script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sys.exit(run_with_lock())
