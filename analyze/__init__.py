"""Top-level package for the analyze tools.

This module exposes the main analyzer entry points so callers can import
them from `analyze` (e.g. `from analyze import frequency_drift`). It also
provides a small `cli()` helper to dispatch subcommands when invoked as a
module (useful for tests or programmatic invocation).

The individual analyzer modules are kept as standalone scripts with their
own `main()` functions; we avoid importing heavy dependencies at package
import time.
"""

from importlib import import_module
from typing import Dict


def _load_analyzers() -> Dict[str, object]:
    """Lazily import analyzer modules and return a mapping name->module."""

    mods: Dict[str, object] = {}
    for name in (
        "frequency_drift",
        "sentiment_shift",
        "silence_detection",
        "topic_convergence",
    ):
        try:
            mods[name] = import_module(f".{name}", __package__)
        except Exception:
            # If an analyzer is missing or has import errors, skip it to
            # keep the package import lightweight.
            continue

    return mods


def analyzers() -> Dict[str, object]:
    """Return available analyzer modules."""

    return _load_analyzers()


def cli(argv=None) -> int:
    """Simple CLI dispatcher: `analyze.cli(['frequency_drift', '--help'])`.

    Returns the exit code produced by the selected analyzer's `main()`.
    """

    import sys

    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv:
        print("Usage: analyze <analyzer> [args...]")
        print("Available analyzers:")
        for k in sorted(analyzers().keys()):
            print(" -", k)
        return 2

    name = argv[0]
    mods = analyzers()
    mod = mods.get(name)
    if mod is None:
        print(f"Unknown analyzer: {name}")
        return 2

    # Call the module's main(), passing the rest of argv via sys.argv
    sys_argv = [name] + argv[1:]
    old_argv = sys.argv
    try:
        sys.argv = sys_argv
        return int(mod.main())
    finally:
        sys.argv = old_argv


__all__ = ["analyzers", "cli"]
