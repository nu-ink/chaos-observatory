#!/usr/bin/env python3
"""
Chaos Observatory pipeline runner

Runs a single full cycle:
- ingest
- normalize
- analyze
- report

Designed to be executed by systemd timer every 15 minutes.
"""

import sys
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

# --- basic logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("chaos-pipeline")


def utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def run_ingest():
    """Run RSS collection step."""
    log.info("Starting ingest")
    from ingest.rss_collector import main as ingest_main

    start_time = time.time()
    exit_code = ingest_main(
        ["--sources", "ingest/sources.yaml", "--outdir", "data/raw"]
    )
    elapsed = time.time() - start_time

    if exit_code != 0:
        log.error(f"Ingest failed with exit code {exit_code} (took {elapsed:.2f}s)")
        raise RuntimeError(f"Ingest step failed with exit code {exit_code}")

    log.info(f"Ingest completed successfully (took {elapsed:.2f}s)")
    return exit_code


def run_normalize(day: str):
    """Run normalization step."""
    log.info("Starting normalize")
    from ingest.normalize import main as normalize_main

    start_time = time.time()
    raw_dir = Path("data/raw") / day
    normalized_dir = Path("data/normalized") / day
    normalized_dir.mkdir(parents=True, exist_ok=True)
    exit_code = normalize_main(["--in", str(raw_dir), "--out", str(normalized_dir)])
    elapsed = time.time() - start_time

    if exit_code != 0:
        log.error(f"Normalize failed with exit code {exit_code} (took {elapsed:.2f}s)")
        raise RuntimeError(f"Normalize step failed with exit code {exit_code}")

    log.info(f"Normalize completed successfully (took {elapsed:.2f}s)")
    return exit_code


def run_analysis(day: str):
    """Run all analysis steps."""
    log.info("Starting analysis")

    # Import all analysis modules
    from analyze.frequency_drift import main as freq_main
    from analyze.topic_convergence import main as conv_main
    from analyze.silence_detection import main as silence_main
    from analyze.sentiment_shift import main as sentiment_main

    analysis_steps = [
        (
            "frequency_drift",
            freq_main,
            ["--normalized-dir", "data/normalized", "--end-date", day],
        ),
        (
            "topic_convergence",
            conv_main,
            ["--normalized-dir", "data/normalized", "--end-date", day],
        ),
        (
            "silence_detection",
            silence_main,
            ["--normalized-dir", "data/normalized", "--end-date", day],
        ),
        (
            "sentiment_shift",
            sentiment_main,
            ["--normalized-dir", "data/normalized", "--end-date", day],
        ),
    ]

    start_time = time.time()
    for step_name, step_main, step_args in analysis_steps:
        step_start = time.time()
        exit_code = step_main(step_args)
        step_elapsed = time.time() - step_start

        if exit_code != 0:
            if exit_code == 2:
                log.warning(
                    f"Analysis step '{step_name}' skipped or incomplete with exit code {exit_code} "
                    f"(took {step_elapsed:.2f}s)"
                )
                continue
            log.error(
                f"Analysis step '{step_name}' failed with exit code {exit_code} (took {step_elapsed:.2f}s)"
            )
            raise RuntimeError(
                f"Analysis step '{step_name}' failed with exit code {exit_code}"
            )

        log.info(f"Analysis step '{step_name}' completed (took {step_elapsed:.2f}s)")

    elapsed = time.time() - start_time
    log.info(f"All analysis steps completed successfully (total: {elapsed:.2f}s)")


def run_report(day: str):
    """Run report generation step."""
    log.info("Starting report")
    from report.weekly_report import main as report_main

    start_time = time.time()
    exit_code = report_main(
        [
            "--normalized-dir",
            "data/normalized",
            "--outdir",
            "reports",
            "--end-date",
            day,
            "--window-days",
            "7",
            "--baseline-days",
            "7",
        ]
    )
    elapsed = time.time() - start_time

    if exit_code != 0:
        log.error(f"Report failed with exit code {exit_code} (took {elapsed:.2f}s)")
        raise RuntimeError(f"Report step failed with exit code {exit_code}")

    log.info(f"Report completed successfully (took {elapsed:.2f}s)")
    return exit_code


def main():
    """Main pipeline orchestrator."""
    pipeline_start = time.time()

    try:
        log.info("Chaos Observatory run started")
        day = utc_day()

        run_ingest()
        run_normalize(day)
        run_analysis(day)
        run_report(day)

        total_elapsed = time.time() - pipeline_start
        log.info(
            f"Chaos Observatory run completed successfully (total: {total_elapsed:.2f}s)"
        )
        return 0

    except Exception as e:
        total_elapsed = time.time() - pipeline_start
        log.exception(f"Chaos Observatory run failed after {total_elapsed:.2f}s: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
