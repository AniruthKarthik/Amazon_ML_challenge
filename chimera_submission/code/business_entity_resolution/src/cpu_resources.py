"""Resolve a bounded CPU worker count for the existing threaded stages."""

from __future__ import annotations

import argparse
import os


def available_cpu_count() -> int:
    """Count logical CPUs available to this process, respecting affinity."""
    if hasattr(os, "sched_getaffinity"):
        try:
            count = len(os.sched_getaffinity(0))
            if count > 0:
                return count
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def parse_thread_count(value: str) -> int:
    """Accept 'auto' or an explicit positive worker count for argparse."""
    if value.casefold() == "auto":
        return available_cpu_count()
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threads must be 'auto' or a positive integer") from exc
    if count < 1:
        raise argparse.ArgumentTypeError("threads must be 'auto' or a positive integer")
    return count
