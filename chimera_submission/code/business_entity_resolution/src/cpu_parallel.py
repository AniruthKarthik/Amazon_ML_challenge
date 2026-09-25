"""Phase I — CPU parallelization helpers (stdlib only).

Rules enforced here:
- Parallelize at S1-block granularity (never per-pair): pickle overhead
  dominates at pair granularity.
- Never nest pools: retrieval threads, feature processes, and LightGBM
  threads must not be active simultaneously.
- Workers are processes for GIL-bound text work, threads for I/O-light
  dict lookups. Caller resolves one worker count and logs it.
"""

from __future__ import annotations

import os


def resolve_workers(requested: int | None = None) -> int:
    """Logical CPUs available to this process (affinity-aware)."""
    if requested is not None:
        if isinstance(requested, bool) or requested < 1:
            raise ValueError("requested workers must be positive")
        return requested
    if hasattr(os, "sched_getaffinity"):
        try:
            count = len(os.sched_getaffinity(0))
            if count > 0:
                return count
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def shard_ranges(total: int, num_shards: int) -> list[tuple[int, int]]:
    """Contiguous [start, stop) S1-seq ranges covering [0, total)."""
    if total < 0:
        raise ValueError("total must be nonnegative")
    if isinstance(num_shards, bool) or num_shards < 1:
        raise ValueError("num_shards must be positive")
    if total == 0:
        return []
    base, extra = divmod(total, num_shards)
    ranges = []
    start = 0
    for i in range(num_shards):
        stop = start + base + (1 if i < extra else 0)
        if stop > start:
            ranges.append((start, stop))
        start = stop
    return ranges


def feature_workers(total_cores: int) -> int:
    """Conservative process count for feature extraction/scoring.

    Model predict is already multithreaded, so use at most half the cores
    for processes to avoid oversubscription.
    """
    if isinstance(total_cores, bool) or total_cores < 1:
        raise ValueError("total_cores must be positive")
    return max(1, total_cores // 2)
