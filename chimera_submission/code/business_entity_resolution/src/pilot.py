"""Phase A — stratified pilot harness (stdlib only, no test labels needed).

Provides deterministic country x cardinality stratified sampling so every
experiment (E0-E15) reuses the same pilot S1 set. Pilot directories are
isolated from full-run paths; nothing here reads test labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def cardinality_bucket(n: int) -> str:
    """Bucket ground-truth match counts for stratification."""
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4+"


def stratum_key(country: str, truth_size: int) -> str:
    """Stratum label: normalized country x cardinality bucket."""
    return f"{(country or '').strip().casefold() or 'missing'}|{cardinality_bucket(truth_size)}"


def stratified_sample(
    source_ids: list[str],
    countries: dict[str, str],
    truth_sizes: dict[str, int],
    n: int,
    seed: int = 42,
) -> list[str]:
    """Deterministic stratified sample preserving stratum proportions.

    Falls back to proportional allocation with largest-remainder rounding;
    every stratum with at least one member contributes at least one sample
    when n >= number of strata. Within-stratum order is sorted for determinism
    with a seeded shuffle.
    """
    if n < 1:
        raise ValueError("pilot size must be positive")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source IDs must be unique")
    if n > len(source_ids):
        raise ValueError("pilot size exceeds population")
    strata: dict[str, list[str]] = {}
    for sid in source_ids:
        strata.setdefault(stratum_key(countries.get(sid, ""), truth_sizes.get(sid, 0)), []).append(sid)
    for members in strata.values():
        members.sort()
    # Proportional allocation with largest remainder.
    total = len(source_ids)
    quotas = {k: len(v) * n / total for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    remainder = n - sum(alloc.values())
    # Ensure minimal coverage when feasible.
    if n >= len(strata):
        for k in strata:
            if alloc[k] == 0:
                alloc[k] = 1
                remainder -= 1
    order = sorted(strata, key=lambda k: (quotas[k] - alloc[k], k), reverse=True)
    idx = 0
    while remainder > 0:
        k = order[idx % len(order)]
        if alloc[k] < len(strata[k]):
            alloc[k] += 1
            remainder -= 1
        idx += 1
        if idx > len(order) * (n + 1):
            break
    while remainder < 0:
        # Remove from largest over-allocated stratum above its minimum.
        k = max(order, key=lambda kk: alloc[kk] - quotas[kk])
        if alloc[k] > 1:
            alloc[k] -= 1
            remainder += 1
        else:
            break
    rng = random.Random(seed)
    sampled: list[str] = []
    for k in sorted(strata):
        members = list(strata[k])
        rng.shuffle(members)
        sampled.extend(members[: alloc[k]])
    if len(sampled) != n:
        raise AssertionError("stratified allocation did not fill pilot size")
    return sorted(sampled)


def pilot_manifest(
    sampled: list[str],
    countries: dict[str, str],
    truth_sizes: dict[str, int],
    seed: int,
) -> dict:
    """Auditable manifest: stratum distribution + content hash."""
    dist = Counter(stratum_key(countries.get(s, ""), truth_sizes.get(s, 0)) for s in sampled)
    digest = hashlib.sha256("\n".join(sampled).encode("utf-8")).hexdigest()
    return {
        "seed": seed,
        "pilot_size": len(sampled),
        "stratum_counts": dict(sorted(dist.items())),
        "sha256": digest,
    }


def write_pilot_manifest(path: str | Path, manifest: dict) -> None:
    """Write manifest without overwriting an existing pilot."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_pilot_ids(path: str | Path, sampled: list[str]) -> None:
    """Write one S1 ID per line (plus header) for reuse across experiments."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id",))
        for sid in sampled:
            writer.writerow((sid,))
