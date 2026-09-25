"""Phase H — hard-negative compute gate (stdlib only).

Full nested mining + downstream re-optimization (Phase 8->10) is expensive.
This gate enforces the architecture's compute-gate rule: evaluate a cheap
proxy (pair AUC / sampled OOF pair F0.5) first; only proceed to the full
loop when the proxy clears a pre-defined margin. Rejections are logged,
not silently dropped.
"""

from __future__ import annotations

import json
from pathlib import Path


def proxy_gate(baseline: float, candidate: float, min_margin: float) -> bool:
    """Return True iff candidate exceeds baseline by at least min_margin."""
    for value, name in ((baseline, "baseline"), (candidate, "candidate"),
                        (min_margin, "min_margin")):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
    if min_margin < 0:
        raise ValueError("min_margin must be nonnegative")
    return (candidate - baseline) >= min_margin


def gate_record(
    experiment_id: str, baseline: float, candidate: float,
    min_margin: float, decision: bool, reason: str,
) -> dict:
    """Auditable proxy-gate decision record."""
    if not experiment_id:
        raise ValueError("experiment_id must be nonempty")
    return {
        "experiment_id": experiment_id,
        "baseline_proxy": baseline,
        "candidate_proxy": candidate,
        "delta": candidate - baseline,
        "min_margin": min_margin,
        "proceed_to_full_reopt": decision,
        "reason": reason,
    }


def append_gate_log(path: str | Path, record: dict) -> None:
    """Append a JSON-lines gate record (creates the log if missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
