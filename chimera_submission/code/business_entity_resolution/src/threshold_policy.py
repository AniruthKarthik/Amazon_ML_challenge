"""Pure Phase 10 match-set rules and immutable inference configuration."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Literal, Sequence


PolicyName = Literal["A", "B", "C"]
POLICY_NAMES = ("A", "B", "C")
CONFIG_VERSION = 1


def _threshold(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return float(value)


def _ordered_candidates(candidate_ids: Sequence[str], scores: Sequence[float]
                        ) -> list[tuple[str, float]]:
    if len(candidate_ids) != len(scores):
        raise ValueError("candidate IDs and scores must align")
    best_by_id: dict[str, float] = {}
    for identifier, score in zip(candidate_ids, scores):
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("candidate IDs must be nonempty strings")
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("candidate scores must be finite probabilities")
        best_by_id[identifier] = max(float(score), best_by_id.get(identifier, -math.inf))
    return sorted(best_by_id.items(), key=lambda pair: (-pair[1], pair[0]))


def predict_match_set(
    candidate_ids: Sequence[str], scores: Sequence[float],
    pair_threshold: float, entity_threshold: float, gap_threshold: float,
) -> list[str]:
    """Policy C: entity gate, top-1 gap branch, else pair gate with top-1 fallback.

    The entity gate fails only when top1 < entity_threshold. Gap and pair gates
    pass on equality. One candidate has decision gap +inf; tied top scores have
    gap zero. Output is unique and ordered by score, then candidate ID.
    """
    pair_threshold = _threshold(pair_threshold, "pair_threshold")
    entity_threshold = _threshold(entity_threshold, "entity_threshold")
    gap_threshold = _threshold(gap_threshold, "gap_threshold")
    ordered = _ordered_candidates(candidate_ids, scores)
    if not ordered or ordered[0][1] < entity_threshold:
        return []
    top1_id, top1_score = ordered[0]
    gap = top1_score - ordered[1][1] if len(ordered) > 1 else math.inf
    if gap >= gap_threshold:
        return [top1_id]
    selected = [identifier for identifier, score in ordered if score >= pair_threshold]
    return selected or [top1_id]


def predict_match_set_nofallback(
    candidate_ids: Sequence[str], scores: Sequence[float],
    pair_threshold: float, entity_threshold: float,
    max_matches: int | None = None,
) -> list[str]:
    """Phase G Policy D: entity gate + pair gate, NO forced top-1 fallback.

    Uncertain entities (gate passes but nothing passes pair_threshold) return
    empty instead of a coerced single. Optional max_matches caps multi-match
    output (covers 99.9% of truth mass at 6-8). Output is a candidate subset
    ordered by score, then candidate ID.
    """
    pair_threshold = _threshold(pair_threshold, "pair_threshold")
    entity_threshold = _threshold(entity_threshold, "entity_threshold")
    if max_matches is not None and (isinstance(max_matches, bool) or max_matches < 1):
        raise ValueError("max_matches must be positive when set")
    ordered = _ordered_candidates(candidate_ids, scores)
    if not ordered or ordered[0][1] < entity_threshold:
        return []
    selected = [identifier for identifier, score in ordered if score >= pair_threshold]
    if max_matches is not None:
        selected = selected[:max_matches]
    return selected


def fine_threshold_grid(lo: float, hi: float, n: int) -> tuple[float, ...]:
    """Phase G helper: evenly spaced [0,1] grid with n>=1 points (not quantiles)."""
    _threshold(lo, "lo")
    _threshold(hi, "hi")
    if isinstance(n, bool) or n < 1:
        raise ValueError("n must be positive")
    if hi < lo:
        raise ValueError("hi must be >= lo")
    if n == 1:
        return (float(lo),)
    return tuple(round(lo + (hi - lo) * i / (n - 1), 6) for i in range(n))


def predict_policy(
    policy: PolicyName, candidate_ids: Sequence[str], scores: Sequence[float],
    pair_threshold: float, entity_threshold: float | None = None,
    gap_threshold: float | None = None,
) -> list[str]:
    """Compare A (pair only), B (entity + pair), and C (full gap rule)."""
    if policy not in POLICY_NAMES:
        raise ValueError("unknown decision policy")
    pair_threshold = _threshold(pair_threshold, "pair_threshold")
    if policy == "C":
        if entity_threshold is None or gap_threshold is None:
            raise ValueError("Policy C needs entity and gap thresholds")
        return predict_match_set(candidate_ids, scores, pair_threshold,
                                 entity_threshold, gap_threshold)
    ordered = _ordered_candidates(candidate_ids, scores)
    if policy == "A":
        return [identifier for identifier, score in ordered if score >= pair_threshold]
    if entity_threshold is None:
        raise ValueError("Policy B needs an entity threshold")
    entity_threshold = _threshold(entity_threshold, "entity_threshold")
    if not ordered or ordered[0][1] < entity_threshold:
        return []
    selected = [identifier for identifier, score in ordered if score >= pair_threshold]
    return selected or [ordered[0][0]]


@dataclass(frozen=True)
class FrozenDecisionConfig:
    """The selected OOF policy; no threshold may be inferred from test data."""

    policy: PolicyName
    pair_threshold: float
    entity_threshold: float | None
    gap_threshold: float | None
    score_path: Literal["raw", "calibrated"]
    selection_method: str = "outer-fold OOF robustness, then all-OOF fit"
    schema_version: int = CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.policy not in POLICY_NAMES or self.score_path not in ("raw", "calibrated"):
            raise ValueError("invalid frozen policy or score path")
        _threshold(self.pair_threshold, "pair_threshold")
        if self.policy in ("B", "C"):
            if self.entity_threshold is None:
                raise ValueError("entity threshold is required")
            _threshold(self.entity_threshold, "entity_threshold")
        elif self.entity_threshold is not None:
            raise ValueError("Policy A cannot carry an unused entity threshold")
        if self.policy == "C":
            if self.gap_threshold is None:
                raise ValueError("gap threshold is required")
            _threshold(self.gap_threshold, "gap_threshold")
        elif self.gap_threshold is not None:
            raise ValueError("Policy A/B cannot carry an unused gap threshold")
        if self.schema_version != CONFIG_VERSION:
            raise ValueError("unsupported decision config version")

    def predict(self, candidate_ids: Sequence[str], scores: Sequence[float]) -> list[str]:
        return predict_policy(self.policy, candidate_ids, scores,
                              self.pair_threshold, self.entity_threshold,
                              self.gap_threshold)


def save_frozen_config(path: str | Path, config: FrozenDecisionConfig) -> None:
    """Serialize exactly once so inference uses the same decision function."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_frozen_config(path: str | Path) -> FrozenDecisionConfig:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) != set(FrozenDecisionConfig.__dataclass_fields__):
        raise ValueError("decision config schema mismatch")
    return FrozenDecisionConfig(**values)
