"""Phase 9 entity aggregation and exact competition macro F₀.₅."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


def entity_f05(truth: frozenset[str] | set[str],
               predicted: frozenset[str] | set[str]) -> float:
    """Exact per-S1 F₀.₅, including the competition singleton convention."""
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    true_positive = len(truth & predicted)
    false_positive = len(predicted - truth)
    false_negative = len(truth - predicted)
    denominator = 1.25 * true_positive + false_positive + 0.25 * false_negative
    return 1.25 * true_positive / denominator if denominator else 0.0


def macro_f05(truth: Mapping[str, frozenset[str]],
              predictions: Mapping[str, frozenset[str]]) -> float:
    """Average exact entity scores over all ground-truth S1 rows."""
    if not truth:
        raise ValueError("macro F0.5 requires at least one S1 entity")
    unexpected = predictions.keys() - truth.keys()
    if unexpected:
        raise ValueError(f"prediction contains unknown S1 entity: {min(unexpected)}")
    return sum(
        entity_f05(expected, predictions.get(source_id, frozenset()))
        for source_id, expected in truth.items()
    ) / len(truth)


@dataclass(frozen=True)
class EntityScoreSummary:
    source_id: str
    candidate_count: int
    top1_target_id: str | None
    max_score: float
    second_score: float
    score_gap: float
    mean_score: float
    score_std: float
    count_above_threshold: int
    score_concentration: float


def summarize_entity_scores(
    source_id: str, candidates: Sequence[tuple[str, float]],
    threshold: float = 0.5,
) -> EntityScoreSummary:
    """Summarize a complete candidate-score distribution for one S1 entity."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if len({target for target, _ in candidates}) != len(candidates):
        raise ValueError("duplicate target in entity candidates")
    if any(not math.isfinite(score) or not 0 <= score <= 1
           for _, score in candidates):
        raise ValueError("candidate probabilities must be finite and in [0, 1]")
    if not candidates:
        return EntityScoreSummary(source_id, 0, None, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0, 0.0)
    ordered = sorted(candidates, key=lambda pair: (-pair[1], pair[0]))
    scores = [score for _, score in ordered]
    maximum = scores[0]
    second = scores[1] if len(scores) > 1 else 0.0
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    total = sum(scores)
    return EntityScoreSummary(
        source_id=source_id,
        candidate_count=len(candidates),
        top1_target_id=ordered[0][0],
        max_score=maximum,
        second_score=second,
        score_gap=maximum - second,
        mean_score=mean,
        score_std=math.sqrt(variance),
        count_above_threshold=sum(score >= threshold for score in scores),
        score_concentration=maximum / total if total else 0.0,
    )


def compare_oof_score_paths(
    source_ids: Sequence[str], target_ids: Sequence[str],
    raw_scores: Sequence[float], calibrated_scores: Sequence[float],
    truth: Mapping[str, frozenset[str]], threshold: float = 0.5,
) -> dict[str, float | int]:
    """Score raw and calibrated OOF pathways at the same supplied threshold.

    Inputs must be sorted by S1 ID. This is a diagnostic comparison; Phase 10
    selects thresholds and later evidence decides whether calibration is kept.
    """
    count = len(source_ids)
    if not (len(target_ids) == len(raw_scores) == len(calibrated_scores) == count):
        raise ValueError("OOF candidate IDs and scores must align")
    if not truth:
        raise ValueError("comparison requires ground truth")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    index = 0
    raw_total = 0.0
    calibrated_total = 0.0
    for source_id in sorted(truth):
        raw_predicted: set[str] = set()
        calibrated_predicted: set[str] = set()
        seen_targets: set[str] = set()
        while index < count and source_ids[index] == source_id:
            target_id = target_ids[index]
            raw, calibrated = raw_scores[index], calibrated_scores[index]
            if target_id in seen_targets:
                raise ValueError("duplicate candidate pair")
            if not (math.isfinite(raw) and math.isfinite(calibrated)
                    and 0 <= raw <= 1 and 0 <= calibrated <= 1):
                raise ValueError("OOF scores must be finite probabilities")
            seen_targets.add(target_id)
            if raw >= threshold:
                raw_predicted.add(target_id)
            if calibrated >= threshold:
                calibrated_predicted.add(target_id)
            index += 1
        if index < count and source_ids[index] < source_id:
            raise ValueError("OOF pairs must be sorted by known S1 ID")
        expected = truth[source_id]
        raw_total += entity_f05(expected, raw_predicted)
        calibrated_total += entity_f05(expected, calibrated_predicted)
    if index != count:
        raise ValueError("OOF pairs contain an unknown or unsorted S1 ID")
    return {
        "entities": len(truth),
        "threshold": threshold,
        "raw_macro_f05": raw_total / len(truth),
        "calibrated_macro_f05": calibrated_total / len(truth),
    }
