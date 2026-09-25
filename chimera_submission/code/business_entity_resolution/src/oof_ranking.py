"""Retrieved-pair ranking diagnostics from Phase 8 OOF LightGBM scores.

Retrieval misses are context only and never enter a ranking-failure numerator
or denominator. Multiple ground-truth targets per S1 are supported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def evaluate_oof_ranking(
    source_ids: Sequence[str], target_ids: Sequence[str],
    oof_scores: Sequence[float], truth: Mapping[str, frozenset[str]],
) -> dict[str, object]:
    """Rank source-sorted pairs by OOF score, breaking ties by target ID."""
    if len(source_ids) != len(target_ids) or len(source_ids) != len(oof_scores):
        raise ValueError("candidate IDs and OOF scores must align")
    retrieved_true = 0
    top1_true = 0
    top5_true = 0
    top10_true = 0
    entities_with_retrieved_true = 0
    entity_any_top1 = 0
    entity_any_top5 = 0
    entity_any_top10 = 0
    false_positives_above_true = 0
    true_pairs_with_fp_above = 0
    def accumulate(source_id: str, candidates: list[tuple[str, float]]) -> None:
        nonlocal retrieved_true, top1_true, top5_true, top10_true
        nonlocal entities_with_retrieved_true, entity_any_top1
        nonlocal entity_any_top5, entity_any_top10
        nonlocal false_positives_above_true, true_pairs_with_fp_above
        expected = truth[source_id]
        ordered = sorted(candidates, key=lambda pair: (-pair[1], pair[0]))
        found_positions = []
        false_positives_so_far = 0
        for position, (target_id, _) in enumerate(ordered, 1):
            if target_id in expected:
                found_positions.append(position)
                false_positives_above_true += false_positives_so_far
                true_pairs_with_fp_above += false_positives_so_far > 0
            else:
                false_positives_so_far += 1
        if not found_positions:
            return
        entities_with_retrieved_true += 1
        retrieved_true += len(found_positions)
        top1_true += sum(position <= 1 for position in found_positions)
        top5_true += sum(position <= 5 for position in found_positions)
        top10_true += sum(position <= 10 for position in found_positions)
        best = min(found_positions)
        entity_any_top1 += best <= 1
        entity_any_top5 += best <= 5
        entity_any_top10 += best <= 10

    current_source = None
    current_candidates: list[tuple[str, float]] = []
    current_targets: set[str] = set()
    for source_id, target_id, score in zip(source_ids, target_ids, oof_scores):
        if source_id not in truth:
            raise ValueError(f"candidate source missing from ground truth: {source_id}")
        if not math.isfinite(score):
            raise ValueError("OOF scores must be finite")
        if current_source is not None and source_id < current_source:
            raise ValueError("candidate pairs must be sorted by S1 entity ID")
        if source_id != current_source:
            if current_source is not None:
                accumulate(current_source, current_candidates)
            current_source = source_id
            current_candidates = []
            current_targets = set()
        if target_id in current_targets:
            raise ValueError("duplicate candidate pair")
        current_targets.add(target_id)
        current_candidates.append((target_id, float(score)))
    if current_source is not None:
        accumulate(current_source, current_candidates)

    total_true = sum(len(targets) for targets in truth.values())
    if retrieved_true > total_true:
        raise AssertionError("retrieved true pairs exceed ground truth")
    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "criterion": "retrieved true pair ranks first by descending OOF LightGBM score; target ID breaks ties",
        "scope": "retrieved true pairs only; retrieval misses excluded from ranking rates",
        "ground_truth_pairs": total_true,
        "retrieved_true_pairs": retrieved_true,
        "context_retrieval_misses": total_true - retrieved_true,
        "top1_pair_ranking_failures": retrieved_true - top1_true,
        "top1_pair_ranking_failure_rate": rate(retrieved_true - top1_true, retrieved_true),
        "retrieved_pair_recall_at_1": rate(top1_true, retrieved_true),
        "retrieved_pair_recall_at_5": rate(top5_true, retrieved_true),
        "retrieved_pair_recall_at_10": rate(top10_true, retrieved_true),
        "entities_with_retrieved_true": entities_with_retrieved_true,
        "entity_any_true_at_1": rate(entity_any_top1, entities_with_retrieved_true),
        "entity_any_true_at_5": rate(entity_any_top5, entities_with_retrieved_true),
        "entity_any_true_at_10": rate(entity_any_top10, entities_with_retrieved_true),
        "retrieved_true_pairs_with_false_positive_above": true_pairs_with_fp_above,
        "mean_false_positives_above_retrieved_true": rate(
            false_positives_above_true, retrieved_true
        ),
    }
