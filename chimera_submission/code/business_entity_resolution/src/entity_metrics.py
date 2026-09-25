"""Phase G — per-cardinality entity metrics (stdlib only).

Complements threshold_search.evaluate_predictions with 0/1/2-3/4+ slices
so multi-match recall collapse is visible separately from singleton behavior.
"""

from __future__ import annotations

from collections import Counter

from .entity_decision import entity_f05


def cardinality_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4+"


def per_cardinality_f05(entities, predictions: dict) -> dict:
    """Mean entity F0.5 + count + mean predicted size per truth-cardinality."""
    buckets: dict[str, list[float]] = {}
    pred_sizes: dict[str, list[int]] = {}
    for entity in entities:
        bucket = cardinality_bucket(len(entity.truth))
        predicted = predictions.get(entity.source_id, [])
        score = entity_f05(set(entity.truth), set(predicted))
        buckets.setdefault(bucket, []).append(score)
        pred_sizes.setdefault(bucket, []).append(len(predicted))
    report = {}
    for bucket in sorted(buckets):
        scores = buckets[bucket]
        sizes = pred_sizes[bucket]
        report[bucket] = {
            "entities": len(scores),
            "macro_f05": sum(scores) / len(scores),
            "mean_predicted": sum(sizes) / len(sizes),
        }
    return report


def prediction_histogram(predictions: dict) -> dict:
    """Count entities by predicted set size bucket."""
    counts = Counter(cardinality_bucket(len(v)) for v in predictions.values())
    return {k: counts.get(k, 0) for k in ("0", "1", "2-3", "4+")}
