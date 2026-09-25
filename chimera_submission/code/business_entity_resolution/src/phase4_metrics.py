"""Streaming, multi-positive-aware Phase 4 retrieval metrics.

All scores here use candidate membership only. No pair model or ranking result is
available at this stage.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from math import ceil
from typing import Iterable

from .retrieval import CHANNELS, Candidate


CHANNEL_BITS = {channel: 1 << position for position, channel in enumerate(CHANNELS)}
VIEWS = {
    **{channel: CHANNEL_BITS[channel] for channel in CHANNELS},
    "Exact": CHANNEL_BITS["exact_name"] | CHANNEL_BITS["exact_core"],
    "Char": CHANNEL_BITS["char_name"] | CHANNEL_BITS["char_address"],
    "Exact ∪ Char": sum(CHANNEL_BITS[channel] for channel in CHANNELS[:4]),
    "Exact ∪ Char ∪ Rare": sum(CHANNEL_BITS.values()),
}
MASK_VIEWS = {
    mask: tuple(name for name, view_mask in VIEWS.items() if mask & view_mask)
    for mask in range(1, 1 << len(CHANNELS))
}
MASK_PAIRS = {
    mask: tuple(
        (first, second) for position, first in enumerate(CHANNELS)
        for second in CHANNELS[position + 1:]
        if mask & CHANNEL_BITS[first] and mask & CHANNEL_BITS[second]
    )
    for mask in range(1, 1 << len(CHANNELS))
}


def oracle_f05(true_count: int, recovered_count: int) -> float:
    """Perfect pair classification over the candidate set, including singletons."""
    if true_count == 0:
        return 1.0
    if recovered_count == 0:
        return 0.0
    return 1.25 * recovered_count / (0.25 * true_count + recovered_count)


@dataclass
class _Counts:
    entities: int = 0
    positive_entities: int = 0
    any_hit_entities: int = 0
    complete_entities: int = 0
    links: int = 0
    recovered_links: int = 0
    candidate_pairs: int = 0
    oracle_sum: float = 0.0

    def add(self, truth_count: int, recovered_count: int, candidate_count: int) -> None:
        self.entities += 1
        self.links += truth_count
        self.recovered_links += recovered_count
        self.candidate_pairs += candidate_count
        self.oracle_sum += oracle_f05(truth_count, recovered_count)
        if truth_count:
            self.positive_entities += 1
            self.any_hit_entities += recovered_count > 0
            self.complete_entities += recovered_count == truth_count

    def report(self, target_pool_size: int) -> dict[str, float | int | None]:
        return {
            "entities": self.entities,
            "positive_entities": self.positive_entities,
            "ground_truth_links": self.links,
            "recovered_links": self.recovered_links,
            "retrieval_misses": self.links - self.recovered_links,
            "candidate_recall": self.recovered_links / self.links if self.links else None,
            "retrieval_miss_rate": (self.links - self.recovered_links) / self.links if self.links else None,
            "entity_any_hit_recall": self.any_hit_entities / self.positive_entities if self.positive_entities else None,
            "entity_complete_recall": self.complete_entities / self.positive_entities if self.positive_entities else None,
            "candidate_pairs": self.candidate_pairs,
            "candidates_per_query": self.candidate_pairs / self.entities if self.entities else None,
            "reduction_ratio": (
                1 - self.candidate_pairs / (self.entities * target_pool_size)
                if self.entities and target_pool_size else None
            ),
            "oracle_entity_macro_f05": self.oracle_sum / self.entities if self.entities else None,
        }


class RetrievalBenchmark:
    """Accumulate metrics one S1 entity at a time without retaining candidates."""

    def __init__(self, target_pool_sizes: dict[str, int], sample_limit: int = 1000,
                 random_seed: int = 42):
        if sample_limit < 0:
            raise ValueError("sample_limit must be nonnegative")
        self.target_pool_sizes = target_pool_sizes
        self.groups: dict[str, _Counts] = {}
        self.views = {name: _Counts() for name in VIEWS}
        self.channel_pair_overlap = Counter()
        self.channel_true_overlap = Counter()
        self.exclusive_true_hits = Counter()
        self.mask_candidate_counts = Counter()
        self.mask_true_counts = Counter()
        self.candidate_count_histogram = Counter()
        self.sample_limit = sample_limit
        self.random = random.Random(random_seed)
        self.misses_seen = 0
        self.miss_sample: list[tuple[str, str]] = []

    def _group(self, key: str) -> _Counts:
        return self.groups.setdefault(key, _Counts())

    def _sample_miss(self, source_id: str, target_id: str) -> None:
        self.misses_seen += 1
        if len(self.miss_sample) < self.sample_limit:
            self.miss_sample.append((source_id, target_id))
            return
        replacement = self.random.randrange(self.misses_seen)
        if replacement < self.sample_limit:
            self.miss_sample[replacement] = (source_id, target_id)

    def add_query(self, source_id: str, source_country: str,
                  truth: dict[str, tuple[str, str]],
                  candidates: Iterable[Candidate],
                  target_metadata: dict[str, tuple[str, str]]) -> None:
        """Add one entity. Metadata values are (source, country) tuples."""
        masks: dict[str, int] = {}
        for candidate in candidates:
            if candidate.source1_entity_id != source_id:
                raise ValueError("candidate belongs to another S1 entity")
            target_id = candidate.candidate_entity_id
            if target_id in masks:
                raise ValueError("duplicate candidate pair")
            if target_id not in target_metadata:
                raise ValueError("candidate target is unknown")
            masks[target_id] = sum(CHANNEL_BITS[channel] for channel in candidate.channels)
        self.add_query_masks(source_id, source_country, truth, masks, target_metadata)

    def add_query_masks(self, source_id: str, source_country: str,
                        truth: dict, masks: dict, target_metadata) -> None:
        """Add integer-indexed candidates from a bounded retrieval runner."""
        candidate_count_by_source = Counter()
        candidate_count_by_country = Counter()
        for target_id in masks:
            if target_id not in target_metadata:
                raise ValueError("candidate target is unknown")
            target_source, target_country = target_metadata[target_id]
            candidate_count_by_source[target_source] += 1
            candidate_count_by_country[target_country] += 1

        true_ids = set(truth)
        recovered = true_ids & masks.keys()
        self.candidate_count_histogram[len(masks)] += 1
        for target_id in sorted(true_ids - recovered):
            self._sample_miss(source_id, target_id)
        self._group("overall").add(len(true_ids), len(recovered), len(masks))
        self._group(f"source_country:{source_country}").add(
            len(true_ids), len(recovered), len(masks)
        )

        for target_source in ("S2", "S3"):
            scoped_true = {identifier for identifier in true_ids if truth[identifier][0] == target_source}
            scoped_recovered = scoped_true & masks.keys()
            self._group(f"target_source:{target_source}").add(
                len(scoped_true), len(scoped_recovered), candidate_count_by_source[target_source]
            )
            self._group(f"source_country:{source_country}|target_source:{target_source}").add(
                len(scoped_true), len(scoped_recovered), candidate_count_by_source[target_source]
            )
        for target_country in self.target_pool_sizes:
            if not target_country.startswith("country:"):
                continue
            country = target_country.removeprefix("country:")
            scoped_true = {identifier for identifier in true_ids if truth[identifier][1] == country}
            scoped_recovered = scoped_true & masks.keys()
            self._group(f"target_country:{country}").add(
                len(scoped_true), len(scoped_recovered),
                candidate_count_by_country[country],
            )

        view_candidates = Counter()
        view_recovered = Counter()
        for target_id, mask in masks.items():
            self.mask_candidate_counts[mask] += 1
            is_true = target_id in true_ids
            if is_true:
                self.mask_true_counts[mask] += 1
                if mask.bit_count() == 1:
                    self.exclusive_true_hits[CHANNELS[mask.bit_length() - 1]] += 1
            for pair in MASK_PAIRS[mask]:
                self.channel_pair_overlap[pair] += 1
                if is_true:
                    self.channel_true_overlap[pair] += 1
            for view_name in MASK_VIEWS[mask]:
                view_candidates[view_name] += 1
                if is_true:
                    view_recovered[view_name] += 1

        for view_name in VIEWS:
            self.views[view_name].add(
                len(true_ids), view_recovered[view_name], view_candidates[view_name]
            )

    def report(self) -> dict[str, object]:
        all_targets = self.target_pool_sizes["all"]
        query_count = sum(self.candidate_count_histogram.values())

        def percentile(fraction: float) -> int | None:
            if not query_count:
                return None
            rank = max(1, ceil(fraction * query_count))
            seen = 0
            for count, frequency in sorted(self.candidate_count_histogram.items()):
                seen += frequency
                if seen >= rank:
                    return count
            raise AssertionError("candidate histogram is incomplete")

        groups = {}
        for key, counts in sorted(self.groups.items()):
            if "|target_source:" in key:
                target_source = key.rsplit(":", 1)[-1]
                pool_size = self.target_pool_sizes[target_source]
            elif key.startswith("target_source:"):
                pool_size = self.target_pool_sizes[key.split(":", 1)[1]]
            elif key.startswith("target_country:"):
                pool_size = self.target_pool_sizes[f"country:{key.split(':', 1)[1]}"]
            else:
                pool_size = all_targets
            groups[key] = counts.report(pool_size)
        return {
            "metric_definitions": {
                "entity_recall_denominator": "S1 entities with at least one ground-truth target in the reported group",
                "oracle_prediction": "ground-truth targets present among retrieved candidates; no false positives",
                "oracle_singleton_score": 1.0,
                "ranking_metrics": "deferred until OOF LightGBM scores after Phases 7/8",
                "channel_diversity_scope": "post-cap final candidate set",
            },
            "groups": groups,
            "candidate_count_distribution": {
                "zero_candidate_queries": self.candidate_count_histogram[0],
                "minimum": min(self.candidate_count_histogram) if query_count else None,
                "p50": percentile(0.5),
                "p90": percentile(0.9),
                "p99": percentile(0.99),
                "maximum": max(self.candidate_count_histogram) if query_count else None,
            },
            "channel_views": {
                name: self.views[name].report(all_targets) for name in VIEWS
            },
            "incremental_union_recovered_links": {
                "Exact": self.views["Exact"].recovered_links,
                "Exact ∪ Char": (
                    self.views["Exact ∪ Char"].recovered_links
                    - self.views["Exact"].recovered_links
                ),
                "Exact ∪ Char ∪ Rare": (
                    self.views["Exact ∪ Char ∪ Rare"].recovered_links
                    - self.views["Exact ∪ Char"].recovered_links
                ),
            },
            "unique_ground_truth_recovered_only_by_channel": {
                channel: self.exclusive_true_hits[channel] for channel in CHANNELS
            },
            "candidate_pairs_by_exact_channel_mask": {
                str(mask): count for mask, count in sorted(self.mask_candidate_counts.items())
            },
            "ground_truth_pairs_by_exact_channel_mask": {
                str(mask): count for mask, count in sorted(self.mask_true_counts.items())
            },
            "channel_overlap": [
                {"channels": list(pair), "candidate_pairs": self.channel_pair_overlap[pair],
                 "ground_truth_pairs": self.channel_true_overlap[pair]}
                for pair in sorted(self.channel_pair_overlap)
            ],
            "miss_sample_size": len(self.miss_sample),
            "misses_seen": self.misses_seen,
        }
