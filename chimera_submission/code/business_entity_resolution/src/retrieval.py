"""Bounded, deterministic lexical candidate generation without label features."""

from __future__ import annotations

import csv
import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalization import NormalizedRecord


CHANNELS = ("exact_name", "exact_core", "char_name", "char_address", "rare_token")

# Phase B — optional numeric/postal blocker. Opt-in only; the default
# CHANNELS tuple is unchanged so existing artifacts and tests are stable.
NUMERIC_CHANNEL = "numeric"
OPTIONAL_CHANNELS = (NUMERIC_CHANNEL,)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 50
    max_candidates: int = 250
    query_batch_size: int = 128
    max_features: int = 1_000_000
    max_token_df: int = 1_000
    char_ngram_range: tuple[int, int] = (2, 4)
    # Phase B (opt-in, defaults preserve legacy behavior):
    # numeric_top_k enables the digit-token blocker when set; cap_sweep
    # records per-cap candidate counts without changing retrieval ranking.
    numeric_top_k: int | None = None
    sweep_caps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if min(self.top_k, self.max_candidates, self.query_batch_size,
               self.max_features, self.max_token_df) < 1:
            raise ValueError("retrieval limits must be positive")
        low, high = self.char_ngram_range
        if low < 1 or high < low:
            raise ValueError("char_ngram_range must be positive and ordered")
        if self.numeric_top_k is not None and self.numeric_top_k < 1:
            raise ValueError("numeric_top_k must be positive when set")
        if any(cap < 1 for cap in self.sweep_caps):
            raise ValueError("sweep_caps must be positive")
        if self.sweep_caps and tuple(sorted(self.sweep_caps)) != tuple(self.sweep_caps):
            raise ValueError("sweep_caps must be sorted ascending")


@dataclass(frozen=True)
class ChannelHit:
    channel: str
    rank: int
    score: float


@dataclass(frozen=True)
class Candidate:
    source1_entity_id: str
    candidate_entity_id: str
    rank: int
    score: float
    hits: tuple[ChannelHit, ...]

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(hit.channel for hit in self.hits)

    @property
    def channel_count(self) -> int:
        return len(self.hits)


def _exact_index(targets: Mapping[str, NormalizedRecord], view: str) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for identifier in sorted(targets):
        value = getattr(targets[identifier], view)
        if value:
            index[value].append(identifier)
    return index


class _CharIndex:
    def __init__(self, targets: Mapping[str, NormalizedRecord], view: str,
                 config: RetrievalConfig):
        self.ids = sorted(targets)
        self.vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=config.char_ngram_range,
            dtype=np.float32, max_features=config.max_features, norm="l2",
        )
        texts = [getattr(targets[identifier], view) for identifier in self.ids]
        if not any(len(text) >= config.char_ngram_range[0] for text in texts):
            self.matrix_transpose = None
        else:
            matrix = self.vectorizer.fit_transform(texts)
            self.matrix_transpose = matrix.T
        self.top_k = config.top_k

    def query_batch(self, texts: list[str]) -> list[list[tuple[str, float]]]:
        if self.matrix_transpose is None:
            return [[] for _ in texts]
        query_matrix = self.vectorizer.transform(texts)
        results = []
        for row_index in range(len(texts)):
            similarities = query_matrix.getrow(row_index) @ self.matrix_transpose
            pairs = (
                (self.ids[index], float(score))
                for index, score in zip(similarities.indices, similarities.data)
                if score > 0
            )
            results.append(heapq.nsmallest(
                self.top_k, pairs, key=lambda pair: (-pair[1], pair[0])
            ))
        return results


class _RareTokenIndex:
    def __init__(self, targets: Mapping[str, NormalizedRecord], max_token_df: int):
        document_frequency = Counter(
            token for record in targets.values()
            for token in set(record.business_name_core.split())
        )
        self.postings: dict[str, list[str]] = defaultdict(list)
        self.weights: dict[str, float] = {}
        for token, frequency in document_frequency.items():
            if frequency <= max_token_df:
                self.weights[token] = log((len(targets) + 1) / (frequency + 1)) + 1
        for identifier in sorted(targets):
            for token in set(targets[identifier].business_name_core.split()):
                if token in self.weights:
                    self.postings[token].append(identifier)

    def query(self, text: str, top_k: int) -> list[tuple[str, float]]:
        tokens = sorted(set(text.split()) & self.weights.keys())
        denominator = sum(self.weights[token] for token in tokens)
        if not denominator:
            return []
        scores: dict[str, float] = defaultdict(float)
        for token in tokens:
            for identifier in self.postings[token]:
                scores[identifier] += self.weights[token]
        pairs = ((identifier, score / denominator) for identifier, score in scores.items())
        return heapq.nsmallest(top_k, pairs, key=lambda pair: (-pair[1], pair[0]))


def _digit_tokens(text: str) -> frozenset[str]:
    """Digit tokens from a normalized address (house/PIN fragments)."""
    return frozenset(token for token in text.split() if any(ch.isdigit() for ch in token))


class _NumericIndex:
    """Opt-in digit-token blocker for transliteration-robust address recall.

    Deterministic IDF-weighted overlap over digit-bearing address tokens.
    Disabled unless RetrievalConfig.numeric_top_k is set.
    """

    def __init__(self, targets: Mapping[str, NormalizedRecord], max_token_df: int):
        document_frequency = Counter(
            token for record in targets.values()
            for token in _digit_tokens(record.business_address_alias)
        )
        self.postings: dict[str, list[str]] = defaultdict(list)
        self.weights: dict[str, float] = {}
        for token, frequency in document_frequency.items():
            if frequency <= max_token_df:
                self.weights[token] = log((len(targets) + 1) / (frequency + 1)) + 1
        for identifier in sorted(targets):
            for token in _digit_tokens(targets[identifier].business_address_alias):
                if token in self.weights:
                    self.postings[token].append(identifier)

    def query(self, text: str, top_k: int) -> list[tuple[str, float]]:
        tokens = sorted(_digit_tokens(text) & self.weights.keys())
        denominator = sum(self.weights[token] for token in tokens)
        if not denominator:
            return []
        scores: dict[str, float] = defaultdict(float)
        for token in tokens:
            for identifier in self.postings[token]:
                scores[identifier] += self.weights[token]
        pairs = ((identifier, score / denominator) for identifier, score in scores.items())
        return heapq.nsmallest(top_k, pairs, key=lambda pair: (-pair[1], pair[0]))


def exact_truncation_stats(
    targets: Mapping[str, NormalizedRecord], view: str, top_k: int,
) -> dict[str, int | float]:
    """Phase B audit: how often exact-hit lists exceed top_k (ID-order drop).

    Returns total keys, truncated keys, truncated hits, and truncation rate.
    No retrieval is performed; pure index-size diagnostic.
    """
    index = _exact_index(targets, view)
    truncated_keys = sum(1 for ids in index.values() if len(ids) > top_k)
    truncated_hits = sum(len(ids) - top_k for ids in index.values() if len(ids) > top_k)
    total_hits = sum(len(ids) for ids in index.values())
    return {
        "keys": len(index),
        "total_hits": total_hits,
        "truncated_keys": truncated_keys,
        "truncated_hits": truncated_hits,
        "truncation_rate": (truncated_hits / total_hits) if total_hits else 0.0,
    }


def cap_sweep_counts(ordered_target_ids: list[str], caps: tuple[int, ...]) -> dict[int, int]:
    """Phase B helper: retained counts at each candidate cap for one S1.

    `ordered_target_ids` must already be in final union-rank order.
    """
    return {cap: min(len(ordered_target_ids), cap) for cap in caps}


def generate_candidates(
    source1: Mapping[str, NormalizedRecord],
    source2: Mapping[str, NormalizedRecord],
    source3: Mapping[str, NormalizedRecord],
    config: RetrievalConfig = RetrievalConfig(),
) -> Iterable[Candidate]:
    """Yield canonical S1→S2/S3 candidates in ID and rank order."""
    targets = {**source2, **source3}
    if len(targets) != len(source2) + len(source3):
        raise ValueError("target entity IDs overlap across sources")
    if not source1 or not targets:
        return
    exact_name = _exact_index(targets, "business_name_clean")
    exact_core = _exact_index(targets, "business_name_core")
    char_name = _CharIndex(targets, "business_name_clean", config)
    char_address = _CharIndex(targets, "business_address_alias", config)
    rare_tokens = _RareTokenIndex(targets, config.max_token_df)
    numeric_index = (
        _NumericIndex(targets, config.max_token_df)
        if config.numeric_top_k is not None else None
    )
    source_ids = sorted(source1)

    for start in range(0, len(source_ids), config.query_batch_size):
        batch_ids = source_ids[start:start + config.query_batch_size]
        name_hits = char_name.query_batch([
            source1[identifier].business_name_clean for identifier in batch_ids
        ])
        address_hits = char_address.query_batch([
            source1[identifier].business_address_alias for identifier in batch_ids
        ])
        for position, source_id in enumerate(batch_ids):
            source = source1[source_id]
            channel_results = (
                (CHANNELS[0], [(target_id, 1.0) for target_id in
                               exact_name.get(source.business_name_clean, ())[:config.top_k]]),
                (CHANNELS[1], [(target_id, 1.0) for target_id in
                               exact_core.get(source.business_name_core, ())[:config.top_k]]),
                (CHANNELS[2], name_hits[position]),
                (CHANNELS[3], address_hits[position]),
                (CHANNELS[4], rare_tokens.query(source.business_name_core, config.top_k)),
            )
            if numeric_index is not None:
                assert config.numeric_top_k is not None
                channel_results = channel_results + (
                    (NUMERIC_CHANNEL, numeric_index.query(
                        source.business_address_alias, config.numeric_top_k)),
                )
            by_target: dict[str, list[ChannelHit]] = defaultdict(list)
            for channel, hits in channel_results:
                for rank, (target_id, score) in enumerate(hits, start=1):
                    by_target[target_id].append(ChannelHit(channel, rank, score))
            ordered = sorted(
                by_target.items(),
                key=lambda item: (
                    -len(item[1]),
                    -max(hit.score for hit in item[1]),
                    min(hit.rank for hit in item[1]),
                    item[0],
                ),
            )[:config.max_candidates]
            for rank, (target_id, hits) in enumerate(ordered, start=1):
                yield Candidate(
                    source1_entity_id=source_id,
                    candidate_entity_id=target_id,
                    rank=rank,
                    score=max(hit.score for hit in hits),
                    hits=tuple(hits),
                )


def channel_volumes(candidates: Iterable[Candidate]) -> dict[str, int]:
    """Count retained candidates per retrieval channel."""
    counts = Counter(channel for candidate in candidates for channel in candidate.channels)
    return {channel: counts[channel] for channel in CHANNELS}


def write_candidate_artifact(path: str | Path, candidates: Iterable[Candidate]) -> None:
    """Write an auditable candidate set without labels or fold-dependent fields."""
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", "candidate_entity_id", "rank", "score",
                         "channel_count", "channels", "provenance"))
        for candidate in candidates:
            writer.writerow((
                candidate.source1_entity_id, candidate.candidate_entity_id,
                candidate.rank, format(candidate.score, ".9g"), candidate.channel_count,
                ",".join(candidate.channels),
                json.dumps({hit.channel: {"rank": hit.rank, "score": hit.score}
                            for hit in candidate.hits}, sort_keys=True),
            ))
