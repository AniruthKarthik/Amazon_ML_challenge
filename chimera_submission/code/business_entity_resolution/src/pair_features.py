"""Label-free, bounded-memory features for S1-to-target candidate pairs.

Fit TF-IDF encoders on an explicitly supplied unlabeled, fold-appropriate
corpus. The feature transform never reads ground truth or model predictions.
"""

from __future__ import annotations

import math
import re
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TypeAlias

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .normalization import NormalizedRecord
from .retrieval import CHANNELS, Candidate
from .text_sim import (
    COUNTRY_FEATURES,
    V2_FEATURE_NAMES,
    _acronym_match,
    _containment,
    _length_ratio,
    strip_country_features,
)


FeatureRow: TypeAlias = dict[str, float]
_NUMBERS = re.compile(r"\d+")
BASE_FEATURE_NAMES = (
    "name_exact", "name_core_exact", "name_folded_exact",
    "name_levenshtein", "name_core_levenshtein", "name_jaro_winkler",
    "name_token_jaccard", "name_tfidf_cosine", "address_exact",
    "address_levenshtein", "address_jaro_winkler",
    "address_token_jaccard", "address_tfidf_cosine",
    "address_shared_number_count", "address_number_jaccard",
    "country_match", "country_mismatch", "country_missing",
    "source_address_missing", "target_address_missing",
    "both_addresses_missing", "one_address_missing",
    "candidate_rank", "candidate_score", "channel_count",
)
# V2 names live in text_sim.py (stdlib-only); re-exported here for callers.


def normalized_levenshtein(left: str, right: str) -> float:
    """Return 1 - edit distance / max length; missing values score zero."""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, char_left in enumerate(left, 1):
        current = [row]
        for column, char_right in enumerate(right, 1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (char_left != char_right),
            ))
        previous = current
    return 1.0 - previous[-1] / len(left)


def jaro_winkler(left: str, right: str) -> float:
    """Standard Jaro-Winkler similarity with a four-character prefix cap."""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    distance = max(len(left), len(right)) // 2 - 1
    distance = max(distance, 0)
    left_matched = [False] * len(left)
    right_matched = [False] * len(right)
    matches = 0
    for index, character in enumerate(left):
        for other in range(max(0, index - distance),
                           min(index + distance + 1, len(right))):
            if not right_matched[other] and character == right[other]:
                left_matched[index] = True
                right_matched[other] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    right_position = 0
    transpositions = 0
    for index, matched in enumerate(left_matched):
        if not matched:
            continue
        while not right_matched[right_position]:
            right_position += 1
        transpositions += left[index] != right[right_position]
        right_position += 1
    jaro = (
        matches / len(left) + matches / len(right)
        + (matches - transpositions / 2) / matches
    ) / 3
    if jaro < 0.7:
        return jaro
    prefix = 0
    for character_left, character_right in zip(left[:4], right[:4]):
        if character_left != character_right:
            break
        prefix += 1
    return jaro + 0.1 * prefix * (1 - jaro)


def token_jaccard(left: str, right: str) -> float:
    first, second = set(left.split()), set(right.split())
    return len(first & second) / len(first | second) if first and second else 0.0


def _numeric_overlap(left: str, right: str) -> tuple[float, float]:
    first, second = set(_NUMBERS.findall(left)), set(_NUMBERS.findall(right))
    shared = len(first & second)
    return float(shared), shared / len(first | second) if first and second else 0.0


class TfidfCosineEncoder:
    """Streaming IDF fit and bounded cached transforms for one text view."""

    def __init__(self, vocabulary: dict[str, int], idf: np.ndarray,
                 ngram_range: tuple[int, int], cache_size: int):
        self.vectorizer = CountVectorizer(
            analyzer="char", ngram_range=ngram_range,
            vocabulary=vocabulary, dtype=np.float32,
        ) if vocabulary else None
        self.idf = idf
        self.cache_size = cache_size
        self.cache: OrderedDict[str, sparse.csr_matrix] = OrderedDict()

    @classmethod
    def fit(cls, texts: Iterable[str], max_features: int = 100_000,
            ngram_range: tuple[int, int] = (2, 4),
            cache_size: int = 4096) -> TfidfCosineEncoder:
        if max_features < 1 or cache_size < 0:
            raise ValueError("max_features must be positive and cache_size nonnegative")
        low, high = ngram_range
        if low < 1 or high < low:
            raise ValueError("ngram_range must be positive and ordered")
        analyzer = TfidfVectorizer(
            analyzer="char", ngram_range=ngram_range,
        ).build_analyzer()
        term_frequency = Counter()
        document_frequency = Counter()
        document_count = 0
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("TF-IDF corpus must contain strings")
            ngrams = analyzer(text)
            term_frequency.update(ngrams)
            document_frequency.update(set(ngrams))
            document_count += 1
        selected = sorted(
            term_frequency, key=lambda term: (-term_frequency[term], term)
        )[:max_features]
        selected.sort()
        vocabulary = {term: position for position, term in enumerate(selected)}
        idf = np.asarray([
            math.log((1 + document_count) / (1 + document_frequency[term])) + 1
            for term in selected
        ], dtype=np.float32)
        return cls(vocabulary, idf, ngram_range, cache_size)

    def _encode(self, text: str) -> sparse.csr_matrix | None:
        if self.vectorizer is None:
            return None
        if self.cache_size and text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]
        matrix = self.vectorizer.transform([text]).tocsr()
        if matrix.nnz:
            matrix.data *= self.idf[matrix.indices]
            normalize(matrix, norm="l2", copy=False)
        if self.cache_size:
            self.cache[text] = matrix
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return matrix

    def cosine(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        first, second = self._encode(left), self._encode(right)
        if first is None or second is None or not first.nnz or not second.nnz:
            return 0.0
        return min(1.0, max(0.0, float(first.multiply(second).sum())))


class PairFeatureExtractor:
    """Transform retrieved pairs with a fixed, label-free feature schema."""

    def __init__(self, name_tfidf: TfidfCosineEncoder,
                 address_tfidf: TfidfCosineEncoder,
                 channels: tuple[str, ...] = CHANNELS,
                 feature_version: int = 1):
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("channels must be nonempty and unique")
        if feature_version not in (1, 2):
            raise ValueError("feature_version must be 1 (legacy) or 2 (precision v2)")
        self.name_tfidf = name_tfidf
        self.address_tfidf = address_tfidf
        self.channels = channels
        self.feature_version = feature_version

    @property
    def feature_names(self) -> tuple[str, ...]:
        base = BASE_FEATURE_NAMES + (V2_FEATURE_NAMES if self.feature_version == 2 else ())
        return base + tuple(
            f"channel_{channel}_{field}"
            for channel in self.channels
            for field in ("present", "rank", "score")
        )

    @classmethod
    def fit_from_records(
        cls, record_factory: Callable[[], Iterable[NormalizedRecord]],
        channels: tuple[str, ...] = CHANNELS,
        max_features: int = 100_000,
        cache_size: int = 4096,
        feature_version: int = 1,
    ) -> PairFeatureExtractor:
        """Fit on a repeatable, fold-appropriate unlabeled record stream."""
        name_tfidf = TfidfCosineEncoder.fit(
            (record.business_name_clean for record in record_factory()),
            max_features=max_features, cache_size=cache_size,
        )
        address_tfidf = TfidfCosineEncoder.fit(
            (record.business_address_alias for record in record_factory()),
            max_features=max_features, cache_size=cache_size,
        )
        return cls(name_tfidf, address_tfidf, channels, feature_version)

    def features(self, source: NormalizedRecord, target: NormalizedRecord,
                 candidate: Candidate) -> FeatureRow:
        if source.raw.entity_id != candidate.source1_entity_id:
            raise ValueError("candidate source ID does not match source record")
        if target.raw.entity_id != candidate.candidate_entity_id:
            raise ValueError("candidate target ID does not match target record")
        if not source.raw.entity_id.startswith("S1-") or not target.raw.entity_id.startswith(("S2-", "S3-")):
            raise ValueError("pair must be S1 to S2/S3")
        if candidate.rank < 1 or not math.isfinite(candidate.score):
            raise ValueError("candidate rank or score is invalid")
        if not candidate.hits:
            raise ValueError("candidate is missing retrieval provenance")
        hits = {}
        for hit in candidate.hits:
            if hit.channel not in self.channels or hit.channel in hits:
                raise ValueError("candidate has unknown or duplicate channel")
            if hit.rank < 1 or not math.isfinite(hit.score):
                raise ValueError("channel rank or score is invalid")
            hits[hit.channel] = hit

        name1, name2 = source.business_name_clean, target.business_name_clean
        core1, core2 = source.business_name_core, target.business_name_core
        address1 = source.business_address_alias
        address2 = target.business_address_alias
        country1 = source.raw.country.strip().casefold()
        country2 = target.raw.country.strip().casefold()
        shared_numbers, number_jaccard = _numeric_overlap(address1, address2)
        features: FeatureRow = {
            "name_exact": float(bool(name1) and name1 == name2),
            "name_core_exact": float(bool(core1) and core1 == core2),
            "name_folded_exact": float(bool(source.business_name_folded) and
                                       source.business_name_folded == target.business_name_folded),
            "name_levenshtein": normalized_levenshtein(name1, name2),
            "name_core_levenshtein": normalized_levenshtein(core1, core2),
            "name_jaro_winkler": jaro_winkler(name1, name2),
            "name_token_jaccard": token_jaccard(core1, core2),
            "name_tfidf_cosine": self.name_tfidf.cosine(name1, name2),
            "address_exact": float(bool(address1) and address1 == address2),
            "address_levenshtein": normalized_levenshtein(address1, address2),
            "address_jaro_winkler": jaro_winkler(address1, address2),
            "address_token_jaccard": token_jaccard(address1, address2),
            "address_tfidf_cosine": self.address_tfidf.cosine(address1, address2),
            "address_shared_number_count": shared_numbers,
            "address_number_jaccard": number_jaccard,
            "country_match": float(bool(country1) and country1 == country2),
            "country_mismatch": float(bool(country1 and country2) and country1 != country2),
            "country_missing": float(not country1 or not country2),
            "source_address_missing": float(not address1),
            "target_address_missing": float(not address2),
            "both_addresses_missing": float(not address1 and not address2),
            "one_address_missing": float(bool(address1) != bool(address2)),
            "candidate_rank": float(candidate.rank),
            "candidate_score": float(candidate.score),
            "channel_count": float(candidate.channel_count),
        }
        if self.feature_version == 2:
            core_tokens1, core_tokens2 = core1.split(), core2.split()
            first_match = float(bool(core_tokens1 and core_tokens2)
                                and core_tokens1[0] == core_tokens2[0])
            last_match = float(bool(core_tokens1 and core_tokens2)
                               and core_tokens1[-1] == core_tokens2[-1])
            nums1 = set(_NUMBERS.findall(address1))
            nums2 = set(_NUMBERS.findall(address2))
            number_conflict = float(bool(nums1 and nums2) and not (nums1 & nums2))
            first_nums1 = _NUMBERS.findall(address1)[:1]
            first_nums2 = _NUMBERS.findall(address2)[:1]
            first_number_match = float(bool(first_nums1 and first_nums2)
                                       and first_nums1[0] == first_nums2[0])
            name_sim = features["name_levenshtein"]
            addr_sim = features["address_levenshtein"]
            features.update({
                "name_containment": _containment(core1, core2),
                "name_sorted_exact": float(
                    bool(core1) and sorted(core1.split()) == sorted(core2.split())),
                "name_first_token_match": first_match,
                "name_last_token_match": last_match,
                "name_acronym_match": _acronym_match(core1, core2),
                "name_len_ratio": _length_ratio(name1, name2),
                "name_token_count_diff": float(abs(len(core_tokens1) - len(core_tokens2))),
                "address_len_ratio": _length_ratio(address1, address2),
                "address_number_conflict": number_conflict,
                "address_first_number_match": first_number_match,
                "is_s2_target": float(target.raw.entity_id.startswith("S2-")),
                "strong_name_weak_address": float(name_sim >= 0.85 and addr_sim < 0.5),
                "exact_name_number_conflict": float(
                    features["name_exact"] == 1.0 and number_conflict == 1.0),
            })
        for channel in self.channels:
            hit = hits.get(channel)
            features[f"channel_{channel}_present"] = float(hit is not None)
            features[f"channel_{channel}_rank"] = float(hit.rank if hit else 0)
            features[f"channel_{channel}_score"] = float(hit.score if hit else 0.0)
        if tuple(features) != self.feature_names:
            raise AssertionError("pair feature schema changed unexpectedly")
        if not all(math.isfinite(value) for value in features.values()):
            raise ValueError("pair features contain NaN or infinity")
        return features

    def transform(
        self, candidates: Iterable[Candidate],
        source1: Mapping[str, NormalizedRecord],
        targets: Mapping[str, NormalizedRecord],
    ) -> Iterator[tuple[Candidate, FeatureRow]]:
        """Yield features lazily; inputs need not be held by this extractor."""
        for candidate in candidates:
            try:
                source = source1[candidate.source1_entity_id]
                target = targets[candidate.candidate_entity_id]
            except KeyError as exc:
                raise ValueError(f"candidate references an unknown record: {exc}") from exc
            yield candidate, self.features(source, target, candidate)
