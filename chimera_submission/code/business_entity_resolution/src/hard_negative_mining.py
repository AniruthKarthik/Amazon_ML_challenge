"""Phase 11 nested, component-disjoint hard-negative mining for raw OOF scores.

For an outer validation fold, mining models see only the other folds. Each
training pair's mining score is itself produced by a model excluding its fold.
The held-out fold's labels cannot affect either mining or its final scorer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np

from .pair_model import BaselineConfig, train_pair_baseline
from .threshold_search import (
    ThresholdGrid, ThresholdSearchResult, assemble_oof_entities, search_thresholds,
)


@dataclass(frozen=True)
class HardNegativeConfig:
    """Explicit experiment settings; no cutoff is inferred from test data."""

    score_threshold: float
    max_per_source: int
    extra_weight: float

    def __post_init__(self) -> None:
        if (isinstance(self.score_threshold, bool)
                or not isinstance(self.score_threshold, Real)
                or not math.isfinite(self.score_threshold)
                or not 0 <= self.score_threshold <= 1):
            raise ValueError("mining score threshold must be in [0, 1]")
        if isinstance(self.max_per_source, bool) or not isinstance(self.max_per_source, int) or self.max_per_source < 1:
            raise ValueError("max_per_source must be a positive integer")
        if (isinstance(self.extra_weight, bool) or not isinstance(self.extra_weight, Real)
                or not math.isfinite(self.extra_weight) or self.extra_weight <= 0):
            raise ValueError("extra_weight must be finite and positive")


def mine_hard_negatives(
    source_ids: Sequence[str], target_ids: Sequence[str], labels: Sequence[int],
    oof_scores: Sequence[float], config: HardNegativeConfig,
) -> tuple[int, ...]:
    """Pick highest-scoring known negatives per S1, with stable target-ID ties.

    A score equal to the cutoff qualifies. The output contains row indices in
    source/score/target order and never includes positives.
    """
    if not (len(source_ids) == len(target_ids) == len(labels) == len(oof_scores)):
        raise ValueError("mining pair arrays must align")
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[tuple[float, str, int]]] = {}
    for index, (source, target, label, score) in enumerate(
        zip(source_ids, target_ids, labels, oof_scores)
    ):
        if (not isinstance(source, str) or not source
                or not isinstance(target, str) or not target
                or (source, target) in seen):
            raise ValueError("mining requires unique, nonempty candidate pair IDs")
        if label not in (0, 1):
            raise ValueError("mining labels must be binary")
        if (isinstance(score, bool) or not isinstance(score, Real)
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("mining scores must be finite probabilities")
        seen.add((source, target))
        if label == 0 and score >= config.score_threshold:
            grouped.setdefault(source, []).append((float(score), target, index))
    selected = []
    for source in sorted(grouped):
        ranked = sorted(grouped[source], key=lambda item: (-item[0], item[1]))
        selected.extend(index for _, _, index in ranked[:config.max_per_source])
    return tuple(selected)


@dataclass(frozen=True)
class HardNegativeOOFResult:
    raw_scores: np.ndarray
    fold_ids: np.ndarray
    fold_models: tuple[lgb.Booster, ...]
    mined_indices_by_fold: dict[int, tuple[int, ...]]
    fold_diagnostics: dict[int, dict[str, float | int | None]]
    mining_config: HardNegativeConfig


def generate_hard_negative_oof_predictions(
    features: np.ndarray, labels: np.ndarray,
    source_ids: Sequence[str], target_ids: Sequence[str],
    fold_by_source: Mapping[str, int],
    component_by_source: Mapping[str, str | int],
    feature_names: tuple[str, ...], mining_config: HardNegativeConfig,
    model_config: BaselineConfig = BaselineConfig(),
) -> HardNegativeOOFResult:
    """Nested-mine training negatives, reweight them, and score outer OOF folds.

    Uses fixed boosting rounds for every inner and outer fit. Reweighting
    avoids duplicating large feature matrices. Raw OOF output must flow through
    Phases 9/10 again; acceptance requires the same downstream re-optimization
    for the baseline and the mined model. Calibration, if kept, must also be
    regenerated rather than reusing the old calibrator.
    """
    matrix = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels)
    count = len(source_ids)
    if (matrix.ndim != 2 or not count or len(matrix) != count
            or targets.ndim != 1 or len(targets) != count
            or len(target_ids) != count):
        raise ValueError("pair features, IDs and labels must align")
    if not np.isfinite(matrix).all() or not np.isin(targets, (0, 1)).all():
        raise ValueError("features must be finite and labels binary")
    if not feature_names or len(feature_names) != matrix.shape[1] or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature names must match the feature columns")
    seen_pairs: set[tuple[str, str]] = set()
    for source, target in zip(source_ids, target_ids):
        if (not isinstance(source, str) or not source
                or not isinstance(target, str) or not target
                or (source, target) in seen_pairs):
            raise ValueError("candidate pairs must have unique, nonempty IDs")
        seen_pairs.add((source, target))
    if not set(source_ids) <= set(fold_by_source) or not set(source_ids) <= set(component_by_source):
        raise ValueError("every pair source needs a fold and graph component")
    folds = np.asarray([fold_by_source[source] for source in source_ids], dtype=np.int32)
    unique_folds = sorted(set(folds))
    if unique_folds != list(range(len(unique_folds))) or len(unique_folds) < 3:
        raise ValueError("nested mining needs at least three contiguous folds")
    components = np.asarray([component_by_source[source] for source in source_ids], dtype=object)
    component_fold = {}
    for component, fold in zip(components, folds):
        previous = component_fold.setdefault(component, int(fold))
        if previous != fold:
            raise ValueError("a graph component crosses validation folds")

    fixed_config = replace(model_config, early_stopping_rounds=0)
    raw_scores = np.full(count, np.nan, dtype=np.float64)
    models: list[lgb.Booster] = []
    mined_by_fold: dict[int, tuple[int, ...]] = {}
    diagnostics: dict[int, dict[str, float | int | None]] = {}
    for outer_fold in unique_folds:
        eligible = folds != outer_fold
        inner_scores = np.full(count, np.nan, dtype=np.float64)
        for inner_fold in unique_folds:
            if inner_fold == outer_fold:
                continue
            inner_valid = folds == inner_fold
            inner_train = eligible & ~inner_valid
            result = train_pair_baseline(
                matrix[inner_train], targets[inner_train],
                matrix[inner_valid], targets[inner_valid], feature_names,
                components[inner_train], components[inner_valid], fixed_config,
            )
            inner_scores[inner_valid] = result.validation_scores
        if not np.isfinite(inner_scores[eligible]).all():
            raise ValueError("inner mining left eligible pairs unscored")
        eligible_indices = np.flatnonzero(eligible)
        local_mined = mine_hard_negatives(
            [source_ids[index] for index in eligible_indices],
            [target_ids[index] for index in eligible_indices],
            targets[eligible], inner_scores[eligible], mining_config,
        )
        mined = tuple(int(eligible_indices[index]) for index in local_mined)
        weights = np.ones(count, dtype=np.float64)
        weights[list(mined)] += mining_config.extra_weight
        outer_valid = ~eligible
        result = train_pair_baseline(
            matrix[eligible], targets[eligible],
            matrix[outer_valid], targets[outer_valid], feature_names,
            components[eligible], components[outer_valid], fixed_config,
            train_weights=weights[eligible],
        )
        raw_scores[outer_valid] = result.validation_scores
        models.append(result.model)
        mined_by_fold[outer_fold] = mined
        diagnostics[outer_fold] = dict(result.pair_diagnostics,
                                       mined_negatives=len(mined),
                                       eligible_training_pairs=int(eligible.sum()))
    if not np.isfinite(raw_scores).all():
        raise ValueError("outer OOF scoring left pairs unscored")
    return HardNegativeOOFResult(raw_scores, folds, tuple(models), mined_by_fold,
                                 diagnostics, mining_config)


@dataclass(frozen=True)
class HardNegativeComparison:
    baseline: ThresholdSearchResult
    mined: ThresholdSearchResult
    crossfit_macro_delta: float
    worst_fold_delta: float
    improves_crossfit_macro: bool


def compare_raw_oof_hard_negatives(
    source_ids: Sequence[str], target_ids: Sequence[str],
    pair_folds: Sequence[int], baseline_scores: Sequence[float],
    mined_scores: Sequence[float], truth: Mapping[str, frozenset[str]],
    entity_folds: Mapping[str, int], countries: Mapping[str, str],
    grid: ThresholdGrid,
) -> HardNegativeComparison:
    """Re-run Phase 9/10 on both raw OOF pathways with an identical grid.

    This is the raw-score acceptance comparison. If calibration was retained,
    its nested OOF pathway must be regenerated and compared separately before
    accepting mining; an old calibrator cannot be reused after model changes.
    """
    baseline_entities = assemble_oof_entities(
        source_ids, target_ids, baseline_scores, pair_folds,
        truth, entity_folds, countries,
    )
    mined_entities = assemble_oof_entities(
        source_ids, target_ids, mined_scores, pair_folds,
        truth, entity_folds, countries,
    )
    baseline = search_thresholds(baseline_entities, grid, "raw")
    mined = search_thresholds(mined_entities, grid, "raw")
    baseline_metrics = baseline.policies[baseline.selected_policy].crossfit_metrics
    mined_metrics = mined.policies[mined.selected_policy].crossfit_metrics
    delta = mined_metrics.overall_macro_f05 - baseline_metrics.overall_macro_f05
    return HardNegativeComparison(
        baseline, mined, delta,
        mined_metrics.worst_fold_f05 - baseline_metrics.worst_fold_f05,
        delta > 0,
    )
