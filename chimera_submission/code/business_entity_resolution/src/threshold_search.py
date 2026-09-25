"""Leakage-safe Phase 10 search over deterministic match-set policies.

Only training ground truth and out-of-fold pair scores belong in this module.
Full-data threshold fitting is reported separately from outer-fold evaluation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from numbers import Real
from pathlib import Path
from statistics import mean, pstdev
from typing import Mapping, Sequence

from .entity_decision import entity_f05
from .threshold_policy import FrozenDecisionConfig, PolicyName, _threshold


@dataclass(frozen=True)
class OOFEntity:
    source_id: str
    candidate_ids: tuple[str, ...]
    scores: tuple[float, ...]
    truth: frozenset[str]
    fold: int
    country: str


def assemble_oof_entities(
    source_ids: Sequence[str], target_ids: Sequence[str], scores: Sequence[float],
    pair_folds: Sequence[int], truth: Mapping[str, frozenset[str]],
    entity_folds: Mapping[str, int], countries: Mapping[str, str],
) -> tuple[OOFEntity, ...]:
    """Include every training S1 entity, including those with zero candidates."""
    if not (len(source_ids) == len(target_ids) == len(scores) == len(pair_folds)):
        raise ValueError("OOF pair arrays must align")
    if set(truth) != set(entity_folds) or set(truth) != set(countries):
        raise ValueError("truth, entity folds and countries must cover the same S1 IDs")
    grouped: dict[str, list[tuple[str, float]]] = {source: [] for source in truth}
    seen: set[tuple[str, str]] = set()
    for source, target, score, fold in zip(source_ids, target_ids, scores, pair_folds):
        if source not in truth or (source, target) in seen:
            raise ValueError("unknown or duplicate OOF candidate pair")
        if fold != entity_folds[source]:
            raise ValueError("OOF pair fold differs from its entity fold")
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("OOF scores must be finite probabilities")
        seen.add((source, target))
        grouped[source].append((target, float(score)))
    return tuple(
        OOFEntity(source, tuple(target for target, _ in grouped[source]),
                  tuple(score for _, score in grouped[source]), truth[source],
                  entity_folds[source], countries[source])
        for source in sorted(truth)
    )


@dataclass(frozen=True)
class ThresholdGrid:
    pair: tuple[float, ...]
    entity: tuple[float, ...]
    gap: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("pair", "entity", "gap"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} grid must be nonempty and unique")
            for value in values:
                _threshold(value, name)


@dataclass(frozen=True)
class PolicyMetrics:
    entities: int
    overall_macro_f05: float
    fold_f05: dict[int, float]
    fold_mean_f05: float
    fold_std_f05: float
    worst_fold_f05: float
    singleton_precision: float | None
    false_positive_singleton_rate: float | None
    average_predicted_matches: float
    empty_percent: float
    top1_percent: float
    multi_percent: float
    country_f05: dict[str, float]


def evaluate_predictions(
    entities: Sequence[OOFEntity], predictions: Mapping[str, Sequence[str]],
) -> PolicyMetrics:
    """Singleton precision is correct empty / all empty predictions.

    False-positive singleton rate is nonempty predictions / true singletons.
    Undefined denominators are represented as None, not invented zeros.
    """
    if not entities or len({entity.source_id for entity in entities}) != len(entities):
        raise ValueError("evaluation needs distinct S1 entities")
    if set(predictions) != {entity.source_id for entity in entities}:
        raise ValueError("predictions must cover exactly the evaluated S1 entities")
    by_fold: dict[int, list[float]] = {}
    by_country: dict[str, list[float]] = {}
    scores: list[float] = []
    sizes = [0, 0, 0]
    true_singletons = predicted_empty = correct_empty = false_positive_singletons = 0
    for entity in entities:
        predicted = tuple(predictions[entity.source_id])
        if len(set(predicted)) != len(predicted) or not set(predicted) <= set(entity.candidate_ids):
            raise ValueError("predictions must be unique and inside the candidate set")
        score = entity_f05(entity.truth, set(predicted))
        scores.append(score)
        by_fold.setdefault(entity.fold, []).append(score)
        by_country.setdefault(entity.country, []).append(score)
        sizes[min(len(predicted), 2)] += 1
        if not predicted:
            predicted_empty += 1
            correct_empty += not entity.truth
        if not entity.truth:
            true_singletons += 1
            false_positive_singletons += bool(predicted)
    fold_scores = {fold: mean(values) for fold, values in sorted(by_fold.items())}
    n = len(entities)
    return PolicyMetrics(
        n, mean(scores), fold_scores, mean(fold_scores.values()),
        pstdev(fold_scores.values()), min(fold_scores.values()),
        correct_empty / predicted_empty if predicted_empty else None,
        false_positive_singletons / true_singletons if true_singletons else None,
        sum(len(predictions[entity.source_id]) for entity in entities) / n,
        *(100 * count / n for count in sizes),
        {country: mean(values) for country, values in sorted(by_country.items())},
    )


def _configs(policy: PolicyName, grid: ThresholdGrid, score_path: str):
    if policy == "A":
        dimensions = product(grid.pair)
    elif policy == "B":
        dimensions = product(grid.pair, grid.entity)
    else:
        dimensions = product(grid.pair, grid.entity, grid.gap)
    for values in dimensions:
        yield FrozenDecisionConfig(policy, values[0], values[1] if len(values) > 1 else None,
                                   values[2] if len(values) > 2 else None, score_path)


def _predict(entities: Sequence[OOFEntity], config: FrozenDecisionConfig) -> dict[str, list[str]]:
    return {entity.source_id: config.predict(entity.candidate_ids, entity.scores)
            for entity in entities}


def _fit(entities: Sequence[OOFEntity], policy: PolicyName,
         grid: ThresholdGrid, score_path: str) -> FrozenDecisionConfig:
    best_config = None
    best_key = None
    for config in _configs(policy, grid, score_path):
        metrics = evaluate_predictions(entities, _predict(entities, config))
        # Maximize training macro, then worst fold, then minimize dispersion.
        # The final threshold tuple breaks exact ties deterministically.
        key = (metrics.overall_macro_f05, metrics.worst_fold_f05,
               -metrics.fold_std_f05, -config.pair_threshold,
               -(config.entity_threshold or 0), -(config.gap_threshold or 0))
        if best_key is None or key > best_key:
            best_config, best_key = config, key
    assert best_config is not None
    return best_config


@dataclass(frozen=True)
class PolicySearchResult:
    policy: PolicyName
    crossfit_metrics: PolicyMetrics
    fold_configs: dict[int, FrozenDecisionConfig]
    frozen_config: FrozenDecisionConfig
    exploratory_all_oof_metrics: PolicyMetrics


@dataclass(frozen=True)
class ThresholdSearchResult:
    selected_policy: PolicyName
    policies: dict[str, PolicySearchResult]
    grid: ThresholdGrid
    selection_rule: str
    score_source: str = "leakage-safe training OOF pair predictions"

    @property
    def frozen_config(self) -> FrozenDecisionConfig:
        return self.policies[self.selected_policy].frozen_config


def search_thresholds(entities: Sequence[OOFEntity], grid: ThresholdGrid,
                      score_path: str) -> ThresholdSearchResult:
    """Nested fold selection: each heldout fold's thresholds use other folds only.

    Family selection prioritizes worst heldout fold, then overall heldout macro,
    then lower dispersion, then simpler family A > B > C on exact ties.
    """
    if score_path not in ("raw", "calibrated"):
        raise ValueError("score path must be raw or calibrated")
    if not entities or len({entity.source_id for entity in entities}) != len(entities):
        raise ValueError("search needs distinct OOF S1 entities")
    folds = sorted({entity.fold for entity in entities})
    if len(folds) < 2:
        raise ValueError("cross-fitted threshold selection requires at least two folds")
    results: dict[str, PolicySearchResult] = {}
    for policy in ("A", "B", "C"):
        predictions: dict[str, list[str]] = {}
        fold_configs: dict[int, FrozenDecisionConfig] = {}
        for fold in folds:
            train = [entity for entity in entities if entity.fold != fold]
            heldout = [entity for entity in entities if entity.fold == fold]
            config = _fit(train, policy, grid, score_path)
            fold_configs[fold] = config
            predictions.update(_predict(heldout, config))
        crossfit = evaluate_predictions(entities, predictions)
        frozen = _fit(entities, policy, grid, score_path)
        exploratory = evaluate_predictions(entities, _predict(entities, frozen))
        results[policy] = PolicySearchResult(policy, crossfit, fold_configs,
                                             frozen, exploratory)
    def robust_key(policy: str) -> tuple[float, float, float, int]:
        metrics = results[policy].crossfit_metrics
        return (metrics.worst_fold_f05, metrics.overall_macro_f05,
                -metrics.fold_std_f05, -("A", "B", "C").index(policy))
    selected = max(results, key=robust_key)
    return ThresholdSearchResult(selected, results, grid,
        "worst heldout fold, overall heldout macro F0.5, lower fold std, simpler family")


def threshold_sensitivity(entities: Sequence[OOFEntity], config: FrozenDecisionConfig,
                          delta: float) -> dict[str, PolicyMetrics]:
    """Evaluate caller-supplied +/- threshold perturbations on labeled OOF data."""
    if isinstance(delta, bool) or not isinstance(delta, Real) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    result = {"baseline": evaluate_predictions(entities, _predict(entities, config))}
    for name in ("pair_threshold", "entity_threshold", "gap_threshold"):
        value = getattr(config, name)
        if value is None:
            continue
        for direction, suffix in ((-1, "minus"), (1, "plus")):
            altered = min(1.0, max(0.0, value + direction * delta))
            values = asdict(config)
            values[name] = altered
            variant = FrozenDecisionConfig(**values)
            result[f"{name}_{suffix}"] = evaluate_predictions(
                entities, _predict(entities, variant))
    return result


def leave_one_country_out(entities: Sequence[OOFEntity], policy: PolicyName,
                          grid: ThresholdGrid, score_path: str,
                          min_entities: int) -> dict[str, dict[str, object]]:
    """Fit on other countries, evaluate eligible heldout country; no test data."""
    if min_entities < 1:
        raise ValueError("min_entities must be positive")
    report = {}
    for country in sorted({entity.country for entity in entities}):
        heldout = [entity for entity in entities if entity.country == country]
        train = [entity for entity in entities if entity.country != country]
        if len(heldout) < min_entities or not train:
            continue
        config = _fit(train, policy, grid, score_path)
        report[country] = {"heldout_entities": len(heldout),
                           "trained_config": asdict(config),
                           "heldout_metrics": asdict(evaluate_predictions(
                               heldout, _predict(heldout, config)))}
    return report


def save_search_report(path: str | Path, result: ThresholdSearchResult,
                       sensitivity: Mapping[str, PolicyMetrics],
                       country_shift: Mapping[str, object]) -> None:
    """Write a reproducible OOF-only search report without overwriting a run."""
    fold_configs = result.policies[result.selected_policy].fold_configs
    dispersion = {}
    for name in ("pair_threshold", "entity_threshold", "gap_threshold"):
        values = [getattr(config, name) for config in fold_configs.values()]
        if all(value is not None for value in values):
            dispersion[name] = {"mean": mean(values), "std": pstdev(values),
                                "min": min(values), "max": max(values)}
    sensitivity_scores = [metrics.overall_macro_f05 for metrics in sensitivity.values()]
    country_scores = [item["heldout_metrics"]["overall_macro_f05"]
                      for item in country_shift.values()]
    payload = {"search": asdict(result),
               "threshold_sensitivity": {key: asdict(value) for key, value in sensitivity.items()},
               "threshold_dispersion_across_folds": dispersion,
               "sensitivity_overall_macro_range": {
                   "min": min(sensitivity_scores), "max": max(sensitivity_scores)}
                   if sensitivity_scores else None,
               "leave_one_country_out": country_shift,
               "country_shift_macro_summary": {
                   "eligible_countries": len(country_scores),
                   "mean": mean(country_scores), "std": pstdev(country_scores),
                   "worst": min(country_scores)} if country_scores else None}
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
