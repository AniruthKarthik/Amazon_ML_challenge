"""CPU-friendly Phase 12 entity cardinality model and OOF set decisions.

The classifier sees only entity aggregations of pair OOF scores. For each outer
fold, its training labels and the multi-match threshold are learned from the
other folds; an inner OOF loop supplies threshold-selection class predictions.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Literal, Mapping, Sequence

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .entity_decision import summarize_entity_scores
from .threshold_policy import FrozenDecisionConfig, _ordered_candidates, _threshold
from .threshold_search import OOFEntity, PolicyMetrics, evaluate_predictions


MetaClass = Literal["ZERO", "ONE", "MANY"]
META_CLASSES = ("ZERO", "ONE", "MANY")
META_FEATURE_NAMES = (
    "candidate_count", "max_score", "second_score", "score_gap",
    "mean_score", "score_std", "score_concentration",
)
META_SCHEMA_VERSION = 1


def predict_meta_match_set(
    predicted_class: MetaClass, candidate_ids: Sequence[str],
    scores: Sequence[float], multi_pair_threshold: float,
) -> list[str]:
    """ZERO -> empty; ONE -> top-1; MANY -> thresholded set or top-1 fallback.

    MANY never forces a second candidate. Scores equal to the threshold pass.
    No candidate always yields empty. IDs are unique, ordered by score then ID.
    """
    threshold = _threshold(multi_pair_threshold, "multi_pair_threshold")
    if predicted_class not in META_CLASSES:
        raise ValueError("predicted class must be ZERO, ONE or MANY")
    ordered = _ordered_candidates(candidate_ids, scores)
    if not ordered or predicted_class == "ZERO":
        return []
    if predicted_class == "ONE":
        return [ordered[0][0]]
    selected = [identifier for identifier, score in ordered if score >= threshold]
    return selected or [ordered[0][0]]


def meta_feature_vector(
    source_id: str, candidate_ids: Sequence[str], scores: Sequence[float],
) -> np.ndarray:
    """Finite label-free aggregations; usable unchanged at test inference."""
    summary = summarize_entity_scores(source_id, _ordered_candidates(candidate_ids, scores))
    return np.asarray((
        summary.candidate_count, summary.max_score, summary.second_score,
        summary.score_gap, summary.mean_score, summary.score_std,
        summary.score_concentration,
    ), dtype=np.float64)


def entity_meta_features(entity: OOFEntity) -> np.ndarray:
    """Extract only pair-score features, never truth, country or fold labels."""
    return meta_feature_vector(entity.source_id, entity.candidate_ids, entity.scores)


def _target_class(entity: OOFEntity) -> str:
    if not entity.truth:
        return "ZERO"
    return "ONE" if len(entity.truth) == 1 else "MANY"


@dataclass(frozen=True)
class MetaModelConfig:
    max_iter: int = 200
    regularization_c: float = 1.0
    num_threads: int = 1

    def __post_init__(self) -> None:
        if (isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int)
                or self.max_iter < 1 or isinstance(self.num_threads, bool)
                or not isinstance(self.num_threads, int) or self.num_threads < 1):
            raise ValueError("meta-model iteration and thread counts must be positive")
        if (isinstance(self.regularization_c, bool)
                or not isinstance(self.regularization_c, Real)
                or not math.isfinite(self.regularization_c)
                or self.regularization_c <= 0):
            raise ValueError("meta-model regularization C must be positive and finite")


def _fit_classifier(features: np.ndarray, targets: np.ndarray,
                    config: MetaModelConfig):
    if not len(features):
        raise ValueError("meta-model training fold is empty")
    if len(set(targets)) == 1:
        classifier = DummyClassifier(strategy="constant", constant=targets[0])
    else:
        classifier = make_pipeline(
            StandardScaler(), LogisticRegression(
                C=config.regularization_c, max_iter=config.max_iter,
                solver="lbfgs", random_state=42,
            ),
        )
    with threadpool_limits(limits=config.num_threads):
        classifier.fit(features, targets)
    return classifier


def _predict_classes(classifier, features: np.ndarray,
                     num_threads: int) -> tuple[str, ...]:
    with threadpool_limits(limits=num_threads):
        probabilities = classifier.predict_proba(features)
    aligned = np.zeros((len(features), len(META_CLASSES)), dtype=np.float64)
    for index, name in enumerate(classifier.classes_):
        aligned[:, META_CLASSES.index(name)] = probabilities[:, index]
    # On an exact posterior tie, prefer ZERO then ONE over MANY for precision.
    return tuple(META_CLASSES[index] for index in np.argmax(aligned, axis=1))


def _features_and_targets(entities: Sequence[OOFEntity]) -> tuple[np.ndarray, np.ndarray]:
    return (np.stack([entity_meta_features(entity) for entity in entities]),
            np.asarray([_target_class(entity) for entity in entities]))


def _inner_oof_classes(
    entities: Sequence[OOFEntity], features: np.ndarray, targets: np.ndarray,
    eligible: np.ndarray, config: MetaModelConfig,
) -> tuple[str, ...]:
    classes = [""] * len(entities)
    folds = sorted({entity.fold for index, entity in enumerate(entities) if eligible[index]})
    if len(folds) < 2:
        raise ValueError("inner meta OOF tuning needs at least two training folds")
    for fold in folds:
        validation = np.asarray([entity.fold == fold for entity in entities]) & eligible
        training = eligible & ~validation
        classifier = _fit_classifier(features[training], targets[training], config)
        predicted = _predict_classes(classifier, features[validation], config.num_threads)
        for index, name in zip(np.flatnonzero(validation), predicted):
            classes[index] = name
    if any(not classes[index] for index in np.flatnonzero(eligible)):
        raise ValueError("inner meta OOF left training entities unpredicted")
    return tuple(classes)


def _choose_threshold(
    entities: Sequence[OOFEntity], classes: Sequence[str],
    threshold_grid: Sequence[float],
) -> float:
    if not threshold_grid:
        raise ValueError("multi threshold grid cannot be empty")
    if len(classes) != len(entities):
        raise ValueError("class predictions must align with entities")
    best_threshold = None
    best_key = None
    for threshold in threshold_grid:
        _threshold(threshold, "multi_pair_threshold")
    for threshold in sorted(set(threshold_grid)):
        predictions = {
            entity.source_id: predict_meta_match_set(
                predicted_class, entity.candidate_ids, entity.scores, threshold,
            )
            for entity, predicted_class in zip(entities, classes)
        }
        metrics = evaluate_predictions(entities, predictions)
        # Macro F0.5 first; worst fold next; higher threshold on exact ties
        # avoids gratuitous additional matches.
        key = (metrics.overall_macro_f05, metrics.worst_fold_f05, threshold)
        if best_key is None or key > best_key:
            best_key, best_threshold = key, float(threshold)
    assert best_threshold is not None
    return best_threshold


@dataclass(frozen=True)
class MetaOOFResult:
    predicted_classes: dict[str, str]
    predictions: dict[str, tuple[str, ...]]
    fold_thresholds: dict[int, float]
    crossfit_metrics: PolicyMetrics
    frozen_multi_pair_threshold: float
    final_model: object
    threshold_grid: tuple[float, ...]
    model_config: MetaModelConfig
    exploratory_all_oof_metrics: PolicyMetrics


def generate_meta_oof_decisions(
    entities: Sequence[OOFEntity], threshold_grid: Sequence[float],
    config: MetaModelConfig = MetaModelConfig(),
) -> MetaOOFResult:
    """Cross-fit class model and independently tune the MANY threshold.

    Outer-fold labels never train that fold's meta classifier or select its
    threshold. The final model and threshold use all training OOF entities and
    their score is explicitly exploratory, not heldout validation.
    """
    if not entities or len({entity.source_id for entity in entities}) != len(entities):
        raise ValueError("meta OOF needs distinct nonempty S1 entities")
    if not threshold_grid:
        raise ValueError("multi threshold grid cannot be empty")
    for threshold in threshold_grid:
        _threshold(threshold, "multi_pair_threshold")
    grid = tuple(sorted(set(threshold_grid)))
    folds = sorted({entity.fold for entity in entities})
    if len(folds) < 3:
        raise ValueError("nested meta threshold tuning needs at least three folds")
    features, targets = _features_and_targets(entities)
    predicted_classes: dict[str, str] = {}
    predictions: dict[str, tuple[str, ...]] = {}
    fold_thresholds: dict[int, float] = {}
    for outer_fold in folds:
        eligible = np.asarray([entity.fold != outer_fold for entity in entities])
        inner_classes = _inner_oof_classes(entities, features, targets, eligible, config)
        training_entities = [entity for entity, included in zip(entities, eligible) if included]
        training_classes = [name for name, included in zip(inner_classes, eligible) if included]
        threshold = _choose_threshold(training_entities, training_classes, grid)
        fold_thresholds[outer_fold] = threshold
        classifier = _fit_classifier(features[eligible], targets[eligible], config)
        heldout = ~eligible
        heldout_classes = _predict_classes(classifier, features[heldout], config.num_threads)
        for entity, name in zip(
            (entity for entity, include in zip(entities, heldout) if include),
            heldout_classes,
        ):
            predicted_classes[entity.source_id] = name
            predictions[entity.source_id] = tuple(predict_meta_match_set(
                name, entity.candidate_ids, entity.scores, threshold,
            ))
    crossfit_metrics = evaluate_predictions(entities, predictions)
    all_mask = np.ones(len(entities), dtype=bool)
    all_oof_classes = _inner_oof_classes(entities, features, targets, all_mask, config)
    frozen_threshold = _choose_threshold(entities, all_oof_classes, grid)
    final_model = _fit_classifier(features, targets, config)
    final_classes = _predict_classes(final_model, features, config.num_threads)
    final_predictions = {
        entity.source_id: predict_meta_match_set(name, entity.candidate_ids,
                                                 entity.scores, frozen_threshold)
        for entity, name in zip(entities, final_classes)
    }
    return MetaOOFResult(
        predicted_classes, predictions, fold_thresholds, crossfit_metrics,
        frozen_threshold, final_model, grid, config,
        evaluate_predictions(entities, final_predictions),
    )


@dataclass(frozen=True)
class MetaComparison:
    deterministic: PolicyMetrics
    meta: PolicyMetrics
    macro_delta: float
    worst_fold_delta: float
    fold_std_delta: float
    keep_meta: bool


def compare_meta_to_phase10(
    entities: Sequence[OOFEntity], meta_result: MetaOOFResult,
    phase10_fold_configs: Mapping[int, FrozenDecisionConfig],
    max_worst_fold_drop: float, max_fold_std_increase: float,
) -> MetaComparison:
    """Compare identical folds; caller explicitly supplies robustness limits."""
    for value, name in ((max_worst_fold_drop, "max_worst_fold_drop"),
                        (max_fold_std_increase, "max_fold_std_increase")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if set(phase10_fold_configs) != {entity.fold for entity in entities}:
        raise ValueError("Phase 10 fold configurations must match meta OOF folds")
    if len({config.policy for config in phase10_fold_configs.values()}) != 1 or len({
        config.score_path for config in phase10_fold_configs.values()
    }) != 1:
        raise ValueError("Phase 10 fold configurations need one locked policy and score path")
    if set(meta_result.predictions) != {entity.source_id for entity in entities}:
        raise ValueError("meta OOF predictions must cover the same S1 entities")
    deterministic_predictions = {
        entity.source_id: phase10_fold_configs[entity.fold].predict(
            entity.candidate_ids, entity.scores,
        ) for entity in entities
    }
    baseline = evaluate_predictions(entities, deterministic_predictions)
    meta = evaluate_predictions(entities, meta_result.predictions)
    macro_delta = meta.overall_macro_f05 - baseline.overall_macro_f05
    worst_delta = meta.worst_fold_f05 - baseline.worst_fold_f05
    std_delta = meta.fold_std_f05 - baseline.fold_std_f05
    return MetaComparison(
        baseline, meta, macro_delta, worst_delta, std_delta,
        macro_delta > 0 and worst_delta >= -max_worst_fold_drop
        and std_delta <= max_fold_std_increase,
    )


def predict_frozen_meta(
    model: object, candidate_ids: Sequence[str], scores: Sequence[float],
    multi_pair_threshold: float, num_threads: int = 1,
) -> list[str]:
    """Apply the unchanged Phase 12 rule at later inference."""
    features = meta_feature_vector("inference", candidate_ids, scores).reshape(1, -1)
    name = _predict_classes(model, features, num_threads)[0]
    return predict_meta_match_set(name, candidate_ids, scores, multi_pair_threshold)


def predict_frozen_meta_batch(
    model: object,
    entities: Sequence[tuple[str, Sequence[str], Sequence[float]]],
    multi_pair_threshold: float,
    num_threads: int = 1,
) -> list[list[str]]:
    """Batch unchanged frozen Phase 12 decisions for CPU-efficient inference."""
    _threshold(multi_pair_threshold, "multi_pair_threshold")
    if not entities:
        return []
    features = np.stack([
        meta_feature_vector(source_id, candidate_ids, scores)
        for source_id, candidate_ids, scores in entities
    ])
    classes = _predict_classes(model, features, num_threads)
    return [
        predict_meta_match_set(name, candidate_ids, scores, multi_pair_threshold)
        for name, (_, candidate_ids, scores) in zip(classes, entities)
    ]


def save_meta_artifact(path: str | Path, result: MetaOOFResult) -> None:
    """Freeze the trained model and threshold in one non-overwriting artifact."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    payload = {
        "schema_version": META_SCHEMA_VERSION,
        "feature_names": META_FEATURE_NAMES,
        "multi_pair_threshold": result.frozen_multi_pair_threshold,
        "model_config": asdict(result.model_config),
        "threshold_grid": result.threshold_grid,
        "model": result.final_model,
    }
    with destination.open("xb") as handle:
        joblib.dump(payload, handle)


def load_meta_artifact(path: str | Path) -> tuple[object, float, MetaModelConfig]:
    """Load trusted, locally produced artifacts only; joblib uses pickle."""
    payload = joblib.load(path)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "feature_names", "multi_pair_threshold",
        "model_config", "threshold_grid", "model",
    }:
        raise ValueError("meta artifact schema mismatch")
    if payload["schema_version"] != META_SCHEMA_VERSION or tuple(payload["feature_names"]) != META_FEATURE_NAMES:
        raise ValueError("meta artifact version or features mismatch")
    threshold = _threshold(payload["multi_pair_threshold"], "multi_pair_threshold")
    if not hasattr(payload["model"], "predict_proba"):
        raise ValueError("meta artifact does not contain a probabilistic classifier")
    return payload["model"], threshold, MetaModelConfig(**payload["model_config"])


def save_meta_report(path: str | Path, result: MetaOOFResult,
                     comparison: MetaComparison) -> None:
    """Record the heldout comparison separately from the all-OOF fit."""
    payload = {
        "score_source": "training OOF pair scores only",
        "threshold_grid": result.threshold_grid,
        "fold_thresholds": result.fold_thresholds,
        "frozen_multi_pair_threshold": result.frozen_multi_pair_threshold,
        "model_config": asdict(result.model_config),
        "meta_crossfit_metrics": asdict(result.crossfit_metrics),
        "exploratory_all_oof_metrics": asdict(result.exploratory_all_oof_metrics),
        "phase10_comparison": asdict(comparison),
    }
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
