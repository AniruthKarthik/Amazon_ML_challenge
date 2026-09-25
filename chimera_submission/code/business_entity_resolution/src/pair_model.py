"""Phase 7 component-aware folds and a deterministic LightGBM pair baseline.

Pair-level diagnostics are not ranking-failure or entity-level F₀.₅ estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import lightgbm as lgb
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold


def source_component_ids(components: Iterable[Iterable[str]],
                         source_ids: Iterable[str]) -> dict[str, int]:
    """Map every S1 entity to its Phase 1 truth-graph component."""
    ids = sorted(source_ids)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("source IDs must be nonempty and unique")
    if any(not identifier.startswith("S1-") for identifier in ids):
        raise ValueError("fold assignment accepts S1 IDs only")
    wanted = set(ids)
    component_by_source = {}
    for component_index, component in enumerate(components):
        for identifier in component:
            if identifier not in wanted:
                continue
            if identifier in component_by_source:
                raise ValueError(f"S1 entity appears in multiple components: {identifier}")
            component_by_source[identifier] = component_index
    if set(component_by_source) != wanted:
        raise ValueError("components do not cover every S1 entity")
    return component_by_source


def assign_component_folds(components: Iterable[Iterable[str]],
                           source_ids: Iterable[str],
                           n_splits: int = 5) -> dict[str, int]:
    """Assign every S1 entity to exactly one Phase 1 truth-graph component fold."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    ids = sorted(source_ids)
    component_by_source = source_component_ids(components, ids)
    groups = np.asarray([component_by_source[identifier] for identifier in ids])
    if len(set(groups)) < n_splits:
        raise ValueError("fewer graph components than requested folds")
    fold_by_source = {}
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, valid_indices) in enumerate(
        splitter.split(np.zeros(len(ids)), groups=groups)
    ):
        for index in valid_indices:
            fold_by_source[ids[index]] = fold
    if len(fold_by_source) != len(ids):
        raise AssertionError("incomplete fold assignment")
    return fold_by_source


@dataclass(frozen=True)
class BaselineConfig:
    num_boost_round: int = 500
    early_stopping_rounds: int = 50
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    num_threads: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if min(self.num_boost_round, self.num_leaves,
               self.min_data_in_leaf, self.num_threads) < 1:
            raise ValueError("LightGBM counts must be positive")
        if self.early_stopping_rounds < 0:
            raise ValueError("early_stopping_rounds must be nonnegative")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")


@dataclass
class PairBaselineResult:
    model: lgb.Booster
    validation_scores: np.ndarray
    pair_diagnostics: dict[str, float | int | None]
    evaluation_history: dict[str, dict[str, list[float]]]
    feature_names: tuple[str, ...]


def _validate_matrix(values: np.ndarray, labels: np.ndarray,
                     feature_names: tuple[str, ...], name: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    targets = np.asarray(labels)
    if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[1] != len(feature_names):
        raise ValueError(f"{name} feature matrix has an invalid shape")
    if targets.ndim != 1 or len(targets) != len(matrix):
        raise ValueError(f"{name} label vector has an invalid shape")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} feature matrix contains NaN or infinity")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError(f"{name} labels must be binary")
    return matrix, targets.astype(np.int8, copy=False)


def _pair_diagnostics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int | None]:
    predicted = scores >= 0.5
    positives = int(np.count_nonzero(labels))
    return {
        "pairs": int(len(labels)),
        "positives": positives,
        "predicted_positives_at_0_5": int(np.count_nonzero(predicted)),
        "precision_at_0_5": float(precision_score(labels, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if 0 < positives < len(labels) else None,
    }


def train_pair_baseline(
    train_features: np.ndarray, train_labels: np.ndarray,
    valid_features: np.ndarray, valid_labels: np.ndarray,
    feature_names: tuple[str, ...],
    train_components: Iterable[str | int],
    valid_components: Iterable[str | int],
    config: BaselineConfig = BaselineConfig(),
    train_weights: np.ndarray | None = None,
) -> PairBaselineResult:
    """Fit one component-disjoint fold; no full-data or test-set training here."""
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature names must be nonempty and unique")
    train_x, train_y = _validate_matrix(
        train_features, train_labels, feature_names, "train"
    )
    valid_x, valid_y = _validate_matrix(
        valid_features, valid_labels, feature_names, "validation"
    )
    train_groups, valid_groups = tuple(train_components), tuple(valid_components)
    if len(train_groups) != len(train_x) or len(valid_groups) != len(valid_x):
        raise ValueError("component IDs must align with pair rows")
    if set(train_groups) & set(valid_groups):
        raise ValueError("training and validation share a graph component")
    if len(np.unique(train_y)) != 2:
        raise ValueError("training fold needs both positive and negative pairs")
    weights = None
    if train_weights is not None:
        weights = np.asarray(train_weights, dtype=np.float64)
        if (weights.ndim != 1 or len(weights) != len(train_x)
                or not np.isfinite(weights).all() or np.any(weights <= 0)):
            raise ValueError("training weights must be aligned, finite and positive")

    train_data = lgb.Dataset(
        train_x, label=train_y, weight=weights,
        feature_name=list(feature_names), free_raw_data=True,
    )
    valid_data = lgb.Dataset(
        valid_x, label=valid_y, feature_name=list(feature_names),
        reference=train_data, free_raw_data=True,
    )
    history: dict[str, dict[str, list[float]]] = {}
    callbacks = [lgb.record_evaluation(history)]
    if config.early_stopping_rounds:
        callbacks.append(lgb.early_stopping(
            config.early_stopping_rounds, verbose=False
        ))
    model = lgb.train(
        {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "min_data_in_leaf": config.min_data_in_leaf,
            "num_threads": config.num_threads,
            "seed": config.seed,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        train_data,
        num_boost_round=config.num_boost_round,
        valid_sets=[valid_data],
        valid_names=["validation"],
        callbacks=callbacks,
    )
    scores = np.asarray(
        model.predict(valid_x, num_iteration=model.best_iteration or None,
                      num_threads=config.num_threads),
        dtype=np.float64,
    )
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("LightGBM produced invalid validation probabilities")
    return PairBaselineResult(
        model=model, validation_scores=scores,
        pair_diagnostics=_pair_diagnostics(valid_y, scores),
        evaluation_history=history, feature_names=feature_names,
    )
