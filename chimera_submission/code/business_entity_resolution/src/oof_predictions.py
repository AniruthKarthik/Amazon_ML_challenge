"""Leakage-safe Phase 8 pair predictions and cross-fitted calibration."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import fbeta_score

from .pair_model import BaselineConfig, train_pair_baseline


@dataclass
class OOFResult:
    raw_scores: np.ndarray
    calibrated_scores: np.ndarray
    fold_ids: np.ndarray
    fold_models: tuple[lgb.Booster, ...]
    fold_diagnostics: tuple[dict[str, float | int | None], ...]
    calibration_models: tuple[IsotonicRegression, ...]
    final_calibrator: IsotonicRegression
    pair_f05_at_0_5: dict[str, float]


def _nested_calibrator(
    matrix: np.ndarray, labels: np.ndarray, folds: np.ndarray,
    components: np.ndarray, feature_names: tuple[str, ...],
    config: BaselineConfig, outer_fold: int,
) -> IsotonicRegression:
    """Fit on inner OOF scores from non-outer-fold entities only."""
    eligible = folds != outer_fold
    inner_scores = np.full(len(matrix), np.nan, dtype=np.float64)
    for inner_fold in sorted(set(folds[eligible])):
        validation = folds == inner_fold
        training = eligible & ~validation
        result = train_pair_baseline(
            matrix[training], labels[training],
            matrix[validation], labels[validation], feature_names,
            components[training], components[validation], config,
        )
        inner_scores[validation] = result.validation_scores
    if not np.isfinite(inner_scores[eligible]).all():
        raise ValueError("inner OOF calibration left scores unfilled")
    if len(np.unique(labels[eligible])) != 2:
        raise ValueError("calibration training folds need both classes")
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    calibrator.fit(inner_scores[eligible], labels[eligible])
    return calibrator


def generate_oof_predictions(
    features: np.ndarray, labels: np.ndarray,
    source_ids: Sequence[str],
    fold_by_source: Mapping[str, int],
    component_by_source: Mapping[str, str | int],
    feature_names: tuple[str, ...],
    config: BaselineConfig = BaselineConfig(),
) -> OOFResult:
    """Fit one LightGBM model per held-out component fold on supplied pairs."""
    matrix = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels)
    if matrix.ndim != 2 or not len(matrix) or len(source_ids) != len(matrix):
        raise ValueError("features and source IDs must have aligned nonempty rows")
    if targets.ndim != 1 or len(targets) != len(matrix):
        raise ValueError("labels must align with pair rows")
    if not np.isfinite(matrix).all() or not np.isin(targets, (0, 1)).all():
        raise ValueError("features must be finite and labels binary")
    used_sources = set(source_ids)
    if not used_sources <= fold_by_source.keys() or not used_sources <= component_by_source.keys():
        raise ValueError("every pair source must have a fold and graph component")
    folds = np.asarray([fold_by_source[identifier] for identifier in source_ids], dtype=np.int32)
    components = np.asarray(
        [component_by_source[identifier] for identifier in source_ids], dtype=object
    )
    unique_folds = sorted(set(folds))
    if unique_folds != list(range(len(unique_folds))) or len(unique_folds) < 3:
        raise ValueError("nested OOF calibration needs at least three contiguous folds")
    component_fold = {}
    for component, fold in zip(components, folds):
        previous = component_fold.setdefault(component, int(fold))
        if previous != fold:
            raise ValueError("a graph component crosses validation folds")

    # The held-out labels may be used for diagnostics but never for early
    # stopping or model selection on the scores assigned to that fold.
    oof_config = replace(config, early_stopping_rounds=0)
    raw_scores = np.full(len(matrix), np.nan, dtype=np.float64)
    models, diagnostics = [], []
    for fold in unique_folds:
        validation = folds == fold
        training = ~validation
        result = train_pair_baseline(
            matrix[training], targets[training],
            matrix[validation], targets[validation], feature_names,
            components[training], components[validation], oof_config,
        )
        raw_scores[validation] = result.validation_scores
        models.append(result.model)
        diagnostics.append(result.pair_diagnostics)
    if not np.isfinite(raw_scores).all():
        raise ValueError("OOF scoring left pair rows unfilled")
    calibrated = np.full(len(matrix), np.nan, dtype=np.float64)
    calibrators = []
    for fold in unique_folds:
        model = _nested_calibrator(
            matrix, targets, folds, components, feature_names,
            oof_config, fold,
        )
        validation = folds == fold
        calibrated[validation] = model.predict(raw_scores[validation])
        calibrators.append(model)
    if not np.isfinite(calibrated).all():
        raise ValueError("nested calibration left scores unfilled")
    final_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    final_calibrator.fit(raw_scores, targets)
    pair_f05 = {
        "raw_at_0_5": float(fbeta_score(
            targets, raw_scores >= 0.5, beta=0.5, zero_division=0
        )),
        "crossfit_calibrated_at_0_5": float(fbeta_score(
            targets, calibrated >= 0.5, beta=0.5, zero_division=0
        )),
    }
    return OOFResult(
        raw_scores=raw_scores, calibrated_scores=calibrated,
        fold_ids=folds, fold_models=tuple(models),
        fold_diagnostics=tuple(diagnostics),
        calibration_models=tuple(calibrators),
        final_calibrator=final_calibrator,
        pair_f05_at_0_5=pair_f05,
    )


def write_oof_artifact(path: str | Path, source_ids: Sequence[str],
                       target_ids: Sequence[str], result: OOFResult) -> None:
    """Write auditable pair scores without labels; never overwrite an artifact."""
    count = len(result.raw_scores)
    if not (len(source_ids) == len(target_ids) == len(result.calibrated_scores)
            == len(result.fold_ids) == count):
        raise ValueError("OOF artifact IDs and scores must align")
    previous_source = None
    targets_for_source: set[str] = set()
    for source_id, target_id, raw, calibrated in zip(
        source_ids, target_ids, result.raw_scores, result.calibrated_scores,
    ):
        if previous_source is not None and source_id < previous_source:
            raise ValueError("OOF artifact pairs must be source-sorted")
        if source_id != previous_source:
            targets_for_source.clear()
            previous_source = source_id
        if target_id in targets_for_source:
            raise ValueError("OOF artifact contains duplicate candidate pairs")
        targets_for_source.add(target_id)
        if not math.isfinite(raw) or not math.isfinite(calibrated):
            raise ValueError("OOF artifact contains non-finite scores")
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", "candidate_entity_id", "fold_id",
                         "raw_oof_score", "calibrated_oof_score"))
        for source_id, target_id, fold, raw, calibrated in zip(
            source_ids, target_ids, result.fold_ids,
            result.raw_scores, result.calibrated_scores,
        ):
            writer.writerow((source_id, target_id, int(fold),
                             format(float(raw), ".17g"),
                             format(float(calibrated), ".17g")))
