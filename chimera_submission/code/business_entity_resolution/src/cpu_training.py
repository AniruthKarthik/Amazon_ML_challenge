"""Bounded CPU training on explicit S1 samples; no test inputs or labels.

The sample is an exploratory OOF experiment, not a full-training estimate.
Each S1 is a separate truth-graph component because training truth requires
each S2/S3 target to have at most one S1 owner.
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

from .pair_features import PairFeatureExtractor
from .pair_model import BaselineConfig, train_pair_baseline
from .oof_ranking import evaluate_oof_ranking
from .pipeline_store import DiskCandidateStore, QueryCandidates
from .threshold_policy import save_frozen_config
from .threshold_search import (
    OOFEntity, ThresholdGrid, leave_one_country_out, save_search_report,
    search_thresholds, threshold_sensitivity,
)


@dataclass(frozen=True)
class CPUTrainingConfig:
    train_entities: int
    max_train_pairs: int = 1_000_000
    folds: int = 3
    grid_points: int = 3
    feature_max_features: int = 30_000
    seed: int = 42
    sensitivity_delta: float = 0.05
    country_min_entities: int = 100
    model: BaselineConfig = BaselineConfig(
        num_boost_round=100, early_stopping_rounds=0, num_threads=4,
    )

    def __post_init__(self) -> None:
        if min(self.train_entities, self.max_train_pairs, self.folds,
               self.grid_points, self.feature_max_features,
               self.country_min_entities) < 1 or self.folds < 3:
            raise ValueError("training limits must be positive and folds >= 3")
        if self.model.early_stopping_rounds:
            raise ValueError("OOF boosting rounds must not use heldout early stopping")


def select_source_sequences(source_count: int, count: int, seed: int) -> tuple[int, ...]:
    """Fixed-seed uniform S1 sample; the caller explicitly sets its size."""
    if source_count < 1 or not 1 <= count <= source_count:
        raise ValueError("requested training sample exceeds S1 population")
    return tuple(sorted(random.Random(seed).sample(range(source_count), count)))


def _folds_for_sequences(sequences: tuple[int, ...], folds: int) -> dict[int, int]:
    if len(sequences) < folds:
        raise ValueError("training sample needs at least one S1 per fold")
    result = {}
    splitter = GroupKFold(n_splits=folds)
    groups = np.asarray(sequences)
    for fold, (_, heldout) in enumerate(splitter.split(groups, groups=groups)):
        for index in heldout:
            result[sequences[index]] = fold
    return result


def _truth(connection: sqlite3.Connection, seq: int) -> frozenset[str]:
    return frozenset(row[0] for row in connection.execute(
        "SELECT t.entity_id FROM truth_indexed x JOIN targets t ON t.seq=x.target_seq "
        "WHERE x.source_seq=?", (seq,),
    ))


def _record_factory(store: DiskCandidateStore, sequences: tuple[int, ...]):
    def records():
        for seq in sequences:
            query = store.get_query(seq)
            yield query.source
            for candidate in query.candidates:
                yield query.targets[candidate.candidate_entity_id]
    return records


def _matrix(
    store: DiskCandidateStore, sequences: tuple[int, ...],
    extractor: PairFeatureExtractor, limit: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build a bounded numeric matrix; never retain record objects globally."""
    width = len(extractor.feature_names)
    capacity = min(limit, max(1, len(sequences) * store.max_candidates))
    features = np.empty((capacity, width), dtype=np.float32)
    labels = np.empty(capacity, dtype=np.int8)
    groups: list[str] = []
    used = 0
    for seq in sequences:
        query = store.get_query(seq)
        truth = _truth(store.connection, seq)
        source_id = query.source.raw.entity_id
        for candidate in query.candidates:
            if used == capacity:
                raise ValueError("candidate-pair limit exceeded; reduce --train-entities")
            row = extractor.features(
                query.source, query.targets[candidate.candidate_entity_id], candidate)
            features[used] = tuple(row.values())
            labels[used] = candidate.candidate_entity_id in truth
            groups.append(source_id)
            used += 1
    if not used:
        raise ValueError("training sample has no candidate pairs")
    return features[:used], labels[:used], tuple(groups)


def _score_queries(
    store: DiskCandidateStore, sequences: tuple[int, ...],
    extractor: PairFeatureExtractor, model: lgb.Booster,
    fold_by_seq: dict[int, int],
) -> list[OOFEntity]:
    entities = []
    for seq in sequences:
        query: QueryCandidates = store.get_query(seq)
        candidates = query.candidates
        if candidates:
            matrix = np.asarray([
                tuple(extractor.features(
                    query.source, query.targets[item.candidate_entity_id], item,
                ).values()) for item in candidates
            ], dtype=np.float32)
            scores = tuple(float(value) for value in model.predict(matrix))
        else:
            scores = ()
        entities.append(OOFEntity(
            query.source.raw.entity_id,
            tuple(item.candidate_entity_id for item in candidates), scores,
            _truth(store.connection, seq), fold_by_seq[seq],
            query.source.raw.country,
        ))
    return entities


def _quantiles(values: list[float], points: int) -> tuple[float, ...]:
    if not values:
        return (0.0,)
    levels = np.linspace(0, 1, points + 2)[1:-1]
    return tuple(sorted(set(float(value) for value in np.quantile(values, levels))))


def oof_threshold_grid(entities: list[OOFEntity], points: int) -> ThresholdGrid:
    """Derive search locations from training OOF scores, never test data."""
    pair = [score for entity in entities for score in entity.scores]
    top = [max(entity.scores) for entity in entities if entity.scores]
    gaps = []
    for entity in entities:
        if len(entity.scores) > 1:
            first, second = sorted(entity.scores, reverse=True)[:2]
            gaps.append(first - second)
    return ThresholdGrid(_quantiles(pair, points), _quantiles(top, points),
                         _quantiles(gaps, points))


def train_cpu_baseline(
    store: DiskCandidateStore, output_dir: str | Path,
    config: CPUTrainingConfig,
) -> dict[str, object]:
    """Cross-fit raw pair scores, tune policy, then fit a separate final model."""
    if store.connection.execute(
        "SELECT name FROM sqlite_master WHERE name='truth_indexed'"
    ).fetchone() is None:
        raise ValueError("CPU training requires the validated training truth store")
    output_dir = Path(output_dir)
    building = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists() or building.exists():
        raise FileExistsError(output_dir)
    sequences = select_source_sequences(store.source_count, config.train_entities,
                                        config.seed)
    fold_by_seq = _folds_for_sequences(sequences, config.folds)
    building.mkdir(parents=True)
    oof_entities = []
    fold_diagnostics = {}
    for fold in range(config.folds):
        train_seqs = tuple(seq for seq in sequences if fold_by_seq[seq] != fold)
        valid_seqs = tuple(seq for seq in sequences if fold_by_seq[seq] == fold)
        extractor = PairFeatureExtractor.fit_from_records(
            _record_factory(store, train_seqs), channels=store.channels,
            max_features=config.feature_max_features,
        )
        feature_names = extractor.feature_names
        train_x, train_y, train_groups = _matrix(
            store, train_seqs, extractor, config.max_train_pairs)
        valid_x, valid_y, valid_groups = _matrix(
            store, valid_seqs, extractor, config.max_train_pairs)
        result = train_pair_baseline(
            train_x, train_y, valid_x, valid_y, feature_names,
            train_groups, valid_groups, config.model,
        )
        fold_diagnostics[fold] = result.pair_diagnostics
        oof_entities.extend(_score_queries(
            store, valid_seqs, extractor, result.model, fold_by_seq))
        print(f"completed OOF fold {fold + 1}/{config.folds}", flush=True)
    grid = oof_threshold_grid(oof_entities, config.grid_points)
    search = search_thresholds(oof_entities, grid, "raw")
    sensitivity = threshold_sensitivity(
        oof_entities, search.frozen_config, config.sensitivity_delta)
    country_shift = leave_one_country_out(
        oof_entities, search.selected_policy, grid, "raw",
        config.country_min_entities)
    save_search_report(building / "threshold_search.json", search,
                       sensitivity, country_shift)
    save_frozen_config(building / "decision_config.json",
                       search.frozen_config)
    final_extractor = PairFeatureExtractor.fit_from_records(
        _record_factory(store, sequences), channels=store.channels,
        max_features=config.feature_max_features,
    )
    final_x, final_y, _ = _matrix(
        store, sequences, final_extractor, config.max_train_pairs)
    if len(np.unique(final_y)) != 2:
        raise ValueError("final training matrix needs both positive and negative pairs")
    dataset = lgb.Dataset(final_x, label=final_y,
                          feature_name=list(final_extractor.feature_names))
    model = lgb.train({
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": config.model.learning_rate,
        "num_leaves": config.model.num_leaves,
        "min_data_in_leaf": config.model.min_data_in_leaf,
        "num_threads": config.model.num_threads,
        "seed": config.model.seed, "deterministic": True,
        "force_col_wise": True, "verbosity": -1,
    }, dataset, num_boost_round=config.model.num_boost_round)
    model.save_model(str(building / "pair_model.txt"))
    final_extractor.name_tfidf.cache.clear()
    final_extractor.address_tfidf.cache.clear()
    joblib.dump(final_extractor, building / "feature_extractor.joblib")
    ordered = sorted(oof_entities, key=lambda item: item.source_id)
    ranking = evaluate_oof_ranking(
        [entity.source_id for entity in ordered for _ in entity.candidate_ids],
        [target for entity in ordered for target in entity.candidate_ids],
        [score for entity in ordered for score in entity.scores],
        {entity.source_id: entity.truth for entity in ordered},
    )
    report: dict[str, object] = {
        "scope": "fixed-seed sampled training OOF; not a full-training estimate",
        "training_source_population": store.source_count,
        "sampled_source_entities": len(sequences),
        "sampled_pair_rows": len(final_y),
        "sampled_retrieved_positives": int(final_y.sum()),
        "fold_pair_diagnostics": fold_diagnostics,
        "oof_retrieved_pair_ranking": ranking,
        "selected_policy": search.selected_policy,
        "selected_crossfit_metrics": asdict(
            search.policies[search.selected_policy].crossfit_metrics),
        "config": asdict(config),
        "retrieval_channels": store.channels,
        "retrieval_top_k": store.top_k,
        "retrieval_max_candidates": store.max_candidates,
    }
    (building / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(building, output_dir)
    return report
