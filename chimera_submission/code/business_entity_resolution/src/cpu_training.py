"""Bounded CPU/GPU training on explicit S1 samples; no test inputs or labels.

The sample is an exploratory OOF experiment, not a full-training estimate.
Each S1 is a separate truth-graph component because training truth requires
each S2/S3 target to have at most one S1 owner.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import random
import sqlite3
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

from .entity_meta_model import (
    MetaModelConfig, compare_meta_to_phase10, generate_meta_oof_decisions,
    save_meta_artifact, save_meta_report,
)
from .pair_features import PairFeatureExtractor
from .pair_model import (
    BaselineConfig, lightgbm_parameters, resolve_training_device,
    train_pair_baseline,
)
from .oof_ranking import evaluate_oof_ranking
from .pipeline_store import DiskCandidateStore, QueryCandidates
from .pipeline_provenance import model_code_sha256
from .threshold_policy import save_frozen_config
from .threshold_search import (
    OOFEntity, ThresholdGrid, leave_one_country_out, save_search_report,
    search_thresholds, threshold_sensitivity,
)
from .workflow_progress import WorkflowProgress


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
    evaluate_meta: bool = False
    max_worst_fold_drop: float | None = None
    max_fold_std_increase: float | None = None
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
        if self.evaluate_meta:
            for value in (self.max_worst_fold_drop, self.max_fold_std_increase):
                if isinstance(value, bool) or not isinstance(value, Real) or \
                   not math.isfinite(value) or value < 0:
                    raise ValueError("meta robustness tolerances must be explicit and nonnegative")
        elif self.max_worst_fold_drop is not None or self.max_fold_std_increase is not None:
            raise ValueError("meta robustness tolerances require evaluate_meta")


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
    fold_by_seq: dict[int, int], num_threads: int,
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
            scores = tuple(float(value) for value in model.predict(
                matrix, num_threads=num_threads))
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


def _write_audit_artifacts(
    directory: Path, store: DiskCandidateStore,
    sequences: tuple[int, ...], fold_by_seq: dict[int, int],
    entities: list[OOFEntity],
) -> dict[str, object]:
    """Persist replayable sample membership and label-free raw OOF pair scores."""
    sample_path = directory / "sampled_sources.tsv"
    sample_ids = set()
    with sample_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("source_seq", "source1_entity_id", "fold_id"))
        for seq in sequences:
            source_id = store.connection.execute(
                "SELECT entity_id FROM source1 WHERE seq=?", (seq,)
            ).fetchone()[0]
            sample_ids.add(source_id)
            writer.writerow((seq, source_id, fold_by_seq[seq]))
    scores_path = directory / "oof_pairs.tsv.gz"
    ordered = sorted(entities, key=lambda entity: entity.source_id)
    if len(ordered) != len(sequences) or \
       {entity.source_id for entity in ordered} != sample_ids:
        raise ValueError("OOF entities do not cover sampled S1 exactly")
    pair_count = 0
    with scores_path.open("xb") as binary:
        with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("source1_entity_id", "candidate_entity_id",
                                 "candidate_rank", "fold_id", "raw_oof_score"))
                for entity in ordered:
                    if len(entity.candidate_ids) != len(entity.scores) or \
                       len(set(entity.candidate_ids)) != len(entity.candidate_ids):
                        raise ValueError("OOF candidate IDs/scores are invalid")
                    for rank, (target_id, score) in enumerate(
                        zip(entity.candidate_ids, entity.scores), 1
                    ):
                        if not math.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError("OOF pair score is invalid")
                        writer.writerow((entity.source_id, target_id, rank,
                                         entity.fold, format(score, ".17g")))
                        pair_count += 1

    def digest(path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                sha.update(block)
        return sha.hexdigest()

    return {
        "sampled_sources": sample_path.name,
        "sampled_sources_sha256": digest(sample_path),
        "oof_pairs": scores_path.name,
        "oof_pairs_sha256": digest(scores_path),
        "oof_pair_rows": pair_count,
    }


def train_cpu_baseline(
    store: DiskCandidateStore, output_dir: str | Path,
    config: CPUTrainingConfig,
    training_files_sha256: dict[str, str] | None = None,
    progress: WorkflowProgress | None = None,
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
    runtime_model, training_backend = resolve_training_device(config.model)
    backend_message = (
        "LightGBM training backend: "
        f"{training_backend['resolved_device']} "
        f"(requested {training_backend['requested_device']}, "
        f"max_bin={training_backend['max_bin']})"
    )
    if progress is None:
        print(backend_message, flush=True)
    else:
        progress.detail(backend_message)
    sequences = select_source_sequences(store.source_count, config.train_entities,
                                        config.seed)
    fold_by_seq = _folds_for_sequences(sequences, config.folds)
    building.mkdir(parents=True)
    oof_entities = []
    fold_diagnostics = {}
    for fold in range(config.folds):
        if progress is not None:
            progress.start(f"OOF LightGBM fold {fold + 1}/{config.folds}")
        train_seqs = tuple(seq for seq in sequences if fold_by_seq[seq] != fold)
        valid_seqs = tuple(seq for seq in sequences if fold_by_seq[seq] == fold)
        if progress is not None:
            progress.detail("Fitting fold-specific pair feature encoders")
        extractor = PairFeatureExtractor.fit_from_records(
            _record_factory(store, train_seqs), channels=store.channels,
            max_features=config.feature_max_features,
        )
        feature_names = extractor.feature_names
        if progress is not None:
            progress.detail("Building fold train/validation pair matrices")
        train_x, train_y, train_groups = _matrix(
            store, train_seqs, extractor, config.max_train_pairs)
        valid_x, valid_y, valid_groups = _matrix(
            store, valid_seqs, extractor, config.max_train_pairs)
        if progress is not None:
            progress.detail("Training LightGBM and scoring held-out entities")
        result = train_pair_baseline(
            train_x, train_y, valid_x, valid_y, feature_names,
            train_groups, valid_groups, runtime_model,
        )
        fold_diagnostics[fold] = result.pair_diagnostics
        oof_entities.extend(_score_queries(
            store, valid_seqs, extractor, result.model, fold_by_seq,
            runtime_model.num_threads))
        print(f"completed OOF fold {fold + 1}/{config.folds}", flush=True)
        if progress is not None:
            progress.finish()
    if progress is not None:
        progress.start("OOF audit and entity threshold search")
        progress.detail("Writing sampled S1 and raw OOF pair-score audits")
    audit = _write_audit_artifacts(
        building, store, sequences, fold_by_seq, oof_entities)
    if progress is not None:
        progress.detail("Searching A/B/C thresholds on OOF entity scores")
    grid = oof_threshold_grid(oof_entities, config.grid_points)
    search = search_thresholds(oof_entities, grid, "raw")
    if progress is not None:
        progress.detail("Evaluating threshold sensitivity and country holdouts")
    sensitivity = threshold_sensitivity(
        oof_entities, search.frozen_config, config.sensitivity_delta)
    country_shift = leave_one_country_out(
        oof_entities, search.selected_policy, grid, "raw",
        config.country_min_entities)
    save_search_report(building / "threshold_search.json", search,
                       sensitivity, country_shift)
    save_frozen_config(building / "decision_config.json",
                       search.frozen_config)
    if progress is not None:
        progress.finish()
    selected_entity_decision = "deterministic"
    if config.evaluate_meta:
        if progress is not None:
            progress.start("Optional ZERO/ONE/MANY meta-model comparison")
        meta_result = generate_meta_oof_decisions(
            oof_entities, grid.pair, MetaModelConfig(num_threads=runtime_model.num_threads))
        comparison = compare_meta_to_phase10(
            oof_entities, meta_result,
            search.policies[search.selected_policy].fold_configs,
            config.max_worst_fold_drop, config.max_fold_std_increase,
        )
        save_meta_report(building / "meta_comparison.json", meta_result, comparison)
        if comparison.keep_meta:
            save_meta_artifact(building / "meta_model.joblib", meta_result)
            selected_entity_decision = "meta"
        if progress is not None:
            progress.finish()
    if progress is not None:
        progress.start("Final sampled LightGBM model and training report")
        progress.detail("Fitting final pair feature encoders and matrix")
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
    if progress is not None:
        progress.detail("Training final LightGBM pair scorer")
    model = lgb.train(
        lightgbm_parameters(runtime_model), dataset,
        num_boost_round=runtime_model.num_boost_round,
    )
    model.save_model(str(building / "pair_model.txt"))
    final_extractor.name_tfidf.cache.clear()
    final_extractor.address_tfidf.cache.clear()
    joblib.dump(final_extractor, building / "feature_extractor.joblib")
    if progress is not None:
        progress.detail("Writing OOF ranking diagnostics and model report")
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
        "selected_entity_decision": selected_entity_decision,
        "selected_crossfit_metrics": asdict(
            search.policies[search.selected_policy].crossfit_metrics),
        "config": asdict(config),
        "training_backend": training_backend,
        "training_files_sha256": training_files_sha256,
        "model_code_sha256": model_code_sha256(),
        "audit_artifacts": audit,
        "retrieval_channels": store.channels,
        "retrieval_top_k": store.top_k,
        "retrieval_max_candidates": store.max_candidates,
        "retrieval_dense_top_k": store.dense_top_k,
    }
    (building / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(building, output_dir)
    if progress is not None:
        progress.finish()
    return report
