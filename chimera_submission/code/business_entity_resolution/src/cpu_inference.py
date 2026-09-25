"""CPU-batched frozen inference and independent streaming TSV validation.

No training truth, test ground truth, or test labels are read here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from .phase4_benchmark import run_channel
from .phase4_store import open_store
from .pipeline_store import DiskCandidateStore, QueryCandidates, build_unlabeled_store
from .retrieval import CHANNELS, RetrievalConfig
from .threshold_policy import load_frozen_config


MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def _hash_inputs(test_dir: Path) -> dict[str, str]:
    result = {}
    for source in (1, 2, 3):
        name = f"test_source{source}.tsv"
        digest = hashlib.sha256()
        with (test_dir / name).open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        result[name] = digest.hexdigest()
    return result


def prepare_test_retrieval(
    test_dir: str | Path, work_dir: str | Path,
    top_k: int, max_candidates: int,
    shard_size: int = 100_000, threads: int = 4,
) -> dict[str, int]:
    """Validate unlabeled test TSVs and materialize the same five channels."""
    test_dir, work_dir = Path(test_dir), Path(work_dir)
    config = RetrievalConfig(top_k=top_k, max_candidates=max_candidates)
    if min(shard_size, threads) < 1:
        raise ValueError("shard size and thread count must be positive")
    work_dir.mkdir(parents=True, exist_ok=True)
    hashes = _hash_inputs(test_dir)
    manifest = work_dir / "test_inputs.json"
    store_path = work_dir / "store.sqlite"
    if store_path.exists():
        if not manifest.exists() or json.loads(manifest.read_text()) != hashes:
            raise ValueError("test TSVs changed or reusable store lacks input manifest")
    else:
        if manifest.exists():
            raise ValueError("test manifest exists without its validated store")
        build_unlabeled_store(test_dir, store_path)
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump(hashes, handle, indent=2, sort_keys=True)
            handle.write("\n")
    connection = open_store(store_path)
    try:
        counts = {
            "S1": connection.execute("SELECT COUNT(*) FROM source1").fetchone()[0],
            "S2": connection.execute(
                "SELECT COUNT(*) FROM targets WHERE source='S2'").fetchone()[0],
            "S3": connection.execute(
                "SELECT COUNT(*) FROM targets WHERE source='S3'").fetchone()[0],
        }
        if connection.execute(
            "SELECT name FROM sqlite_master WHERE name='truth_indexed'"
        ).fetchone():
            raise ValueError("test store must not contain ground truth")
        for channel in CHANNELS:
            run_channel(connection, channel, work_dir, config, shard_size, threads)
        # Verify completed arrays, including score channels, before accepting.
        DiskCandidateStore(connection, work_dir, top_k, max_candidates)
        return counts
    finally:
        connection.close()


def _write_batch(
    queries: list[QueryCandidates], extractor, model: lgb.Booster,
    decision, matching_writer, candidate_writer,
) -> None:
    rows = []
    lengths = []
    for query in queries:
        lengths.append(len(query.candidates))
        for candidate in query.candidates:
            rows.append(tuple(extractor.features(
                query.source, query.targets[candidate.candidate_entity_id], candidate,
            ).values()))
    probabilities = (
        np.asarray(model.predict(np.asarray(rows, dtype=np.float32)), dtype=np.float64)
        if rows else np.empty(0, dtype=np.float64)
    )
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("pair model produced invalid probabilities")
    offset = 0
    for query, length in zip(queries, lengths):
        source_id = query.source.raw.entity_id
        ids = [candidate.candidate_entity_id for candidate in query.candidates]
        scores = probabilities[offset:offset + length].tolist()
        offset += length
        matched = decision.predict(ids, scores)
        if len(matched) != len(set(matched)) or not set(matched) <= set(ids):
            raise AssertionError("frozen policy emitted a duplicate or unknown target")
        matching_writer.writerow((source_id, ",".join(matched)))
        candidate_writer.writerow((source_id, ",".join(ids)))


def predict_test(
    model_dir: str | Path, test_work_dir: str | Path,
    output_dir: str | Path, batch_entities: int = 128,
    allow_exploratory: bool = False,
    dense_test_work_dir: str | Path | None = None,
) -> dict[str, int]:
    """Apply the frozen raw-score model and policy; write complete TSVs atomically."""
    if batch_entities < 1:
        raise ValueError("batch_entities must be positive")
    model_dir, test_work_dir, output_dir = map(
        Path, (model_dir, test_work_dir, output_dir))
    report = json.loads((model_dir / "training_report.json").read_text())
    if report.get("scope", "").startswith("fixed-seed sampled") and not allow_exploratory:
        raise ValueError("sampled model is exploratory; pass allow_exploratory=True for a smoke run")
    decision = load_frozen_config(model_dir / "decision_config.json")
    if decision.score_path != "raw":
        raise ValueError("this inference runner only supports frozen raw-score policies")
    # joblib is pickle-based; only load a trusted, locally produced artifact.
    extractor = joblib.load(model_dir / "feature_extractor.joblib")
    model = lgb.Booster(model_file=str(model_dir / "pair_model.txt"))
    if tuple(model.feature_name()) != extractor.feature_names:
        raise ValueError("frozen model/feature schema mismatch")
    if not (test_work_dir / "store.sqlite").is_file():
        raise FileNotFoundError(test_work_dir / "store.sqlite")
    connection = open_store(test_work_dir / "store.sqlite")
    building = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists() or building.exists():
        connection.close()
        raise FileExistsError("inference output or partial output directory already exists")
    building.mkdir(parents=True)
    matching_tmp = building / "matching_results.tsv"
    candidate_tmp = building / "candidate_pairs.tsv"
    try:
        store = DiskCandidateStore(
            connection, test_work_dir,
            int(report["retrieval_top_k"]), int(report["retrieval_max_candidates"]),
            dense_work_dir=dense_test_work_dir,
            dense_top_k=int(report.get("retrieval_dense_top_k") or 20),
        )
        if tuple(report["retrieval_channels"]) != store.channels or \
           extractor.channels != store.channels or \
           report.get("retrieval_dense_top_k") != store.dense_top_k:
            raise ValueError("inference retrieval channels do not match training")
        with matching_tmp.open("x", encoding="utf-8", newline="") as match_handle, \
             candidate_tmp.open("x", encoding="utf-8", newline="") as candidate_handle:
            matching_writer = csv.writer(match_handle, delimiter="\t", lineterminator="\n")
            candidate_writer = csv.writer(candidate_handle, delimiter="\t", lineterminator="\n")
            matching_writer.writerow(MATCHING_HEADER)
            candidate_writer.writerow(CANDIDATE_HEADER)
            batch = []
            for query in store.iter_queries():
                batch.append(query)
                if len(batch) == batch_entities:
                    _write_batch(batch, extractor, model, decision,
                                 matching_writer, candidate_writer)
                    batch.clear()
            if batch:
                _write_batch(batch, extractor, model, decision,
                             matching_writer, candidate_writer)
        validate_outputs(connection, matching_tmp, candidate_tmp)
        os.replace(building, output_dir)
        return {"source1_rows": store.source_count,
                "target_rows": store.target_count}
    finally:
        connection.close()


def _parse_ids(raw: str) -> list[str]:
    if not raw:
        return []
    ids = raw.split(",")
    if any(not identifier or identifier.strip() != identifier for identifier in ids):
        raise ValueError("output contains malformed candidate ID list")
    if len(ids) != len(set(ids)):
        raise ValueError("output contains duplicate candidate IDs")
    return ids


def validate_outputs(connection, matching_path: str | Path,
                     candidate_path: str | Path) -> dict[str, int]:
    """Streaming independent S1 coverage, subset, format, and target-ID checks."""
    pending_ids: set[str] = set()

    def check_pending() -> None:
        ordered = sorted(pending_ids)
        for start in range(0, len(ordered), 900):
            chunk = tuple(ordered[start:start + 900])
            found = {row[0] for row in connection.execute(
                "SELECT entity_id FROM targets WHERE entity_id IN (" +
                ",".join("?" for _ in chunk) + ")", chunk,
            )}
            if found != set(chunk):
                raise ValueError("candidate ID does not exist in test targets")
        pending_ids.clear()

    with Path(matching_path).open(encoding="utf-8", newline="") as matching, \
         Path(candidate_path).open(encoding="utf-8", newline="") as candidate:
        match_reader = csv.reader(matching, delimiter="\t", strict=True)
        candidate_reader = csv.reader(candidate, delimiter="\t", strict=True)
        if tuple(next(match_reader, ())) != MATCHING_HEADER or \
           tuple(next(candidate_reader, ())) != CANDIDATE_HEADER:
            raise ValueError("submission header mismatch")
        expected = connection.execute("SELECT entity_id FROM source1 ORDER BY seq")
        rows = nonempty = 0
        for (source_id,) in expected:
            match_row = next(match_reader, None)
            candidate_row = next(candidate_reader, None)
            if match_row is None or candidate_row is None or \
               len(match_row) != 2 or len(candidate_row) != 2 or \
               match_row[0] != source_id or candidate_row[0] != source_id:
                raise ValueError("submission rows must cover S1 exactly in store order")
            matches = _parse_ids(match_row[1])
            candidates = _parse_ids(candidate_row[1])
            if not set(matches) <= set(candidates):
                raise ValueError("matching result is not a subset of candidates")
            for identifier in candidates:
                if not identifier.startswith(("S2-", "S3-")):
                    raise ValueError("candidate ID has invalid source prefix")
                pending_ids.add(identifier)
            if len(pending_ids) >= 20_000:
                check_pending()
            rows += 1
            nonempty += bool(matches)
        if pending_ids:
            check_pending()
        if next(match_reader, None) is not None or next(candidate_reader, None) is not None:
            raise ValueError("submission contains extra S1 rows")
        return {"source1_rows": rows, "nonempty_match_rows": nonempty}
