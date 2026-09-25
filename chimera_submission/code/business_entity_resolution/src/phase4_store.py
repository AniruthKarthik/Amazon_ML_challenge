"""Disk-backed training-only input store and bounded lexical channels.

This store avoids retaining 12 million normalized records in Python memory.
It is a scratch artifact; source TSVs remain untouched.
"""

from __future__ import annotations

import csv
import os
import sqlite3
from array import array
from collections import Counter, defaultdict
from math import log
from pathlib import Path

import numpy as np

from .data_contract import DataContractError, TRUTH_COLUMNS, _read_tsv, iter_source
from .normalization import normalize_record
from .retrieval import RetrievalConfig


def open_store(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def build_store(train_dir: str | Path, path: str | Path) -> dict[str, int]:
    """Validate and normalize all training rows into an indexed SQLite store."""
    train_dir, path = Path(train_dir), Path(path)
    if path.exists():
        raise FileExistsError(path)
    building = path.with_name(path.name + ".building")
    if building.exists():
        raise FileExistsError(building)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = open_store(building)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE source1 (seq INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL, "
        "name_clean TEXT NOT NULL, name_core TEXT NOT NULL, address_alias TEXT NOT NULL, "
        "country TEXT NOT NULL, raw_name TEXT NOT NULL, raw_address TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE targets (seq INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL, "
        "source TEXT NOT NULL, name_clean TEXT NOT NULL, name_core TEXT NOT NULL, "
        "address_alias TEXT NOT NULL, country TEXT NOT NULL, raw_name TEXT NOT NULL, "
        "raw_address TEXT NOT NULL)"
    )
    counts = {"S1": 0, "S2": 0, "S3": 0, "ground_truth_links": 0}
    for source in (1, 2, 3):
        batch = []
        for record in iter_source(train_dir / f"train_source{source}.tsv", source):
            view = normalize_record(record)
            if source == 1:
                batch.append((counts["S1"], record.entity_id, view.business_name_clean,
                              view.business_name_core, view.business_address_alias,
                              record.country, record.business_name, record.business_address))
            else:
                batch.append((counts["S2"] + counts["S3"], record.entity_id,
                              f"S{source}", view.business_name_clean,
                              view.business_name_core, view.business_address_alias,
                              record.country, record.business_name, record.business_address))
            counts[f"S{source}"] += 1
            if len(batch) >= 10_000:
                try:
                    connection.executemany(
                        "INSERT INTO source1 VALUES (?,?,?,?,?,?,?,?)" if source == 1
                        else "INSERT INTO targets VALUES (?,?,?,?,?,?,?,?,?)", batch
                    )
                except sqlite3.IntegrityError as exc:
                    raise DataContractError(f"duplicate entity ID in source {source}") from exc
                connection.commit()
                batch.clear()
        if batch:
            try:
                connection.executemany(
                    "INSERT INTO source1 VALUES (?,?,?,?,?,?,?,?)" if source == 1
                    else "INSERT INTO targets VALUES (?,?,?,?,?,?,?,?,?)", batch
                )
            except sqlite3.IntegrityError as exc:
                raise DataContractError(f"duplicate entity ID in source {source}") from exc
            connection.commit()
    connection.execute("CREATE INDEX target_exact_name ON targets(name_clean, entity_id)")
    connection.execute("CREATE INDEX target_exact_core ON targets(name_core, entity_id)")
    connection.execute("CREATE TABLE truth_entities (entity_id TEXT PRIMARY KEY)")
    connection.execute("CREATE TABLE truth_links (source_id TEXT NOT NULL, target_id TEXT UNIQUE NOT NULL)")
    entity_batch, link_batch = [], []
    truth_path = train_dir / "train_ground_truth.tsv"
    for line, row in _read_tsv(truth_path, TRUTH_COLUMNS):
        source_id = row["source1_entity_id"]
        if not source_id.startswith("S1-"):
            raise DataContractError(f"{truth_path}:{line}: invalid S1 ID {source_id!r}")
        entity_batch.append((source_id,))
        if row["matched_entity_ids"]:
            for target_id in row["matched_entity_ids"].split(","):
                if not (target_id.startswith("S2-") or target_id.startswith("S3-")):
                    raise DataContractError(f"{truth_path}:{line}: invalid target {target_id!r}")
                link_batch.append((source_id, target_id))
                counts["ground_truth_links"] += 1
        if len(entity_batch) >= 10_000:
            try:
                connection.executemany("INSERT INTO truth_entities VALUES (?)", entity_batch)
                connection.executemany("INSERT INTO truth_links VALUES (?,?)", link_batch)
            except sqlite3.IntegrityError as exc:
                raise DataContractError("duplicate ground-truth entity or target") from exc
            connection.commit()
            entity_batch.clear()
            link_batch.clear()
    if entity_batch:
        try:
            connection.executemany("INSERT INTO truth_entities VALUES (?)", entity_batch)
            connection.executemany("INSERT INTO truth_links VALUES (?,?)", link_batch)
        except sqlite3.IntegrityError as exc:
            raise DataContractError("duplicate ground-truth entity or target") from exc
        connection.commit()
    if connection.execute("SELECT COUNT(*) FROM truth_entities").fetchone()[0] != counts["S1"]:
        raise DataContractError("ground truth does not cover every S1 entity exactly once")
    if connection.execute(
        "SELECT e.entity_id FROM truth_entities e LEFT JOIN source1 s "
        "ON s.entity_id=e.entity_id WHERE s.entity_id IS NULL LIMIT 1"
    ).fetchone():
        raise DataContractError("ground truth contains an unknown S1 entity")
    if connection.execute(
        "SELECT l.target_id FROM truth_links l LEFT JOIN targets t "
        "ON t.entity_id=l.target_id WHERE t.entity_id IS NULL LIMIT 1"
    ).fetchone():
        raise DataContractError("ground truth contains an unknown target")
    connection.execute(
        "CREATE TABLE truth_indexed AS SELECT s.seq AS source_seq, t.seq AS target_seq "
        "FROM truth_links l JOIN source1 s ON s.entity_id=l.source_id "
        "JOIN targets t ON t.entity_id=l.target_id"
    )
    connection.execute("CREATE INDEX truth_by_source ON truth_indexed(source_seq, target_seq)")
    connection.commit()
    connection.close()
    os.replace(building, path)
    return counts


def candidate_array(path: str | Path, rows: int, top_k: int, create: bool) -> np.memmap:
    mode = "w+" if create else "r"
    values = np.memmap(path, dtype=np.int32, mode=mode, shape=(rows, top_k))
    if create:
        values[:] = -1
        values.flush()
    return values


def exact_channel(connection: sqlite3.Connection, view: str, path: str | Path,
                  config: RetrievalConfig) -> None:
    """Write top-K exact target indices in lexicographic target-ID order."""
    if view not in ("name_clean", "name_core"):
        raise ValueError("invalid exact view")
    source_count = connection.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
    output = candidate_array(path, source_count, config.top_k, create=True)
    find = connection.cursor()
    for seq, value in connection.execute(f"SELECT seq, {view} FROM source1 ORDER BY seq"):
        if not value:
            continue
        hits = find.execute(
            f"SELECT seq FROM targets WHERE {view}=? ORDER BY entity_id LIMIT ?",
            (value, config.top_k),
        ).fetchall()
        for rank, (target_seq,) in enumerate(hits):
            output[seq, rank] = target_seq
    output.flush()


def rare_token_channel(connection: sqlite3.Connection, path: str | Path,
                       config: RetrievalConfig) -> None:
    """Build the Phase 3 rare-token score and persist bounded target indices."""
    source_count = connection.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
    target_count = connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    document_frequency = Counter(
        token for (name,) in connection.execute("SELECT name_core FROM targets ORDER BY seq")
        for token in set(name.split())
    )
    weights = {
        token: log((target_count + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
        if frequency <= config.max_token_df
    }
    del document_frequency
    postings: dict[str, array] = defaultdict(lambda: array("i"))
    for seq, name in connection.execute("SELECT seq, name_core FROM targets ORDER BY seq"):
        for token in set(name.split()):
            if token in weights:
                postings[token].append(seq)
    max_id_bytes = connection.execute(
        "SELECT MAX(LENGTH(CAST(entity_id AS BLOB))) FROM targets"
    ).fetchone()[0]
    target_ids = np.array(
        [identifier.encode("utf-8") for (identifier,) in connection.execute(
            "SELECT entity_id FROM targets ORDER BY seq")],
        dtype=f"S{max(max_id_bytes or 1, 1)}",
    )
    output = candidate_array(path, source_count, config.top_k, create=True)
    score_path = Path(path).with_suffix(".float32")
    if score_path.exists():
        raise FileExistsError(score_path)
    output_scores = np.memmap(score_path, dtype=np.float32, mode="w+",
                              shape=(source_count, config.top_k))
    output_scores[:] = -np.inf
    for seq, name in connection.execute("SELECT seq, name_core FROM source1 ORDER BY seq"):
        tokens = sorted(set(name.split()) & weights.keys())
        denominator = sum(weights[token] for token in tokens)
        if not denominator:
            continue
        scores: dict[int, float] = defaultdict(float)
        for token in tokens:
            for target_seq in postings[token]:
                scores[target_seq] += weights[token]
        ranked = sorted(scores, key=lambda target_seq: (
            -scores[target_seq] / denominator, target_ids[target_seq]
        ))[:config.top_k]
        output[seq, :len(ranked)] = ranked
        output_scores[seq, :len(ranked)] = [scores[target_seq] / denominator
                                           for target_seq in ranked]
    output.flush()
    output_scores.flush()
