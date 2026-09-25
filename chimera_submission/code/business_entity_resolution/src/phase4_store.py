"""Disk-backed training-only input store and bounded lexical channels.

This store avoids retaining 12 million normalized records in Python memory.
It is a scratch artifact; source TSVs remain untouched.
"""

from __future__ import annotations

import csv
import os
import sqlite3
import tempfile
import time
from collections import defaultdict
from functools import lru_cache
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
    started = time.monotonic()
    for seq, value in connection.execute(f"SELECT seq, {view} FROM source1 ORDER BY seq"):
        if (seq + 1) % 100_000 == 0:
            elapsed = max(time.monotonic() - started, 0.001)
            print(f"{view}: {seq + 1:,}/{source_count:,} S1 queries "
                  f"({(seq + 1) / elapsed:,.0f}/s)", flush=True)
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
    """Build identical rare-token rankings with disk-backed target postings.

    The SQLite scratch index is removed after completion; only bounded query
    accumulators and the fixed-width target-ID array remain in process memory.
    """
    path = Path(path)
    score_path = path.with_suffix(".float32")
    if path.exists() or score_path.exists():
        raise FileExistsError("rare-token channel output already exists")
    source_count = connection.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
    target_count = connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    max_id_bytes = connection.execute(
        "SELECT MAX(LENGTH(CAST(entity_id AS BLOB))) FROM targets"
    ).fetchone()[0]
    target_ids = np.fromiter(
        (identifier.encode("utf-8") for (identifier,) in connection.execute(
            "SELECT entity_id FROM targets ORDER BY seq")),
        dtype=f"S{max(max_id_bytes or 1, 1)}", count=target_count,
    )
    with tempfile.TemporaryDirectory(prefix="rare_postings_", dir=path.parent) as scratch:
        postings_db = sqlite3.connect(Path(scratch) / "postings.sqlite")
        try:
            postings_db.execute("PRAGMA journal_mode=OFF")
            postings_db.execute("PRAGMA synchronous=OFF")
            postings_db.execute("PRAGMA temp_store=FILE")
            postings_db.execute("PRAGMA cache_size=-65536")
            postings_db.execute(
                "CREATE TABLE postings (token TEXT NOT NULL, target_seq INTEGER NOT NULL)")
            batch: list[tuple[str, int]] = []
            for target_seq, name in connection.execute(
                "SELECT seq, name_core FROM targets ORDER BY seq"):
                batch.extend((token, target_seq) for token in set(name.split()))
                if len(batch) >= 50_000:
                    postings_db.executemany("INSERT INTO postings VALUES (?,?)", batch)
                    postings_db.commit()
                    batch.clear()
                if (target_seq + 1) % 1_000_000 == 0:
                    print(f"rare_token: indexed {target_seq + 1:,}/{target_count:,} targets",
                          flush=True)
            if batch:
                postings_db.executemany("INSERT INTO postings VALUES (?,?)", batch)
                postings_db.commit()
            postings_db.execute("CREATE INDEX posting_token_seq ON postings(token,target_seq)")
            print("rare_token: built disk posting index", flush=True)
            postings_db.execute(
                "CREATE TABLE allowed AS SELECT token, COUNT(*) AS frequency "
                "FROM postings GROUP BY token HAVING COUNT(*)<=?",
                (config.max_token_df,),
            )
            postings_db.execute("CREATE UNIQUE INDEX allowed_token ON allowed(token)")
            postings_db.commit()
            print("rare_token: applied document-frequency cutoff", flush=True)

            @lru_cache(maxsize=100_000)
            def token_weight(token: str) -> float | None:
                row = postings_db.execute(
                    "SELECT frequency FROM allowed WHERE token=?", (token,)
                ).fetchone()
                return log((target_count + 1) / (row[0] + 1)) + 1 if row else None

            @lru_cache(maxsize=2_048)
            def token_postings(token: str) -> tuple[int, ...]:
                return tuple(row[0] for row in postings_db.execute(
                    "SELECT target_seq FROM postings WHERE token=? ORDER BY target_seq",
                    (token,),
                ))

            output = candidate_array(path, source_count, config.top_k, create=True)
            output_scores = np.memmap(score_path, dtype=np.float32, mode="w+",
                                      shape=(source_count, config.top_k))
            output_scores[:] = -np.inf
            for seq, name in connection.execute(
                "SELECT seq, name_core FROM source1 ORDER BY seq"):
                if (seq + 1) % 100_000 == 0:
                    print(f"rare_token: processed {seq + 1:,}/{source_count:,} S1 queries",
                          flush=True)
                weighted_tokens = [(token, token_weight(token))
                                   for token in sorted(set(name.split()))]
                weighted_tokens = [(token, weight) for token, weight in weighted_tokens
                                   if weight is not None]
                denominator = sum(weight for _, weight in weighted_tokens)
                if not denominator:
                    continue
                scores: dict[int, float] = defaultdict(float)
                for token, weight in weighted_tokens:
                    for target_seq in token_postings(token):
                        scores[target_seq] += weight
                ranked = sorted(scores, key=lambda target_seq: (
                    -scores[target_seq] / denominator, target_ids[target_seq]
                ))[:config.top_k]
                output[seq, :len(ranked)] = ranked
                output_scores[seq, :len(ranked)] = [scores[target_seq] / denominator
                                                   for target_seq in ranked]
            output.flush()
            output_scores.flush()
        finally:
            postings_db.close()
