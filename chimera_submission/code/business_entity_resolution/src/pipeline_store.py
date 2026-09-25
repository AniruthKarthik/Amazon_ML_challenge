"""Disk-backed bridge from retrieval arrays to pair features and test inputs."""

from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from .data_contract import BusinessRecord, DataContractError, iter_source
from .dense_retrieval import DENSE_CHANNEL
from .normalization import NormalizedRecord, clean_text, fold_accents, normalize_record
from .phase4_store import open_store
from .retrieval import CHANNELS, Candidate, ChannelHit


def build_unlabeled_store(test_dir: str | Path, path: str | Path) -> dict[str, int]:
    """Validate and normalize test source files; never look for test labels."""
    test_dir, path = Path(test_dir), Path(path)
    if path.exists() or path.with_name(path.name + ".building").exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    building = path.with_name(path.name + ".building")
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
    counts = {"S1": 0, "S2": 0, "S3": 0}
    try:
        for source in (1, 2, 3):
            batch = []
            for record in iter_source(test_dir / f"test_source{source}.tsv", source):
                view = normalize_record(record)
                if source == 1:
                    batch.append((counts["S1"], record.entity_id,
                                  view.business_name_clean, view.business_name_core,
                                  view.business_address_alias, record.country,
                                  record.business_name, record.business_address))
                else:
                    batch.append((counts["S2"] + counts["S3"], record.entity_id,
                                  f"S{source}", view.business_name_clean,
                                  view.business_name_core, view.business_address_alias,
                                  record.country, record.business_name,
                                  record.business_address))
                counts[f"S{source}"] += 1
                if len(batch) >= 10_000:
                    connection.executemany(
                        "INSERT INTO source1 VALUES (?,?,?,?,?,?,?,?)" if source == 1
                        else "INSERT INTO targets VALUES (?,?,?,?,?,?,?,?,?)", batch,
                    )
                    connection.commit()
                    batch.clear()
            if batch:
                connection.executemany(
                    "INSERT INTO source1 VALUES (?,?,?,?,?,?,?,?)" if source == 1
                    else "INSERT INTO targets VALUES (?,?,?,?,?,?,?,?,?)", batch,
                )
                connection.commit()
        connection.execute("CREATE INDEX target_exact_name ON targets(name_clean, entity_id)")
        connection.execute("CREATE INDEX target_exact_core ON targets(name_core, entity_id)")
        connection.commit()
    except sqlite3.IntegrityError as exc:
        connection.close()
        building.unlink(missing_ok=True)
        raise DataContractError("duplicate entity ID in test source") from exc
    except BaseException:
        connection.close()
        building.unlink(missing_ok=True)
        raise
    connection.close()
    os.replace(building, path)
    return counts


def _normalized_from_store(row: tuple) -> NormalizedRecord:
    """Recover Phase 2 views from the validated store without losing raw text."""
    identifier, name_clean, name_core, address_alias, country, raw_name, raw_address = row
    return NormalizedRecord(
        BusinessRecord(identifier, raw_name, raw_address, country),
        name_clean, fold_accents(name_clean), name_core,
        clean_text(raw_address), address_alias,
    )


def _mapped_array(path: Path, dtype, rows: int, columns: int) -> np.memmap:
    expected = rows * columns * np.dtype(dtype).itemsize
    if not path.exists() or path.stat().st_size != expected:
        raise ValueError(f"retrieval array has missing or incorrect size: {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=(rows, columns))


@dataclass(frozen=True)
class QueryCandidates:
    source_seq: int
    source: NormalizedRecord
    candidates: tuple[Candidate, ...]
    targets: dict[str, NormalizedRecord]


class DiskCandidateStore:
    """Bounded random/sequential S1 access to immutable retrieval channel arrays."""

    def __init__(
        self, connection: sqlite3.Connection, work_dir: str | Path,
        top_k: int, max_candidates: int,
        dense_work_dir: str | Path | None = None, dense_top_k: int = 20,
    ):
        if min(top_k, max_candidates, dense_top_k) < 1:
            raise ValueError("candidate limits must be positive")
        self.connection = connection
        self.source_count = connection.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
        self.target_count = connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
        self.top_k = top_k
        self.max_candidates = max_candidates
        self.dense_top_k = dense_top_k if dense_work_dir is not None else None
        self.channels = CHANNELS + ((DENSE_CHANNEL,) if dense_work_dir else ())
        work_dir = Path(work_dir)
        self.indices = {
            channel: _mapped_array(work_dir / f"{channel}.int32", np.int32,
                                   self.source_count, top_k)
            for channel in CHANNELS
        }
        self.scores: dict[str, np.memmap] = {}
        for channel in ("char_name", "char_address"):
            path = work_dir / f"{channel}.float32"
            self.scores[channel] = _mapped_array(path, np.float32,
                                                 self.source_count, top_k)
        rare_scores = work_dir / "rare_token.float32"
        if rare_scores.exists():
            self.scores["rare_token"] = _mapped_array(
                rare_scores, np.float32, self.source_count, top_k)
        if dense_work_dir is not None:
            dense_dir = Path(dense_work_dir)
            self.indices[DENSE_CHANNEL] = _mapped_array(
                dense_dir / "dense.int32", np.int32,
                self.source_count, dense_top_k)
            self.scores[DENSE_CHANNEL] = _mapped_array(
                dense_dir / "dense.float32", np.float32,
                self.source_count, dense_top_k)

    def _hits_by_target(self, source_seq: int) -> dict[int, list[ChannelHit]]:
        hits_by_target: dict[int, list[ChannelHit]] = defaultdict(list)
        for channel in self.channels:
            for position, target_seq in enumerate(self.indices[channel][source_seq]):
                if target_seq < 0:
                    continue
                target_seq = int(target_seq)
                if target_seq >= self.target_count:
                    raise ValueError("retrieval array references unknown target sequence")
                rank = position + 1
                score = (float(self.scores[channel][source_seq, position])
                         if channel in self.scores else
                         1.0 if channel.startswith("exact_") else 1.0 / rank)
                if not np.isfinite(score):
                    raise ValueError("retrieval array has invalid score")
                if any(hit.channel == channel for hit in hits_by_target[target_seq]):
                    raise ValueError("retrieval channel contains duplicate target")
                hits_by_target[target_seq].append(ChannelHit(channel, rank, score))
        return hits_by_target

    def _target_rows(self, sequences: Sequence[int]) -> dict[int, NormalizedRecord]:
        targets = {}
        for start in range(0, len(sequences), 900):
            chunk = sequences[start:start + 900]
            placeholders = ",".join("?" for _ in chunk)
            for row in self.connection.execute(
                "SELECT seq,entity_id,name_clean,name_core,address_alias,country,"
                f"raw_name,raw_address FROM targets WHERE seq IN ({placeholders})",
                chunk,
            ):
                targets[row[0]] = _normalized_from_store(row[1:])
        if len(targets) != len(sequences):
            raise ValueError("retrieval array references missing target row")
        return targets

    def get_query(self, source_seq: int) -> QueryCandidates:
        if not 0 <= source_seq < self.source_count:
            raise IndexError("S1 source sequence out of range")
        row = self.connection.execute(
            "SELECT entity_id,name_clean,name_core,address_alias,country,"
            "raw_name,raw_address FROM source1 WHERE seq=?", (source_seq,),
        ).fetchone()
        if row is None:
            raise ValueError("S1 source sequence missing from store")
        source = _normalized_from_store(row)
        hits_by_target = self._hits_by_target(source_seq)
        if not hits_by_target:
            return QueryCandidates(source_seq, source, (), {})
        target_rows = self._target_rows(sorted(hits_by_target))
        ranked = sorted(hits_by_target.items(), key=lambda item: (
            -len(item[1]), -max(hit.score for hit in item[1]),
            min(hit.rank for hit in item[1]),
            target_rows[item[0]].raw.entity_id,
        ))[:self.max_candidates]
        candidates = tuple(
            Candidate(source.raw.entity_id,
                      target_rows[target_seq].raw.entity_id, rank,
                      max(hit.score for hit in hits), tuple(hits))
            for rank, (target_seq, hits) in enumerate(ranked, 1)
        )
        selected_targets = {
            target_rows[target_seq].raw.entity_id: target_rows[target_seq]
            for target_seq, _ in ranked
        }
        return QueryCandidates(source_seq, source, candidates, selected_targets)

    def iter_queries(self, start: int = 0, stop: int | None = None) -> Iterator[QueryCandidates]:
        stop = self.source_count if stop is None else stop
        if not 0 <= start <= stop <= self.source_count:
            raise ValueError("invalid S1 sequence range")
        for source_seq in range(start, stop):
            yield self.get_query(source_seq)
