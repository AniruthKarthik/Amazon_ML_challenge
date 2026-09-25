"""Phase D — database-free file store (stdlib only).

Replaces every SQLite usage with lightweight file-based structures:
CSVs/TSVs + JSON indexes + optional joblib shards. No database dependency.

Migration map (SQLite -> here):
- phase4_store.build_store source1/targets tables
    -> build_file_store() writes source1.tsv / targets.tsv (normalized views).
- truth_entities / truth_links / truth_indexed
    -> truth.json (S1 -> sorted target list) + in-memory dict lookups.
- target_exact_name / target_exact_core indexes
    -> build_exact_index() dict[normalized -> sorted ID list], optionally
       sharded by hash prefix via shard_exact_index().
- rare-token temp postings.sqlite
    -> build_token_postings() in-memory dict[token -> sorted seq list]
       with DF pruning; persisted as JSON shards when needed.
- per-query SELECT ... WHERE seq=? / WHERE seq IN (...)
    -> array/dict indexing via load_records() (seq -> record dict).
- SELECT COUNT(*) / ORDER BY seq scans
    -> len(records) / sorted seq iteration over TSV order.

Parquet is NOT required: TSV + JSON cover every access pattern with lower
RAM and no new dependency. If pyarrow/polars become available, the same
record dicts can be written as Parquet without changing callers.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .data_contract import TRUTH_COLUMNS, _read_tsv, iter_source
from .normalization import normalize_record

SOURCE1_FILENAME = "source1.tsv"
TARGETS_FILENAME = "targets.tsv"
TRUTH_FILENAME = "truth.json"
MANIFEST_FILENAME = "file_store_manifest.json"

STORE_HEADER = (
    "seq", "entity_id", "source", "name_clean", "name_core",
    "address_alias", "country", "raw_name", "raw_address",
)


def _manifest_hashes(train_dir: Path, files: tuple[str, ...]) -> dict[str, str]:
    hashes = {}
    for name in files:
        digest = hashlib.sha256()
        with (train_dir / name).open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    return hashes


def build_file_store(
    train_dir: str | Path,
    out_dir: str | Path,
    train_files: tuple[str, ...] = (
        "train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
        "train_ground_truth.tsv",
    ),
) -> dict[str, int]:
    """Stream TSVs -> normalized TSVs + truth.json. No SQLite involved."""
    train_dir, out_dir = Path(train_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    counts = {"S1": 0, "S2": 0, "S3": 0, "ground_truth_links": 0}
    s1_path = out_dir / SOURCE1_FILENAME
    tgt_path = out_dir / TARGETS_FILENAME
    seen: set[str] = set()
    with s1_path.open("x", encoding="utf-8", newline="") as s1_handle, \
         tgt_path.open("x", encoding="utf-8", newline="") as tgt_handle:
        s1_writer = csv.writer(s1_handle, delimiter="\t", lineterminator="\n")
        tgt_writer = csv.writer(tgt_handle, delimiter="\t", lineterminator="\n")
        s1_writer.writerow(STORE_HEADER)
        tgt_writer.writerow(STORE_HEADER)
        for source in (1, 2, 3):
            for record in iter_source(train_dir / train_files[source - 1], source):
                if record.entity_id in seen:
                    raise ValueError(f"duplicate entity ID {record.entity_id!r}")
                seen.add(record.entity_id)
                view = normalize_record(record)
                if source == 1:
                    seq = counts["S1"]
                    s1_writer.writerow((
                        seq, record.entity_id, "S1",
                        view.business_name_clean, view.business_name_core,
                        view.business_address_alias, record.country,
                        record.business_name, record.business_address,
                    ))
                else:
                    seq = counts["S2"] + counts["S3"]
                    tgt_writer.writerow((
                        seq, record.entity_id, f"S{source}",
                        view.business_name_clean, view.business_name_core,
                        view.business_address_alias, record.country,
                        record.business_name, record.business_address,
                    ))
                counts[f"S{source}"] += 1
    # Ground truth -> truth.json (validates coverage + target ownership).
    truth: dict[str, list[str]] = {}
    target_owner: dict[str, str] = {}
    valid_targets = set()
    with tgt_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            valid_targets.add(row["entity_id"])
    s1_ids = set()
    with s1_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            s1_ids.add(row["entity_id"])
    for _, row in _read_tsv(train_dir / train_files[3], TRUTH_COLUMNS):
        source_id = row["source1_entity_id"]
        if source_id not in s1_ids:
            raise ValueError(f"unknown S1 entity {source_id!r}")
        if source_id in truth:
            raise ValueError(f"duplicate S1 row {source_id!r}")
        raw = row["matched_entity_ids"]
        target_ids = [] if raw == "" else raw.split(",")
        for tid in target_ids:
            if tid not in valid_targets:
                raise ValueError(f"unknown target {tid!r}")
            if tid in target_owner:
                raise ValueError(f"target {tid!r} already linked")
            target_owner[tid] = source_id
        truth[source_id] = sorted(target_ids)
        counts["ground_truth_links"] += len(target_ids)
    missing = s1_ids - set(truth)
    if missing:
        raise ValueError(f"missing ground truth for {len(missing)} S1 entities")
    with (out_dir / TRUTH_FILENAME).open("x", encoding="utf-8") as handle:
        json.dump(truth, handle, sort_keys=True)
        handle.write("\n")
    with (out_dir / MANIFEST_FILENAME).open("x", encoding="utf-8") as handle:
        json.dump(_manifest_hashes(train_dir, train_files), handle,
                  indent=2, sort_keys=True)
        handle.write("\n")
    return counts


def load_records(path: str | Path) -> dict[int, dict]:
    """Load a normalized TSV into {seq: record dict} (seq-ordered)."""
    records: dict[int, dict] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != STORE_HEADER:
            raise ValueError(f"unexpected store header in {path}")
        for row in reader:
            seq = int(row["seq"])
            if seq in records:
                raise ValueError(f"duplicate seq {seq}")
            records[seq] = row
    return records


def load_truth(path: str | Path) -> dict[str, frozenset[str]]:
    """Load truth.json as {S1: frozenset(targets)}."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: frozenset(v) for k, v in payload.items()}


def build_exact_index(
    records: dict[int, dict], view: str,
) -> dict[str, list[str]]:
    """In-memory exact index: normalized value -> sorted entity IDs."""
    if view not in ("name_clean", "name_core"):
        raise ValueError("invalid exact view")
    index: dict[str, list[str]] = defaultdict(list)
    for seq in sorted(records):
        value = records[seq][view]
        if value:
            index[value].append(records[seq]["entity_id"])
    for ids in index.values():
        ids.sort()
    return dict(index)


def shard_exact_index(
    index: dict[str, list[str]], num_shards: int, out_dir: str | Path,
) -> list[Path]:
    """Persist exact index as hash-prefix JSON shards (bounded RAM per worker)."""
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: dict[int, dict] = {i: {} for i in range(num_shards)}
    for key, ids in index.items():
        slot = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % num_shards
        shards[slot][key] = ids
    paths = []
    for slot, payload in shards.items():
        path = out_dir / f"exact_index_shard_{slot:03d}.json"
        if path.exists():
            raise FileExistsError(path)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        paths.append(path)
    return paths


def build_token_postings(
    records: dict[int, dict], max_token_df: int,
) -> tuple[dict[str, list[int]], dict[str, float]]:
    """In-memory rare-token postings over name_core with DF pruning.

    Returns (postings, idf_weights). Deterministic: postings sorted by seq.
    Replaces the temp postings.sqlite without any database.
    """
    import math

    df: dict[str, int] = defaultdict(int)
    for seq in sorted(records):
        for token in set(records[seq]["name_core"].split()):
            df[token] += 1
    allowed = {t for t, f in df.items() if f <= max_token_df}
    n = len(records)
    weights = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in allowed}
    postings: dict[str, list[int]] = {t: [] for t in allowed}
    for seq in sorted(records):
        for token in set(records[seq]["name_core"].split()):
            if token in postings:
                postings[token].append(seq)
    return postings, weights
