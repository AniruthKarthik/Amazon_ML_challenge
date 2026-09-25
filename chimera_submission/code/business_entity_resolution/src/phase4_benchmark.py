"""Reproducible, training-only Phase 4 retrieval benchmark."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from .phase4_char import char_channel
from .cpu_resources import available_cpu_count, parse_thread_count
from .phase4_metrics import CHANNEL_BITS, RetrievalBenchmark
from .phase4_store import (
    build_store, candidate_array, exact_channel, open_store, rare_token_channel,
)
from .phase4_taxonomy import classify_retrieval_miss
from .retrieval import CHANNELS, RetrievalConfig
from .workflow_progress import WorkflowProgress


class _TargetMetadata:
    def __init__(self, connection: sqlite3.Connection, source2_count: int):
        self.source2_count = source2_count
        self.count = connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
        max_length = connection.execute(
            "SELECT MAX(LENGTH(CAST(country AS BLOB))) FROM targets"
        ).fetchone()[0]
        self.country_bytes = np.fromiter(
            (country.encode("utf-8") for (country,) in connection.execute(
                "SELECT country FROM targets ORDER BY seq"
            )), dtype=f"S{max(max_length or 1, 1)}", count=self.count,
        )
        self.decoded = {value: value.decode("utf-8") for value in np.unique(self.country_bytes)}

    def __contains__(self, index: object) -> bool:
        return isinstance(index, (int, np.integer)) and 0 <= index < self.count

    def __getitem__(self, index: int) -> tuple[str, str]:
        if index not in self:
            raise KeyError(index)
        return ("S2" if index < self.source2_count else "S3",
                self.decoded[self.country_bytes[index]])


def _target_ids(connection: sqlite3.Connection) -> np.ndarray:
    count = connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    max_length = connection.execute(
        "SELECT MAX(LENGTH(CAST(entity_id AS BLOB))) FROM targets"
    ).fetchone()[0]
    return np.fromiter(
        (identifier.encode("utf-8") for (identifier,) in connection.execute(
            "SELECT entity_id FROM targets ORDER BY seq"
        )), dtype=f"S{max(max_length or 1, 1)}", count=count,
    )


def _pool_sizes(connection: sqlite3.Connection) -> dict[str, int]:
    pools = {"all": connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]}
    for source in ("S2", "S3"):
        pools[source] = connection.execute(
            "SELECT COUNT(*) FROM targets WHERE source=?", (source,)
        ).fetchone()[0]
    for country, count in connection.execute(
        "SELECT country, COUNT(*) FROM targets GROUP BY country"
    ):
        pools[f"country:{country}"] = count
    return pools


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_path(work_dir: Path, channel: str) -> Path:
    return work_dir / f"{channel}.int32"


def _stage_config(channel: str, config: RetrievalConfig, shard_size: int) -> dict:
    return json.loads(json.dumps({
        "channel": channel, "retrieval_config": config.__dict__,
        "shard_size": shard_size if channel.startswith("char_") else None,
    }))


def run_channel(connection: sqlite3.Connection, channel: str, work_dir: Path,
                config: RetrievalConfig, shard_size: int, threads: int) -> bool:
    """Build one channel, returning True when a completed artifact was reused."""
    path = _stage_path(work_dir, channel)
    marker = work_dir / f"{channel}.complete"
    stage_config = _stage_config(channel, config, shard_size)
    if marker.exists():
        if not path.exists():
            raise FileNotFoundError(path)
        if json.loads(marker.read_text(encoding="utf-8")) != stage_config:
            raise ValueError(f"cached {channel} channel uses a different configuration")
        print(f"reusing complete channel {channel}", flush=True)
        return True
    # Only incomplete scratch outputs are replaced on a retry.
    for incomplete in (path, work_dir / f"{channel}.float32"):
        if incomplete.exists():
            incomplete.unlink()
    print(f"building channel {channel}", flush=True)
    if channel == "exact_name":
        exact_channel(connection, "name_clean", path, config, threads=threads)
    elif channel == "exact_core":
        exact_channel(connection, "name_core", path, config, threads=threads)
    elif channel == "rare_token":
        rare_token_channel(connection, path, config)
    elif channel in ("char_name", "char_address"):
        view = "name_clean" if channel == "char_name" else "address_alias"
        char_channel(connection, view, path, work_dir / f"{channel}.float32",
                     config, shard_size=shard_size, threads=threads)
    else:
        raise ValueError(channel)
    marker.write_text(json.dumps(stage_config, sort_keys=True), encoding="utf-8")
    print(f"completed channel {channel}", flush=True)
    return False


def evaluate(connection: sqlite3.Connection, work_dir: Path, output_dir: Path,
             config: RetrievalConfig) -> dict[str, object]:
    """Measure every S1 query and write status for every ground-truth link."""
    if config.max_candidates < len(CHANNELS) * config.top_k:
        raise ValueError("benchmark requires cap >= sum of per-channel top-K limits")
    source_count = connection.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
    source2_count = connection.execute(
        "SELECT COUNT(*) FROM targets WHERE source='S2'"
    ).fetchone()[0]
    arrays = {
        channel: candidate_array(_stage_path(work_dir, channel),
                                 source_count, config.top_k, create=False)
        for channel in CHANNELS
    }
    target_ids = _target_ids(connection)
    metadata = _TargetMetadata(connection, source2_count)
    benchmark = RetrievalBenchmark(_pool_sizes(connection), sample_limit=1000,
                                   random_seed=42)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "ground_truth_link_status.tsv.gz"
    building_status = output_dir / "ground_truth_link_status.tsv.gz.building"
    truth_cursor = connection.execute(
        "SELECT source_seq, target_seq FROM truth_indexed ORDER BY source_seq, target_seq"
    )
    next_truth = truth_cursor.fetchone()
    with gzip.open(building_status, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", "target_entity_id", "retrieved", "channels"))
        for seq, source_id, source_country in connection.execute(
            "SELECT seq, entity_id, country FROM source1 ORDER BY seq"
        ):
            truth = {}
            while next_truth is not None and next_truth[0] == seq:
                target_seq = next_truth[1]
                truth[target_seq] = metadata[target_seq]
                next_truth = truth_cursor.fetchone()
            masks = {}
            for channel, hits in arrays.items():
                bit = CHANNEL_BITS[channel]
                for target_seq in hits[seq]:
                    if target_seq >= 0:
                        target_seq = int(target_seq)
                        masks[target_seq] = masks.get(target_seq, 0) | bit
            benchmark.add_query_masks(source_id, source_country, truth, masks, metadata)
            for target_seq in sorted(truth):
                mask = masks.get(target_seq, 0)
                writer.writerow((
                    source_id, target_ids[target_seq].decode("utf-8"), int(bool(mask)),
                    ",".join(channel for channel in CHANNELS if mask & CHANNEL_BITS[channel]),
                ))
            if (seq + 1) % 250_000 == 0 or seq + 1 == source_count:
                print(f"training benchmark: {seq + 1:,}/{source_count:,} "
                      "S1 entities evaluated", flush=True)
    if next_truth is not None:
        raise ValueError("truth contains a source sequence outside the S1 table")
    os.replace(building_status, status_path)
    result = benchmark.report()
    result["input_counts"] = {
        "S1": source_count, "S2": source2_count,
        "S3": len(target_ids) - source2_count,
    }
    _write_miss_sample(connection, benchmark.miss_sample, output_dir, result)
    return result


def _write_miss_sample(connection: sqlite3.Connection, sample: list[tuple[str, int]],
                       output_dir: Path, result: dict[str, object]) -> None:
    path = output_dir / "retrieval_miss_sample.tsv"
    counts = Counter()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", "target_entity_id", "source_country",
                         "target_country", "source_name", "target_name",
                         "source_address", "target_address", "primary_reason",
                         "indicators", "name_similarity", "address_similarity"))
        for source_id, target_seq in sorted(sample):
            source = connection.execute(
                "SELECT country, raw_name, raw_address FROM source1 WHERE entity_id=?",
                (source_id,),
            ).fetchone()
            target = connection.execute(
                "SELECT entity_id, country, raw_name, raw_address FROM targets WHERE seq=?",
                (target_seq,),
            ).fetchone()
            if source is None or target is None:
                raise ValueError("sampled miss no longer exists in the training store")
            taxonomy = classify_retrieval_miss(source[1], source[2], target[2], target[3])
            counts[taxonomy["primary"]] += 1
            writer.writerow((
                source_id, target[0], source[0], target[1], source[1], target[2],
                source[2], target[3], taxonomy["primary"],
                ",".join(taxonomy["indicators"]), taxonomy["name_similarity"],
                taxonomy["address_similarity"],
            ))
    result["miss_taxonomy"] = {
        "method": "deterministic rule-based primary categories with nonexclusive indicators; exploratory",
        "sample_size": len(sample),
        "primary_counts": dict(sorted(counts.items())),
    }


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def write_report(result: dict[str, object], output_dir: Path) -> None:
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                       ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Phase 4 training retrieval benchmark", "",
        "Exploratory full-training diagnostics. These are not cross-fitted model-selection scores.",
        "No ranking-failure rate is reported; OOF LightGBM ranking analysis belongs after Phases 7/8.",
        "Entity any-hit and complete recall use only entities with at least one ground-truth link in the group.",
        "Oracle F₀.₅ predicts exactly the retrieved true links and scores correctly empty singletons as 1.",
        "", "## Overall and group metrics", "",
        "| Group | GT links | Recall | Any-hit | Complete | Candidates/query | Reduction | Oracle F₀.₅ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, row in result["groups"].items():
        lines.append(
            f"| {group} | {row['ground_truth_links']} | {_format_rate(row['candidate_recall'])} "
            f"| {_format_rate(row['entity_any_hit_recall'])} | {_format_rate(row['entity_complete_recall'])} "
            f"| {_format_rate(row['candidates_per_query'])} | {_format_rate(row['reduction_ratio'])} "
            f"| {_format_rate(row['oracle_entity_macro_f05'])} |"
        )
    distribution = result["candidate_count_distribution"]
    lines.extend([
        "", "Candidate count/query distribution: "
        f"min {distribution['minimum']}, p50 {distribution['p50']}, "
        f"p90 {distribution['p90']}, p99 {distribution['p99']}, "
        f"max {distribution['maximum']}; zero-candidate queries "
        f"{distribution['zero_candidate_queries']}.",
    ])
    lines.extend(["", "## Channel diversity", "",
                  "Incremental union gain compares Exact, then Exact ∪ Char, then Exact ∪ Char ∪ Rare.",
                  "", "| Channel or union | Candidates | GT recovered | GT recall | Incremental union gain | Unique GT only this channel |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    unique = result["unique_ground_truth_recovered_only_by_channel"]
    incremental = result["incremental_union_recovered_links"]
    for name, row in result["channel_views"].items():
        lines.append(
            f"| {name} | {row['candidate_pairs']} | {row['recovered_links']} "
            f"| {_format_rate(row['candidate_recall'])} "
            f"| {incremental.get(name, '—')} | {unique.get(name, '—')} |"
        )
    lines.extend(["", "Pairwise channel overlap in the final capped candidate set:", "",
                  "| Channels | Candidate pairs | GT pairs |",
                  "| --- | ---: | ---: |"])
    for overlap in result["channel_overlap"]:
        lines.append(
            f"| {' ∩ '.join(overlap['channels'])} | {overlap['candidate_pairs']} "
            f"| {overlap['ground_truth_pairs']} |"
        )
    lines.extend(["", "## Genuine retrieval misses", "",
                  f"Misses: {result['groups']['overall']['retrieval_misses']}.",
                  f"Deterministic reservoir sample: {result['miss_taxonomy']['sample_size']} links.",
                  "Taxonomy categories are rule-based exploratory labels; see `retrieval_miss_sample.tsv` for raw pairs and indicators.",
                  "", "| Primary reason | Sample count |", "| --- | ---: |"])
    for reason, count in result["miss_taxonomy"]["primary_counts"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Decision", "",
                  result.get("decision_rationale") or "Pending measured-error review.", ""])
    (output_dir / "phase4_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(train_dir: Path, output_dir: Path, work_dir: Path,
        config: RetrievalConfig, stage: str, shard_size: int, threads: int,
        progress: WorkflowProgress | None = None) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    store_path = work_dir / "store.sqlite"
    manifest_path = work_dir / "training_inputs.json"
    if progress is not None:
        progress.start("Training store and input validation")
    reused_store = store_path.exists()
    if not store_path.exists():
        print("building validated training store", flush=True)
        print(build_store(train_dir, store_path), flush=True)
    input_hashes = {
        name: _sha256(train_dir / name) for name in (
            "train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
            "train_ground_truth.tsv",
        )
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != input_hashes:
            raise ValueError("training TSVs changed since the retrieval store was built")
    else:
        manifest_path.write_text(json.dumps(input_hashes, sort_keys=True), encoding="utf-8")
    if progress is not None:
        progress.finish(reused=reused_store)
    if stage == "store":
        return
    connection = open_store(store_path)
    channels = CHANNELS if stage in ("all", "metrics") else (stage,)
    if stage != "metrics":
        for channel in channels:
            if progress is not None:
                progress.start(f"Training retrieval: {channel}")
            reused = run_channel(connection, channel, work_dir, config,
                                 shard_size, threads)
            if progress is not None:
                progress.finish(reused=reused)
    if stage in ("all", "metrics"):
        if not all((work_dir / f"{channel}.complete").exists() for channel in CHANNELS):
            raise RuntimeError("all five channels must be complete before metrics")
        source_count = connection.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
        for channel in CHANNELS:
            marker = work_dir / f"{channel}.complete"
            if json.loads(marker.read_text(encoding="utf-8")) != _stage_config(
                channel, config, shard_size
            ):
                raise ValueError(f"cached {channel} channel uses a different configuration")
            if _stage_path(work_dir, channel).stat().st_size != source_count * config.top_k * 4:
                raise ValueError(f"cached {channel} channel has an unexpected size")
        print("evaluating all training S1 entities", flush=True)
        if progress is not None:
            progress.start("Training retrieval benchmark and report")
        result = evaluate(connection, work_dir, output_dir, config)
        result["manifest"] = {
            "training_files_sha256": input_hashes,
            "retrieval_config": config.__dict__,
            "shard_size": shard_size,
            "threads": threads,
            "git_head_before_phase4_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "evaluation_scope": "exploratory full training; no test labels or test ground truth",
        }
        write_report(result, output_dir)
        print("Phase 4 metrics and artifacts written", flush=True)
        if progress is not None:
            progress.finish()
    connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "store", "metrics", *CHANNELS),
                        default="all")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=250)
    parser.add_argument("--shard-size", type=int, default=250_000)
    parser.add_argument("--threads", type=parse_thread_count,
                        default=available_cpu_count(),
                        help="parallel exact/character retrieval workers; auto uses all available CPU cores (default)")
    args = parser.parse_args()
    config = RetrievalConfig(top_k=args.top_k, max_candidates=args.max_candidates)
    total = 7 if args.stage == "all" else 1 if args.stage == "store" else 2
    progress = WorkflowProgress(total)
    run(args.train_dir, args.output_dir, args.work_dir, config,
        args.stage, args.shard_size, args.threads, progress=progress)
    progress.check_complete()


if __name__ == "__main__":
    main()
