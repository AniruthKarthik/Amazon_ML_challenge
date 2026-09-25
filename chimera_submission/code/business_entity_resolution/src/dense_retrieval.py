"""Optional Phase 13 disk-backed, CPU-only dense retrieval.

Dense is never enabled by default. Build uses a float16 disk cache and a
compressed Faiss IVF-PQ index; query output is a bounded disk-backed array.
All rows retain stable source/target sequence IDs from the SQLite store.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from contextlib import closing
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np

from .retrieval import Candidate, ChannelHit
from .entity_decision import entity_f05


MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_CARD = "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DENSE_CHANNEL = "dense_name_address"


class Encoder(Protocol):
    dimension: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Lazy optional dependency; a pinned model revision is required."""

    def __init__(self, revision: str, batch_size: int = 32, threads: int = 2):
        if not revision or not revision.strip() or min(batch_size, threads) < 1:
            raise ValueError("dense model revision, batch size and threads are required")
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "dense branch requires optional sentence-transformers and torch"
            ) from exc
        torch.set_num_threads(threads)
        self.model = SentenceTransformer(MODEL_ID, revision=revision, device="cpu")
        self.revision = revision
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = self.model.encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class DenseConfig:
    top_k: int = 20
    batch_size: int = 256
    nlist: int = 1024
    pq_subquantizers: int = 48
    nprobe: int = 16
    train_samples: int = 100_000
    threads: int = 4

    def __post_init__(self) -> None:
        values = (self.top_k, self.batch_size, self.nlist,
                  self.pq_subquantizers, self.nprobe,
                  self.train_samples, self.threads)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in values):
            raise ValueError("dense resource settings must be positive integers")
        if self.nprobe > self.nlist:
            raise ValueError("nprobe cannot exceed nlist")


def dense_text(name: str, address: str) -> str:
    """Keep Unicode and raw wording; do not hard-code country labels."""
    return f"{name.strip()} [SEP] {address.strip()}" if address.strip() else name.strip()


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("dense branch requires optional faiss-cpu") from exc
    return faiss


def _checkpoint(path: Path, data: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".building")
    temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _normalize_batch(values: np.ndarray, expected_rows: int,
                     dimension: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (expected_rows, dimension) or not np.isfinite(matrix).all():
        raise ValueError("encoder produced an invalid embedding batch")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, np.maximum(norms, 1e-12), out=matrix)
    return matrix


def encode_target_cache(
    rows: Iterable[tuple[int, str, str]], count: int, encoder: Encoder,
    cache_path: str | Path, checkpoint_path: str | Path,
    config: DenseConfig, model_revision: str, input_identity: str,
) -> None:
    """Encode ordered target rows to float16 disk, resuming at batch boundaries.

    `rows` must be replayable from sequence zero on each run. The checkpoint
    commits only after the memmap flush, so an interrupted batch is reencoded.
    """
    if count < 1 or encoder.dimension < 1 or not model_revision or not input_identity:
        raise ValueError("dense cache requires count, dimension, revision and input identity")
    cache_path, checkpoint_path = Path(cache_path), Path(checkpoint_path)
    identity = {"count": count, "dimension": encoder.dimension,
                "model_id": MODEL_ID, "revision": model_revision,
                "batch_size": config.batch_size, "text_format_version": 1,
                "input_identity": input_identity}
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if {key: state.get(key) for key in identity} != identity:
            raise ValueError("dense embedding checkpoint configuration mismatch")
        if not cache_path.exists() or cache_path.stat().st_size != count * encoder.dimension * 2:
            raise ValueError("dense embedding cache is missing or has wrong size")
        completed = state["completed"]
        if not isinstance(completed, int) or not 0 <= completed <= count:
            raise ValueError("dense embedding checkpoint position is invalid")
        matrix = np.memmap(cache_path, dtype=np.float16, mode="r+",
                           shape=(count, encoder.dimension))
    else:
        if cache_path.exists():
            raise FileExistsError("dense embedding cache exists without checkpoint")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.memmap(cache_path, dtype=np.float16, mode="w+",
                           shape=(count, encoder.dimension))
        completed = 0
        _checkpoint(checkpoint_path, dict(identity, completed=0))
    batch: list[str] = []
    expected = 0
    for seq, name, address in rows:
        if seq != expected:
            raise ValueError("target rows must have contiguous sequence IDs")
        expected += 1
        if seq < completed:
            continue
        batch.append(dense_text(name, address))
        if len(batch) == config.batch_size:
            vectors = _normalize_batch(encoder.encode(batch), len(batch), encoder.dimension)
            matrix[seq + 1 - len(batch):seq + 1] = vectors.astype(np.float16)
            matrix.flush()
            _checkpoint(checkpoint_path, dict(identity, completed=seq + 1))
            batch.clear()
    if expected != count:
        raise ValueError("target row count differs from cache declaration")
    if batch:
        vectors = _normalize_batch(encoder.encode(batch), len(batch), encoder.dimension)
        matrix[count - len(batch):count] = vectors.astype(np.float16)
        matrix.flush()
        _checkpoint(checkpoint_path, dict(identity, completed=count))


def build_faiss_index(
    cache_path: str | Path, index_path: str | Path,
    count: int, dimension: int, config: DenseConfig,
) -> None:
    """Train IVF-PQ on a bounded sample; add all vectors in CPU-sized batches."""
    faiss = _import_faiss()
    index_path = Path(index_path)
    if index_path.exists():
        raise FileExistsError(index_path)
    if dimension % config.pq_subquantizers or count < config.nlist:
        raise ValueError("PQ subquantizers must divide dimension; count >= nlist")
    if count >= 2**31:
        raise ValueError("target sequence IDs exceed int32 candidate artifact range")
    if config.train_samples < config.nlist:
        raise ValueError("train_samples must be at least nlist")
    if Path(cache_path).stat().st_size != count * dimension * 2:
        raise ValueError("dense embedding cache size mismatch")
    embeddings = np.memmap(cache_path, dtype=np.float16, mode="r",
                           shape=(count, dimension))
    sample_count = min(count, config.train_samples)
    # Evenly spaced indices avoid a source-order-biased prefix sample.
    sample_indices = np.linspace(0, count - 1, sample_count, dtype=np.int64)
    sample = np.asarray(embeddings[sample_indices], dtype=np.float32)
    faiss.normalize_L2(sample)
    faiss.omp_set_num_threads(config.threads)
    index = faiss.IndexIVFPQ(
        faiss.IndexFlatIP(dimension), dimension, config.nlist,
        config.pq_subquantizers, 8, faiss.METRIC_INNER_PRODUCT,
    )
    index.train(sample)
    for start in range(0, count, config.batch_size):
        batch = np.asarray(embeddings[start:start + config.batch_size], dtype=np.float32)
        faiss.normalize_L2(batch)
        index.add(batch)
    if index.ntotal != count:
        raise ValueError("dense index did not contain every target")
    temporary = index_path.with_name(index_path.name + ".building")
    if temporary.exists():
        raise FileExistsError(temporary)
    faiss.write_index(index, str(temporary))
    os.replace(temporary, index_path)


def query_faiss_index(
    rows: Iterable[tuple[int, str, str]], count: int, encoder: Encoder,
    index_path: str | Path, candidate_path: str | Path,
    score_path: str | Path, checkpoint_path: str | Path,
    config: DenseConfig, input_identity: str,
) -> None:
    """Write bounded top-K S1 arrays; resume only after flushed batches."""
    faiss = _import_faiss()
    candidate_path, score_path = Path(candidate_path), Path(score_path)
    checkpoint_path = Path(checkpoint_path)
    if count < 1 or not input_identity:
        raise ValueError("dense query requires S1 rows and input identity")
    index = faiss.read_index(str(index_path))
    if index.d != encoder.dimension:
        raise ValueError("dense index and encoder dimensions differ")
    faiss.omp_set_num_threads(config.threads)
    index.nprobe = config.nprobe
    index_stat = Path(index_path).stat()
    identity = {"count": count, "dimension": encoder.dimension,
                "top_k": config.top_k, "nprobe": config.nprobe,
                "index_path": str(Path(index_path).resolve()),
                "index_bytes": index_stat.st_size,
                "index_mtime_ns": index_stat.st_mtime_ns,
                "encoder_revision": getattr(encoder, "revision", None),
                "input_identity": input_identity}
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if {key: state.get(key) for key in identity} != identity:
            raise ValueError("dense query checkpoint configuration mismatch")
        expected_bytes = count * config.top_k * 4
        if (not candidate_path.exists() or not score_path.exists()
                or candidate_path.stat().st_size != expected_bytes
                or score_path.stat().st_size != expected_bytes):
            raise ValueError("dense query arrays are missing or have wrong size")
        completed = state["completed"]
        if not isinstance(completed, int) or not 0 <= completed <= count:
            raise ValueError("dense query checkpoint position is invalid")
        mode = "r+"
    else:
        if candidate_path.exists() or score_path.exists():
            raise FileExistsError("dense query output exists without checkpoint")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        completed, mode = 0, "w+"
    indices = np.memmap(candidate_path, dtype=np.int32, mode=mode,
                        shape=(count, config.top_k))
    scores = np.memmap(score_path, dtype=np.float32, mode=mode,
                       shape=(count, config.top_k))
    if not checkpoint_path.exists():
        indices[:] = -1
        scores[:] = -np.inf
        indices.flush()
        scores.flush()
        _checkpoint(checkpoint_path, dict(identity, completed=0))
    batch: list[str] = []
    expected = 0
    for seq, name, address in rows:
        if seq != expected:
            raise ValueError("S1 query rows must have contiguous sequence IDs")
        expected += 1
        if seq < completed:
            continue
        batch.append(dense_text(name, address))
        if len(batch) == config.batch_size:
            vectors = _normalize_batch(encoder.encode(batch), len(batch), encoder.dimension)
            distances, matches = index.search(vectors, config.top_k)
            start = seq + 1 - len(batch)
            indices[start:seq + 1] = matches.astype(np.int32)
            scores[start:seq + 1] = distances.astype(np.float32)
            indices.flush()
            scores.flush()
            _checkpoint(checkpoint_path, dict(identity, completed=seq + 1))
            batch.clear()
    if expected != count:
        raise ValueError("S1 row count differs from dense query declaration")
    if batch:
        vectors = _normalize_batch(encoder.encode(batch), len(batch), encoder.dimension)
        distances, matches = index.search(vectors, config.top_k)
        indices[count - len(batch):count] = matches.astype(np.int32)
        scores[count - len(batch):count] = distances.astype(np.float32)
        indices.flush()
        scores.flush()
        _checkpoint(checkpoint_path, dict(identity, completed=count))


def merge_dense_candidates(
    source_id: str, lexical: Sequence[Candidate],
    dense_hits: Sequence[tuple[str, float]], max_candidates: int,
) -> tuple[Candidate, ...]:
    """Union one S1's dense and lexical hits without duplicate candidate IDs."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    by_id: dict[str, list[ChannelHit]] = {}
    for candidate in lexical:
        if candidate.source1_entity_id != source_id or candidate.candidate_entity_id in by_id:
            raise ValueError("lexical candidates must be unique and share S1 ID")
        by_id[candidate.candidate_entity_id] = list(candidate.hits)
    for rank, (target_id, score) in enumerate(dense_hits, 1):
        if not target_id or not math.isfinite(score):
            raise ValueError("dense hits need valid IDs and finite scores")
        hits = by_id.setdefault(target_id, [])
        if not any(hit.channel == DENSE_CHANNEL for hit in hits):
            hits.append(ChannelHit(DENSE_CHANNEL, rank, float(score)))
    ranked = sorted(by_id.items(), key=lambda item: (
        -len(item[1]), -max(hit.score for hit in item[1]),
        min(hit.rank for hit in item[1]), item[0],
    ))[:max_candidates]
    return tuple(
        Candidate(source_id, target_id, rank, max(hit.score for hit in hits),
                  tuple(hits))
        for rank, (target_id, hits) in enumerate(ranked, 1)
    )


@dataclass(frozen=True)
class DenseRetrievalDelta:
    entities: int
    truth_links: int
    base_recovered: int
    union_recovered: int
    dense_unique_truth_recovered: int
    base_candidate_recall: float
    union_candidate_recall: float
    base_oracle_macro_f05: float
    union_oracle_macro_f05: float
    average_base_candidates: float
    average_union_candidates: float


def evaluate_dense_union(
    rows: Iterable[tuple[set[str] | frozenset[str], set[str], set[str]]],
    max_candidates: int | None = None,
) -> DenseRetrievalDelta:
    """Training-only retrieval proxy; never calls a pair/ranking model.

    Input rows are (ground truth, retained lexical IDs, retained dense IDs).
    If a cap is used upstream, pass the already capped candidate sets here.
    """
    entities = truth_links = base_recovered = union_recovered = 0
    base_candidates = union_candidates = 0
    base_oracle = union_oracle = 0.0
    for truth, lexical, dense in rows:
        if not isinstance(truth, (set, frozenset)) or not isinstance(lexical, set) or not isinstance(dense, set):
            raise ValueError("dense evaluation requires truth, lexical and dense ID sets")
        union = lexical | dense
        if max_candidates is not None and len(union) > max_candidates:
            raise ValueError("dense union exceeds the declared candidate cap")
        entities += 1
        truth_links += len(truth)
        base_hits, union_hits = truth & lexical, truth & union
        base_recovered += len(base_hits)
        union_recovered += len(union_hits)
        base_candidates += len(lexical)
        union_candidates += len(union)
        base_oracle += entity_f05(truth, base_hits)
        union_oracle += entity_f05(truth, union_hits)
    if not entities:
        raise ValueError("dense evaluation needs at least one S1 entity")
    return DenseRetrievalDelta(
        entities, truth_links, base_recovered, union_recovered,
        union_recovered - base_recovered,
        base_recovered / truth_links if truth_links else 1.0,
        union_recovered / truth_links if truth_links else 1.0,
        base_oracle / entities, union_oracle / entities,
        base_candidates / entities, union_candidates / entities,
    )


def evaluate_dense_training_store(
    connection, base_work_dir: str | Path, dense_candidate_path: str | Path,
    top_k: int, dense_top_k: int,
) -> DenseRetrievalDelta:
    """Stream Phase 4 train-only truth and lexical/dense arrays for a proxy."""
    from .phase4_store import candidate_array
    from .retrieval import CHANNELS

    source_count = connection.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
    base_work_dir = Path(base_work_dir)
    base = {
        channel: candidate_array(base_work_dir / f"{channel}.int32",
                                 source_count, top_k, create=False)
        for channel in CHANNELS
    }
    dense = candidate_array(dense_candidate_path, source_count,
                            dense_top_k, create=False)
    truth_cursor = connection.execute(
        "SELECT source_seq,target_seq FROM truth_indexed ORDER BY source_seq,target_seq"
    )
    next_truth = truth_cursor.fetchone()

    def rows():
        nonlocal next_truth
        for seq in range(source_count):
            truth: set[int] = set()
            while next_truth is not None and next_truth[0] == seq:
                truth.add(next_truth[1])
                next_truth = truth_cursor.fetchone()
            lexical = {int(target) for values in base.values()
                       for target in values[seq] if target >= 0}
            semantic = {int(target) for target in dense[seq] if target >= 0}
            yield truth, lexical, semantic
        if next_truth is not None:
            raise ValueError("truth contains a source outside the S1 table")

    return evaluate_dense_union(rows())


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional CPU dense retrieval stages")
    parser.add_argument("--store", type=Path, required=True,
                        help="Phase 4 SQLite store; never modified")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--revision", default="",
                        help="pinned encoder commit revision, required for encode/query")
    parser.add_argument("--base-work-dir", type=Path,
                        help="required only for train-only dense union evaluation")
    parser.add_argument("--stage", choices=("encode", "index", "query", "evaluate", "all"),
                        default="all")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--base-top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--pq-subquantizers", type=int, default=48)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    config = DenseConfig(args.top_k, args.batch_size, args.nlist,
                         args.pq_subquantizers, args.nprobe,
                         args.train_samples, args.threads)
    from .phase4_store import open_store

    args.work_dir.mkdir(parents=True, exist_ok=True)
    cache = args.work_dir / "target.float16"
    cache_state = args.work_dir / "target.json"
    index_path = args.work_dir / "target.faiss"
    hits = args.work_dir / "dense.int32"
    scores = args.work_dir / "dense.float32"
    hit_state = args.work_dir / "query.json"
    with closing(open_store(args.store)) as connection:
        store_stat = args.store.stat()
        input_identity = (
            f"{args.store.resolve()}:{store_stat.st_size}:{store_stat.st_mtime_ns}"
        )
        target_count = connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
        source_count = connection.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
        if args.stage in ("encode", "all"):
            if not args.revision:
                parser.error("--revision is required for dense encoding")
            encoder = SentenceTransformerEncoder(args.revision,
                                                 threads=args.threads)
            encode_target_cache(connection.execute(
                "SELECT seq,raw_name,raw_address FROM targets ORDER BY seq"),
                target_count, encoder, cache, cache_state, config,
                args.revision, input_identity)
            del encoder
            gc.collect()
        if args.stage in ("index", "all"):
            state = json.loads(cache_state.read_text(encoding="utf-8"))
            if state["completed"] != target_count:
                raise ValueError("dense target encoding is not complete")
            if not index_path.exists():
                build_faiss_index(cache, index_path, target_count,
                                  state["dimension"], config)
        if args.stage in ("query", "all"):
            if not args.revision:
                parser.error("--revision is required for dense querying")
            state = json.loads(cache_state.read_text(encoding="utf-8"))
            if state["revision"] != args.revision:
                raise ValueError("query encoder revision differs from dense index cache")
            encoder = SentenceTransformerEncoder(args.revision,
                                                 threads=args.threads)
            query_faiss_index(connection.execute(
                "SELECT seq,raw_name,raw_address FROM source1 ORDER BY seq"),
                source_count, encoder, index_path, hits, scores, hit_state,
                config, input_identity)
            del encoder
            gc.collect()
        if args.stage in ("evaluate", "all") and args.base_work_dir:
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='truth_indexed'"
            ).fetchone() is None:
                raise ValueError("dense evaluation requires training ground truth")
            result = evaluate_dense_training_store(
                connection, args.base_work_dir, hits,
                args.base_top_k, config.top_k)
            report = args.work_dir / "dense_union_metrics.json"
            with report.open("x", encoding="utf-8") as handle:
                json.dump(result.__dict__, handle, indent=2, sort_keys=True)
                handle.write("\n")
        elif args.stage == "evaluate":
            parser.error("--base-work-dir is required for train-only evaluation")


if __name__ == "__main__":
    main()
