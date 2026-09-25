"""Bounded, global-IDF character TF-IDF retrieval for full training data."""

from __future__ import annotations

import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from .phase4_store import candidate_array
from .retrieval import RetrievalConfig


def _weighted(matrix: sparse.csr_matrix, idf: np.ndarray) -> sparse.csr_matrix:
    matrix.data *= idf[matrix.indices]
    return normalize(matrix, norm="l2", copy=False)


def _vocabulary_and_idf(connection: sqlite3.Connection, view: str,
                        config: RetrievalConfig) -> tuple[dict[str, int], np.ndarray]:
    analyzer = TfidfVectorizer(
        analyzer="char", ngram_range=config.char_ngram_range
    ).build_analyzer()
    term_frequency = Counter()
    document_frequency = Counter()
    target_count = 0
    for (value,) in connection.execute(f"SELECT {view} FROM targets ORDER BY seq"):
        ngrams = analyzer(value)
        term_frequency.update(ngrams)
        document_frequency.update(set(ngrams))
        target_count += 1
    # TfidfVectorizer uses the most frequent features when max_features is set.
    # Lexical ordering resolves equal-frequency terms deterministically.
    selected = sorted(term_frequency, key=lambda term: (-term_frequency[term], term))[
        :config.max_features
    ]
    selected.sort()
    vocabulary = {term: position for position, term in enumerate(selected)}
    idf = np.array([
        np.log((1 + target_count) / (1 + document_frequency[term])) + 1
        for term in selected
    ], dtype=np.float32)
    return vocabulary, idf


def _query_chunks(connection: sqlite3.Connection, view: str, vectorizer: CountVectorizer,
                  idf: np.ndarray, rows_per_chunk: int,
                  cache_dir: Path) -> list[tuple[Path, int]]:
    cursor = connection.execute(f"SELECT {view} FROM source1 ORDER BY seq")
    chunks = []
    while rows := cursor.fetchmany(rows_per_chunk):
        matrix = vectorizer.transform([row[0] for row in rows]).tocsr()
        path = cache_dir / f"query_{len(chunks):04d}.npz"
        sparse.save_npz(path, _weighted(matrix, idf), compressed=False)
        chunks.append((path, len(rows)))
    return chunks


def _target_ids(connection: sqlite3.Connection) -> np.ndarray:
    max_length = connection.execute(
        "SELECT MAX(LENGTH(CAST(entity_id AS BLOB))) FROM targets"
    ).fetchone()[0]
    count = connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    return np.fromiter(
        (identifier.encode("utf-8") for (identifier,) in connection.execute(
            "SELECT entity_id FROM targets ORDER BY seq"
        )), dtype=f"S{max(max_length or 1, 1)}", count=count,
    )


def _boundary_tie_corrections(
    query_batch: sparse.csr_matrix, target_matrix: sparse.csr_matrix,
    output: sparse.csr_matrix, target_start: int, target_ids: np.ndarray,
    top_k: int, shard_top_n: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Resolve truncation at a tied shard boundary by scoring that row fully."""
    corrections = {}
    if shard_top_n <= top_k:
        return corrections
    transpose = target_matrix.T
    for row, count in enumerate(np.diff(output.indptr)):
        if count != shard_top_n:
            continue
        start, end = output.indptr[row], output.indptr[row + 1]
        returned_scores = np.sort(output.data[start:end])[::-1]
        if returned_scores[top_k - 1] != returned_scores[-1]:
            continue
        full = query_batch.getrow(row) @ transpose
        positive = full.data > 0
        local_indices = full.indices[positive]
        local_scores = full.data[positive]
        order = np.lexsort((target_ids[local_indices + target_start], -local_scores))[
            :top_k
        ]
        corrections[row] = (local_indices[order], local_scores[order])
    return corrections


def _merge_batch(indices: np.memmap, scores: np.memmap, output: sparse.csr_matrix,
                 query_start: int, target_start: int, target_ids: np.ndarray,
                 top_k: int, shard_top_n: int,
                 tie_corrections: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    rows = output.shape[0]
    new_indices = np.full((rows, shard_top_n), -1, dtype=np.int32)
    new_scores = np.full((rows, shard_top_n), -np.inf, dtype=np.float32)
    counts = np.diff(output.indptr)
    if output.nnz:
        row_positions = np.repeat(np.arange(rows), counts)
        col_positions = np.arange(output.nnz) - np.repeat(output.indptr[:-1], counts)
        new_indices[row_positions, col_positions] = output.indices + target_start
        new_scores[row_positions, col_positions] = output.data
    for row, (local_indices, local_scores) in tie_corrections.items():
        new_indices[row] = -1
        new_scores[row] = -np.inf
        new_indices[row, :len(local_indices)] = local_indices + target_start
        new_scores[row, :len(local_scores)] = local_scores
    current_indices = np.asarray(indices[query_start:query_start + rows])
    current_scores = np.asarray(scores[query_start:query_start + rows])
    all_indices = np.concatenate((current_indices, new_indices), axis=1)
    all_scores = np.concatenate((current_scores, new_scores), axis=1)
    all_scores[all_indices < 0] = -np.inf
    safe_indices = np.maximum(all_indices, 0)
    key_ids = target_ids[safe_indices]
    order = np.lexsort((key_ids, -all_scores), axis=1)[:, :top_k]
    indices[query_start:query_start + rows] = np.take_along_axis(all_indices, order, axis=1)
    scores[query_start:query_start + rows] = np.take_along_axis(all_scores, order, axis=1)


def char_channel(connection: sqlite3.Connection, view: str, candidate_path: str | Path,
                 score_path: str | Path, config: RetrievalConfig,
                 shard_size: int = 250_000, query_batch_size: int = 4096,
                 threads: int = 8) -> None:
    """Compute global-IDF cosine top-K using bounded target shards.

    Each target shard contributes its top 2K scores per query; rows truncated
    at a tied top-K boundary are rescored fully to preserve target-ID ordering.
    """
    if view not in ("name_clean", "address_alias"):
        raise ValueError("invalid character view")
    if min(shard_size, query_batch_size, threads) < 1:
        raise ValueError("resource limits must be positive")
    candidate_path, score_path = Path(candidate_path), Path(score_path)
    if candidate_path.exists() or score_path.exists():
        raise FileExistsError("character channel output already exists")
    source_count = connection.execute("SELECT COUNT(*) FROM source1").fetchone()[0]
    target_count = connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    candidates = candidate_array(candidate_path, source_count, config.top_k, create=True)
    scores = np.memmap(score_path, dtype=np.float32, mode="w+",
                       shape=(source_count, config.top_k))
    scores[:] = -np.inf
    if not source_count or not target_count:
        candidates.flush()
        scores.flush()
        return
    vocabulary, idf = _vocabulary_and_idf(connection, view, config)
    if not vocabulary:
        candidates.flush()
        scores.flush()
        return
    vectorizer = CountVectorizer(
        analyzer="char", ngram_range=config.char_ngram_range,
        vocabulary=vocabulary, dtype=np.float32,
    )
    target_ids = _target_ids(connection)
    shard_top_n = min(config.top_k * 2, shard_size)
    with tempfile.TemporaryDirectory(prefix=f"{view}_queries_",
                                     dir=candidate_path.parent) as cache_name:
        query_chunks = _query_chunks(connection, view, vectorizer, idf, 50_000,
                                     Path(cache_name))
        for target_start in range(0, target_count, shard_size):
            target_end = min(target_start + shard_size, target_count)
            texts = [row[0] for row in connection.execute(
                f"SELECT {view} FROM targets WHERE seq>=? AND seq<? ORDER BY seq",
                (target_start, target_end),
            )]
            target_matrix = _weighted(vectorizer.transform(texts).tocsr(), idf)
            del texts
            query_start = 0
            for chunk_path, chunk_rows in query_chunks:
                chunk = sparse.load_npz(chunk_path)
                for offset in range(0, chunk_rows, query_batch_size):
                    query_batch = chunk[offset:offset + query_batch_size]
                    output = sp_matmul_topn(
                        query_batch, target_matrix.T, top_n=shard_top_n,
                        threshold=0.0, n_threads=threads, sort=True,
                    )
                    corrections = _boundary_tie_corrections(
                        query_batch, target_matrix, output, target_start,
                        target_ids, config.top_k, shard_top_n,
                    )
                    _merge_batch(candidates, scores, output, query_start + offset,
                                 target_start, target_ids, config.top_k, shard_top_n,
                                 corrections)
                query_start += chunk_rows
                del chunk
            candidates.flush()
            scores.flush()
            del target_matrix
            print(f"{view}: completed target shard through {target_end:,}/"
                  f"{target_count:,} targets", flush=True)
    del target_ids
