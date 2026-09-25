"""Exact and lexical candidate retrieval channels for business entity resolution.

Implements:
1. Exact Normalized Name channel
2. Exact Core Name channel
3. Character TF-IDF Name KNN channel
4. Character TF-IDF Address KNN channel
5. Rare-Token Inverted Index channel
6. Provenance tracking, deduplication, and candidate export.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import multiprocessing as mp
import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class CandidateProvenance:
    s1_id: str
    target_id: str
    channels: Set[str] = field(default_factory=set)
    scores: Dict[str, float] = field(default_factory=dict)
    ranks: Dict[str, int] = field(default_factory=dict)

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def max_score(self) -> float:
        return max(self.scores.values()) if self.scores else 0.0

    @property
    def min_rank(self) -> int:
        return min(self.ranks.values()) if self.ranks else 999999


class CandidateRetriever:
    """Multi-channel candidate generator linking S1 records to S2/S3 candidates."""

    def __init__(
        self,
        top_k_name_tfidf: int = 20,
        top_k_addr_tfidf: int = 15,
        min_tfidf_score: float = 0.25,
        rare_token_max_doc_freq: int = 50,
        rare_token_min_len: int = 4,
        max_candidates_per_entity: int = 60,
        enable_word_tfidf: bool = False,
        top_k_word_tfidf: int = 15,
        min_word_tfidf_score: float = 0.30,
    ):
        self.top_k_name_tfidf = top_k_name_tfidf
        self.top_k_addr_tfidf = top_k_addr_tfidf
        self.min_tfidf_score = min_tfidf_score
        self.rare_token_max_doc_freq = rare_token_max_doc_freq
        self.rare_token_min_len = rare_token_min_len
        self.max_candidates_per_entity = max_candidates_per_entity
        self.enable_word_tfidf = enable_word_tfidf
        self.top_k_word_tfidf = top_k_word_tfidf
        self.min_word_tfidf_score = min_word_tfidf_score

        # Exact index mappings: normalized_text -> list of target_ids
        self._exact_clean_name_index: Dict[str, List[str]] = defaultdict(list)
        self._exact_core_name_index: Dict[str, List[str]] = defaultdict(list)

        # Rare token inverted index: token -> list of target_ids
        self._rare_token_index: Dict[str, List[str]] = defaultdict(list)

        # TF-IDF models and matrices
        self._name_vectorizer: Optional[TfidfVectorizer] = None
        self._target_name_matrix: Optional[csr_matrix] = None

        self._addr_vectorizer: Optional[TfidfVectorizer] = None
        self._target_addr_matrix: Optional[csr_matrix] = None

        self._word_vectorizer: Optional[TfidfVectorizer] = None
        self._target_word_matrix: Optional[csr_matrix] = None

        self._target_ids: List[str] = []
        self._target_id_to_idx: Dict[str, int] = {}
        self._is_fitted: bool = False

    def fit(self, target_df: pd.DataFrame) -> CandidateRetriever:
        """Index the target corpus (combined S2 and S3 records).

        Expects columns: entity_id, name_clean, name_folded, name_core, address_clean, address_alias.
        """
        self._target_ids = target_df["entity_id"].tolist()
        self._target_id_to_idx = {eid: idx for idx, eid in enumerate(self._target_ids)}

        # 1. Exact clean name & core name inverted indices
        self._exact_clean_name_index.clear()
        self._exact_core_name_index.clear()
        target_eids = target_df["entity_id"].tolist()
        clean_names = (
            target_df["name_clean"].fillna("").tolist()
            if "name_clean" in target_df.columns
            else [""] * len(target_df)
        )
        core_names = (
            target_df["name_core"].fillna("").tolist()
            if "name_core" in target_df.columns
            else [""] * len(target_df)
        )
        for eid, clean_name, core_name in zip(target_eids, clean_names, core_names):
            if clean_name:
                self._exact_clean_name_index[clean_name].append(eid)
            if core_name:
                self._exact_core_name_index[core_name].append(eid)

        # 2. Character TF-IDF Name Vectorizer (char_wb 3-4 grams)
        name_corpus = target_df["name_clean"].fillna("").tolist()
        self._name_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=1,
            max_features=80000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self._target_name_matrix = self._name_vectorizer.fit_transform(name_corpus)

        # 3. Character TF-IDF Address Vectorizer (char_wb 3-5 grams)
        addr_corpus = target_df["address_clean"].fillna("").tolist()
        self._addr_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=60000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self._target_addr_matrix = self._addr_vectorizer.fit_transform(addr_corpus)

        # 4. Rare-Token Inverted Index (Two-Pass Memory Optimization)
        # Pass 1: count doc frequencies without storing entity IDs (avoids gigabytes of sets)
        token_doc_counts: Dict[str, int] = defaultdict(int)
        target_eids = target_df["entity_id"].tolist()
        target_names = (
            target_df["name_clean"].fillna("").tolist() if "name_clean" in target_df.columns else [""] * len(target_df)
        )
        for name_clean in target_names:
            tokens = set(name_clean.split())
            for t in tokens:
                if len(t) >= self.rare_token_min_len and not t.isdigit():
                    token_doc_counts[t] += 1

        rare_tokens = {
            t for t, count in token_doc_counts.items()
            if 1 <= count <= self.rare_token_max_doc_freq
        }
        del token_doc_counts

        # Pass 2: only record entity IDs for genuine rare tokens
        self._rare_token_index.clear()
        for eid, name_clean in zip(target_eids, target_names):
            tokens = set(name_clean.split())
            for t in tokens:
                if t in rare_tokens:
                    self._rare_token_index[t].append(eid)
        del rare_tokens

        # 5. Optional Word TF-IDF Vectorizer (word n-grams 1-2)
        if self.enable_word_tfidf:
            self._word_vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                max_features=60000,
                sublinear_tf=True,
                dtype=np.float32,
            )
            self._target_word_matrix = self._word_vectorizer.fit_transform(name_corpus)

        self._is_fitted = True
        return self

    def retrieve(
        self, s1_df: pd.DataFrame, verbose: bool = True, n_jobs: int = -1
    ) -> Dict[str, Dict[str, CandidateProvenance]]:
        """Retrieve candidates for all S1 entities across all channels with multi-core parallelism.

        Parameters
        ----------
        s1_df : pd.DataFrame with normalized S1 entities.
        verbose : whether to display live channel progress status.
        n_jobs : number of parallel worker processes (-1 for all available cores).

        Returns
        -------
        Dict[s1_id, Dict[target_id, CandidateProvenance]]
        """
        if not self._is_fitted:
            raise RuntimeError("CandidateRetriever must be fit before retrieval.")

        total_s1 = len(s1_df)
        if total_s1 == 0:
            return {}

        if verbose:
            print(
                f"  [Candidate Retrieval] Querying {total_s1} entities across exact, rare-token, and sparse TF-IDF channels..."
            )

        results = self._retrieve_single(s1_df, verbose=verbose)

        total_candidates = sum(len(cands) for cands in results.values())
        if verbose:
            print(
                f"  [Candidate Retrieval] Collected {total_candidates} candidate pairs across {len(results)} entities."
            )

        return results

    def _retrieve_tfidf_channel_batched(
        self,
        results: Dict[str, Dict[str, CandidateProvenance]],
        vectorizer: TfidfVectorizer,
        target_matrix: csr_matrix,
        query_texts: List[str],
        query_s1_ids: List[str],
        min_score: float,
        top_k: int,
        channel_name: str,
        stage_desc: str,
        batch_size: int = 200,
        verbose: bool = True,
    ) -> None:
        """Execute TF-IDF KNN query in streaming memory-bounded batches to prevent OOM."""
        total_queries = len(query_texts)
        if total_queries == 0 or target_matrix is None:
            return

        # target_matrix.T is a zero-copy CSC view sharing underlying arrays
        target_matrix_t = target_matrix.T

        log_interval = max(batch_size * 5, 2000)

        for b_start in range(0, total_queries, batch_size):
            b_end = min(b_start + batch_size, total_queries)
            b_texts = query_texts[b_start:b_end]
            b_ids = query_s1_ids[b_start:b_end]

            sub_mat = vectorizer.transform(b_texts)
            sim_mat = sub_mat.dot(target_matrix_t)

            for local_idx in range(len(b_texts)):
                s1_id = b_ids[local_idx]
                s = sim_mat.indptr[local_idx]
                e = sim_mat.indptr[local_idx + 1]
                if s == e:
                    continue

                scores = sim_mat.data[s:e]
                valid_mask = scores >= min_score
                if not np.any(valid_mask):
                    continue

                col_indices = sim_mat.indices[s:e]
                valid_cols = col_indices[valid_mask]
                valid_scores = scores[valid_mask]

                if len(valid_scores) > top_k:
                    top_order = np.argpartition(-valid_scores, top_k)[:top_k]
                    top_order = top_order[np.argsort(-valid_scores[top_order])]
                else:
                    top_order = np.argsort(-valid_scores)

                for rank, idx in enumerate(top_order, start=1):
                    target_id = self._target_ids[valid_cols[idx]]
                    score = float(valid_scores[idx])
                    prov = self._get_or_create(results, s1_id, target_id)
                    prov.channels.add(channel_name)
                    prov.scores[channel_name] = score
                    prov.ranks[channel_name] = rank

            del sub_mat
            del sim_mat

            if verbose and (b_end % log_interval == 0 or b_end == total_queries or b_end <= batch_size):
                pct = 100.0 * b_end / total_queries
                print(
                    f"\r  [{stage_desc}] {b_end}/{total_queries} queries ({pct:.1f}%)",
                    end="",
                    flush=True,
                )

        if verbose and total_queries > 0:
            print(f"\r  [{stage_desc}] Completed {total_queries}/{total_queries} queries.         ")

    def _retrieve_single(
        self, s1_df: pd.DataFrame, verbose: bool = False
    ) -> Dict[str, Dict[str, CandidateProvenance]]:
        results: Dict[str, Dict[str, CandidateProvenance]] = {
            s1_id: {} for s1_id in s1_df["entity_id"]
        }

        # --- Channel 1 & 2: Exact Name, Exact Core Name, and Rare Tokens ---
        total_s1 = len(s1_df)
        log_s1 = max(200, total_s1 // 20) if total_s1 > 0 else 1
        s1_ids_list = s1_df["entity_id"].tolist()
        clean_names_list = (
            s1_df["name_clean"].fillna("").tolist() if "name_clean" in s1_df.columns else [""] * total_s1
        )
        core_names_list = (
            s1_df["name_core"].fillna("").tolist() if "name_core" in s1_df.columns else [""] * total_s1
        )

        for idx, (s1_id, clean_name, core_name) in enumerate(
            zip(s1_ids_list, clean_names_list, core_names_list)
        ):
            if verbose and ((idx + 1) % log_s1 == 0 or (idx + 1) == total_s1 or (idx + 1) <= 5):
                pct = 100.0 * (idx + 1) / total_s1 if total_s1 > 0 else 100.0
                print(f"\r  [Candidate Retrieval 1/4: Exact & Rare] {idx + 1}/{total_s1} entities ({pct:.1f}%)", end="", flush=True)

            # Channel 1: Exact clean name
            if clean_name and clean_name in self._exact_clean_name_index:
                for target_id in self._exact_clean_name_index[clean_name]:
                    prov = self._get_or_create(results, s1_id, target_id)
                    prov.channels.add("exact_name")
                    prov.scores["exact_name"] = 1.0
                    prov.ranks.setdefault("exact_name", 1)

            # Channel 2: Exact core name
            if core_name and core_name in self._exact_core_name_index:
                for target_id in self._exact_core_name_index[core_name]:
                    prov = self._get_or_create(results, s1_id, target_id)
                    prov.channels.add("exact_core")
                    prov.scores["exact_core"] = 1.0
                    prov.ranks.setdefault("exact_core", 1)

            # Channel 5: Rare Token Index
            tokens = set(clean_name.split())
            for t in tokens:
                if t in self._rare_token_index:
                    for target_id in self._rare_token_index[t]:
                        prov = self._get_or_create(results, s1_id, target_id)
                        prov.channels.add("rare_token")
                        prov.scores.setdefault("rare_token", 0.5)
                        prov.ranks.setdefault("rare_token", 1)

        if verbose and total_s1 > 0:
            print(f"\r  [Candidate Retrieval 1/4: Exact & Rare] Completed {total_s1}/{total_s1} entities.         ")

        # --- Channel 3: Char TF-IDF Name KNN (Streaming Batches) ---
        if self._name_vectorizer and self._target_name_matrix is not None:
            s1_names = s1_df["name_clean"].fillna("").tolist()
            s1_ids = s1_df["entity_id"].tolist()
            self._retrieve_tfidf_channel_batched(
                results=results,
                vectorizer=self._name_vectorizer,
                target_matrix=self._target_name_matrix,
                query_texts=s1_names,
                query_s1_ids=s1_ids,
                min_score=self.min_tfidf_score,
                top_k=self.top_k_name_tfidf,
                channel_name="tfidf_name",
                stage_desc="Candidate Retrieval 2/4: Char TF-IDF",
                batch_size=200,
                verbose=verbose,
            )

        # --- Channel 4: Char TF-IDF Address KNN (Streaming Batches) ---
        if self._addr_vectorizer and self._target_addr_matrix is not None:
            s1_addrs = s1_df["address_clean"].fillna("").tolist()
            s1_ids = s1_df["entity_id"].tolist()
            valid_pairs = [(a, eid) for a, eid in zip(s1_addrs, s1_ids) if a.strip()]
            if valid_pairs:
                query_addrs = [p[0] for p in valid_pairs]
                query_ids = [p[1] for p in valid_pairs]
                self._retrieve_tfidf_channel_batched(
                    results=results,
                    vectorizer=self._addr_vectorizer,
                    target_matrix=self._target_addr_matrix,
                    query_texts=query_addrs,
                    query_s1_ids=query_ids,
                    min_score=0.40,
                    top_k=self.top_k_addr_tfidf,
                    channel_name="tfidf_addr",
                    stage_desc="Candidate Retrieval 3/4: Address TF-IDF",
                    batch_size=200,
                    verbose=verbose,
                )

        # --- Channel 6: Word TF-IDF Name KNN (Conditional) ---
        if self.enable_word_tfidf and self._word_vectorizer and self._target_word_matrix is not None:
            s1_names = s1_df["name_clean"].fillna("").tolist()
            s1_ids = s1_df["entity_id"].tolist()
            self._retrieve_tfidf_channel_batched(
                results=results,
                vectorizer=self._word_vectorizer,
                target_matrix=self._target_word_matrix,
                query_texts=s1_names,
                query_s1_ids=s1_ids,
                min_score=self.min_word_tfidf_score,
                top_k=self.top_k_word_tfidf,
                channel_name="tfidf_word",
                stage_desc="Candidate Retrieval 4/4: Word TF-IDF",
                batch_size=200,
                verbose=verbose,
            )

        # Deduplicate and Cap Candidates per S1 entity
        for s1_id, target_map in results.items():
            if len(target_map) > self.max_candidates_per_entity:
                # Rank candidates by:
                # 1. Channel count (descending)
                # 2. Max score (descending)
                # 3. Min rank (ascending)
                sorted_targets = sorted(
                    target_map.items(),
                    key=lambda item: (
                        item[1].channel_count,
                        item[1].max_score,
                        -item[1].min_rank,
                    ),
                    reverse=True,
                )[: self.max_candidates_per_entity]
                results[s1_id] = dict(sorted_targets)

        total_candidates = sum(len(cands) for cands in results.values())
        if verbose:
            print(f"  [Candidate Retrieval] Collected {total_candidates} candidate pairs across {len(results)} entities.")

        return results

    @staticmethod
    def _get_or_create(
        results: Dict[str, Dict[str, CandidateProvenance]],
        s1_id: str,
        target_id: str,
    ) -> CandidateProvenance:
        if target_id not in results[s1_id]:
            results[s1_id][target_id] = CandidateProvenance(s1_id=s1_id, target_id=target_id)
        return results[s1_id][target_id]

    def to_dataframe(
        self, candidates: Dict[str, Dict[str, CandidateProvenance]]
    ) -> pd.DataFrame:
        """Convert candidate results to tabular pairs DataFrame."""
        rows = []
        for s1_id, targets in candidates.items():
            for target_id, prov in targets.items():
                rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": target_id,
                    "channels": ",".join(sorted(prov.channels)),
                    "channel_count": prov.channel_count,
                    "max_score": prov.max_score,
                    "min_rank": prov.min_rank,
                    "score_exact_name": prov.scores.get("exact_name", 0.0),
                    "score_exact_core": prov.scores.get("exact_core", 0.0),
                    "score_tfidf_name": prov.scores.get("tfidf_name", 0.0),
                    "score_tfidf_word": prov.scores.get("tfidf_word", 0.0),
                    "score_tfidf_addr": prov.scores.get("tfidf_addr", 0.0),
                    "score_rare_token": prov.scores.get("rare_token", 0.0),
                })
        if not rows:
            return pd.DataFrame(columns=[
                "source1_entity_id", "candidate_entity_id", "channels", "channel_count",
                "max_score", "min_rank", "score_exact_name", "score_exact_core",
                "score_tfidf_name", "score_tfidf_word", "score_tfidf_addr", "score_rare_token",
            ])
        return pd.DataFrame(rows)

    def to_candidate_pairs_tsv(
        self,
        candidates: Dict[str, Dict[str, CandidateProvenance]],
        all_s1_ids: Set[str],
    ) -> pd.DataFrame:
        """Convert to official candidate_pairs.tsv format (source1_entity_id, candidate_entity_ids)."""
        rows = []
        for s1_id in sorted(all_s1_ids):
            target_ids = list(candidates.get(s1_id, {}).keys())
            rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(target_ids) if target_ids else "",
            })
        return pd.DataFrame(rows)


_worker_retriever: Optional[CandidateRetriever] = None


def _init_retriever_worker(retriever: CandidateRetriever) -> None:
    """Initialize worker process with pre-fitted retriever instance."""
    global _worker_retriever
    _worker_retriever = retriever


def _retrieve_chunk_worker(chunk_df: pd.DataFrame) -> Dict[str, Dict[str, CandidateProvenance]]:
    """Worker function executing multi-channel retrieval on a DataFrame chunk."""
    global _worker_retriever
    assert _worker_retriever is not None, "Worker retriever not initialized!"
    return _worker_retriever._retrieve_single(chunk_df, verbose=False)

