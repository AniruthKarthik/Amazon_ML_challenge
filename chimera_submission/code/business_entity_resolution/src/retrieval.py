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
        for idx, row in target_df.iterrows():
            eid = row["entity_id"]
            clean_name = row.get("name_clean", "")
            core_name = row.get("name_core", "")
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
            sublinear_tf=True,
            dtype=np.float32,
        )
        self._target_addr_matrix = self._addr_vectorizer.fit_transform(addr_corpus)

        # 4. Rare-Token Inverted Index (from name tokens)
        token_doc_counts: Dict[str, int] = defaultdict(int)
        token_to_eids: Dict[str, Set[str]] = defaultdict(set)

        for _, row in target_df.iterrows():
            eid = row["entity_id"]
            tokens = set(row.get("name_clean", "").split())
            for t in tokens:
                if len(t) >= self.rare_token_min_len and not t.isdigit():
                    token_doc_counts[t] += 1
                    token_to_eids[t].add(eid)

        self._rare_token_index.clear()
        for token, count in token_doc_counts.items():
            if 1 <= count <= self.rare_token_max_doc_freq:
                self._rare_token_index[token] = list(token_to_eids[token])

        # 5. Optional Word TF-IDF Vectorizer (word n-grams 1-2)
        if self.enable_word_tfidf:
            self._word_vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                dtype=np.float32,
            )
            self._target_word_matrix = self._word_vectorizer.fit_transform(name_corpus)

        self._is_fitted = True
        return self

    def retrieve(self, s1_df: pd.DataFrame) -> Dict[str, Dict[str, CandidateProvenance]]:
        """Retrieve candidates for all S1 entities across all channels.

        Returns
        -------
        Dict[s1_id, Dict[target_id, CandidateProvenance]]
        """
        if not self._is_fitted:
            raise RuntimeError("CandidateRetriever must be fit before retrieval.")

        results: Dict[str, Dict[str, CandidateProvenance]] = {
            s1_id: {} for s1_id in s1_df["entity_id"]
        }

        # --- Channel 1 & 2: Exact Name & Exact Core Name ---
        for _, row in s1_df.iterrows():
            s1_id = row["entity_id"]
            clean_name = row.get("name_clean", "")
            core_name = row.get("name_core", "")

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

        # --- Channel 3: Char TF-IDF Name KNN (Batched Matrix Multiplication) ---
        if self._name_vectorizer and self._target_name_matrix is not None:
            s1_names = s1_df["name_clean"].fillna("").tolist()
            s1_name_mat = self._name_vectorizer.transform(s1_names)
            sim_mat = s1_name_mat.dot(self._target_name_matrix.T)

            s1_ids = s1_df["entity_id"].tolist()
            for row_idx in range(sim_mat.shape[0]):
                s1_id = s1_ids[row_idx]
                row_sim = sim_mat.getrow(row_idx)
                if row_sim.nnz == 0:
                    continue

                col_indices = row_sim.indices
                scores = row_sim.data

                # Filter by min threshold
                valid_mask = scores >= self.min_tfidf_score
                valid_cols = col_indices[valid_mask]
                valid_scores = scores[valid_mask]

                if len(valid_scores) > 0:
                    # Sort top K
                    if len(valid_scores) > self.top_k_name_tfidf:
                        top_order = np.argpartition(-valid_scores, self.top_k_name_tfidf)[
                            : self.top_k_name_tfidf
                        ]
                        top_order = top_order[np.argsort(-valid_scores[top_order])]
                    else:
                        top_order = np.argsort(-valid_scores)

                    for rank, idx in enumerate(top_order, start=1):
                        target_id = self._target_ids[valid_cols[idx]]
                        score = float(valid_scores[idx])
                        prov = self._get_or_create(results, s1_id, target_id)
                        prov.channels.add("tfidf_name")
                        prov.scores["tfidf_name"] = score
                        prov.ranks["tfidf_name"] = rank

        # --- Channel 4: Char TF-IDF Address KNN ---
        if self._addr_vectorizer and self._target_addr_matrix is not None:
            s1_addrs = s1_df["address_clean"].fillna("").tolist()
            # Only run address KNN if address is non-empty
            non_empty_indices = [i for i, a in enumerate(s1_addrs) if a.strip()]
            if non_empty_indices:
                sub_addrs = [s1_addrs[i] for i in non_empty_indices]
                s1_addr_mat = self._addr_vectorizer.transform(sub_addrs)
                sim_addr_mat = s1_addr_mat.dot(self._target_addr_matrix.T)

                for local_idx, orig_idx in enumerate(non_empty_indices):
                    s1_id = s1_df["entity_id"].iloc[orig_idx]
                    row_sim = sim_addr_mat.getrow(local_idx)
                    if row_sim.nnz == 0:
                        continue

                    col_indices = row_sim.indices
                    scores = row_sim.data
                    valid_mask = scores >= 0.40  # higher threshold for address
                    valid_cols = col_indices[valid_mask]
                    valid_scores = scores[valid_mask]

                    if len(valid_scores) > 0:
                        if len(valid_scores) > self.top_k_addr_tfidf:
                            top_order = np.argpartition(-valid_scores, self.top_k_addr_tfidf)[
                                : self.top_k_addr_tfidf
                            ]
                            top_order = top_order[np.argsort(-valid_scores[top_order])]
                        else:
                            top_order = np.argsort(-valid_scores)

                        for rank, idx in enumerate(top_order, start=1):
                            target_id = self._target_ids[valid_cols[idx]]
                            score = float(valid_scores[idx])
                            prov = self._get_or_create(results, s1_id, target_id)
                            prov.channels.add("tfidf_addr")
                            prov.scores["tfidf_addr"] = score
                            prov.ranks["tfidf_addr"] = rank

        # --- Channel 6: Word TF-IDF Name KNN (Conditional) ---
        if self.enable_word_tfidf and self._word_vectorizer and self._target_word_matrix is not None:
            s1_names = s1_df["name_clean"].fillna("").tolist()
            s1_word_mat = self._word_vectorizer.transform(s1_names)
            sim_word_mat = s1_word_mat.dot(self._target_word_matrix.T)

            s1_ids = s1_df["entity_id"].tolist()
            for row_idx in range(sim_word_mat.shape[0]):
                s1_id = s1_ids[row_idx]
                row_sim = sim_word_mat.getrow(row_idx)
                if row_sim.nnz == 0:
                    continue

                col_indices = row_sim.indices
                scores = row_sim.data
                valid_mask = scores >= self.min_word_tfidf_score
                valid_cols = col_indices[valid_mask]
                valid_scores = scores[valid_mask]

                if len(valid_scores) > 0:
                    if len(valid_scores) > self.top_k_word_tfidf:
                        top_order = np.argpartition(-valid_scores, self.top_k_word_tfidf)[
                            : self.top_k_word_tfidf
                        ]
                        top_order = top_order[np.argsort(-valid_scores[top_order])]
                    else:
                        top_order = np.argsort(-valid_scores)

                    for rank, idx in enumerate(top_order, start=1):
                        target_id = self._target_ids[valid_cols[idx]]
                        score = float(valid_scores[idx])
                        prov = self._get_or_create(results, s1_id, target_id)
                        prov.channels.add("tfidf_word")
                        prov.scores["tfidf_word"] = score
                        prov.ranks["tfidf_word"] = rank

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
