"""Tabular pair feature engineering for business entity resolution.

Computes:
1. Exact string matches across all views.
2. Levenshtein edit distances and normalized similarities.
3. Jaro-Winkler name similarities.
4. Token overlaps (Jaccard, Dice, length ratios).
5. Numeric address overlap and mismatch flags.
6. Interaction features (country match/mismatch/missing, address missingness).
7. Retrieval provenance signals (channel indicators, ranks, scores).
"""

from __future__ import annotations

import os
import multiprocessing as mp
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


# Fast Levenshtein distance
def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Wagner-Fischer Levenshtein distance with O(min(M, N)) space."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    prev_row = list(range(len(s1) + 1))
    for i, c2 in enumerate(s2):
        curr_row = [i + 1] * (len(s1) + 1)
        for j, c1 in enumerate(s1):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row[j + 1] = min(insertions, deletions, substitutions)
        prev_row = curr_row

    return prev_row[-1]


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity in [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return max(0.0, 1.0 - (dist / max_len))


def jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """Compute Jaro-Winkler string similarity in [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_bound = max(len1, len2) // 2 - 1
    if match_bound < 0:
        match_bound = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0

    for i in range(len1):
        start = max(0, i - match_bound)
        end = min(i + match_bound + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    transpositions = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        (matches / len1)
        + (matches / len2)
        + ((matches - transpositions / 2.0) / matches)
    ) / 3.0

    # Common prefix bonus up to 4 characters
    prefix_len = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * prefix_weight * (1.0 - jaro)


def token_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def token_dice(tokens1: Set[str], tokens2: Set[str]) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    total = len(tokens1) + len(tokens2)
    return (2.0 * intersection) / total if total > 0 else 0.0


_NUMERIC_PATTERN = re.compile(r"\d+")


def extract_numbers(text: str) -> Set[str]:
    return set(_NUMERIC_PATTERN.findall(text))


_worker_s1_records: Dict[str, Dict[str, str]] = {}
_worker_cand_records: Dict[str, Dict[str, str]] = {}


def _init_feature_worker(
    s1_records: Dict[str, Dict[str, str]], cand_records: Dict[str, Dict[str, str]]
) -> None:
    """Initialize worker process with entity records maps once at pool startup."""
    global _worker_s1_records, _worker_cand_records
    _worker_s1_records = s1_records
    _worker_cand_records = cand_records


def _extract_chunk_worker(chunk_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Worker function for multi-core feature extraction."""
    global _worker_s1_records, _worker_cand_records
    return PairFeatureExtractor._extract_features_list(
        chunk_df, _worker_s1_records, _worker_cand_records, verbose=False
    )


class PairFeatureExtractor:
    """Extracts leakage-free tabular feature vectors for candidate pairs."""

    @staticmethod
    def _extract_features_list(
        pairs_df: pd.DataFrame,
        s1_records: Dict[str, Dict[str, str]],
        cand_records: Dict[str, Dict[str, str]],
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        features_list = []
        total_pairs = len(pairs_df)
        recs = pairs_df.to_dict(orient="records")
        log_interval = max(100, total_pairs // 50) if total_pairs > 0 else 1

        for idx, row in enumerate(recs):
            if verbose and ((idx + 1) % log_interval == 0 or (idx + 1) == total_pairs or (idx + 1) <= 5):
                pct = (100.0 * (idx + 1) / total_pairs) if total_pairs > 0 else 100.0
                print(f"\r  [Feature Extraction] {idx + 1}/{total_pairs} pairs ({pct:.1f}%)", end="", flush=True)

            s1_id = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"]

            s1 = s1_records.get(s1_id, {})
            cand = cand_records.get(cand_id, {})

            s1_clean_name = s1.get("name_clean", "")
            cand_clean_name = cand.get("name_clean", "")

            s1_core_name = s1.get("name_core", "")
            cand_core_name = cand.get("name_core", "")

            s1_clean_addr = s1.get("address_clean", "")
            cand_clean_addr = cand.get("address_clean", "")

            s1_alias_addr = s1.get("address_alias", "")
            cand_alias_addr = cand.get("address_alias", "")

            s1_country = s1.get("country", "").strip().lower()
            cand_country = cand.get("country", "").strip().lower()

            # 1. Exact Match Indicators
            exact_clean_name = 1.0 if s1_clean_name and s1_clean_name == cand_clean_name else 0.0
            exact_core_name = 1.0 if s1_core_name and s1_core_name == cand_core_name else 0.0
            exact_clean_addr = 1.0 if s1_clean_addr and s1_clean_addr == cand_clean_addr else 0.0
            exact_alias_addr = 1.0 if s1_alias_addr and s1_alias_addr == cand_alias_addr else 0.0

            # 2. String Similarities (Levenshtein & Jaro-Winkler)
            lev_clean_name = normalized_levenshtein_similarity(s1_clean_name, cand_clean_name)
            lev_core_name = normalized_levenshtein_similarity(s1_core_name, cand_core_name)
            lev_clean_addr = (
                normalized_levenshtein_similarity(s1_clean_addr, cand_clean_addr)
                if (s1_clean_addr and cand_clean_addr)
                else 0.0
            )
            lev_alias_addr = (
                normalized_levenshtein_similarity(s1_alias_addr, cand_alias_addr)
                if (s1_alias_addr and cand_alias_addr)
                else 0.0
            )

            jw_clean_name = jaro_winkler_similarity(s1_clean_name, cand_clean_name)
            jw_core_name = jaro_winkler_similarity(s1_core_name, cand_core_name)

            # 3. Token-Level Similarities
            s1_name_tokens = set(s1_clean_name.split())
            cand_name_tokens = set(cand_clean_name.split())
            jaccard_name = token_jaccard(s1_name_tokens, cand_name_tokens)
            dice_name = token_dice(s1_name_tokens, cand_name_tokens)

            len_s1_name = len(s1_clean_name)
            len_cand_name = len(cand_clean_name)
            max_len_name = max(len_s1_name, len_cand_name)
            len_ratio_name = (min(len_s1_name, len_cand_name) / max_len_name) if max_len_name > 0 else 0.0
            token_diff_name = float(abs(len(s1_name_tokens) - len(cand_name_tokens)))

            s1_addr_tokens = set(s1_clean_addr.split())
            cand_addr_tokens = set(cand_clean_addr.split())
            jaccard_addr = token_jaccard(s1_addr_tokens, cand_addr_tokens)
            dice_addr = token_dice(s1_addr_tokens, cand_addr_tokens)

            # 4. Numeric Address Overlap
            s1_numbers = extract_numbers(s1_clean_addr)
            cand_numbers = extract_numbers(cand_clean_addr)
            has_s1_numbers = 1.0 if s1_numbers else 0.0
            has_cand_numbers = 1.0 if cand_numbers else 0.0
            numeric_overlap_count = float(len(s1_numbers & cand_numbers))
            numeric_jaccard = token_jaccard(s1_numbers, cand_numbers)
            numeric_mismatch = (
                1.0
                if (s1_numbers and cand_numbers and not (s1_numbers & cand_numbers))
                else 0.0
            )

            # 5. Country and Missingness Interactions
            country_missing = 1.0 if (not s1_country or not cand_country) else 0.0
            country_match = 1.0 if (s1_country and s1_country == cand_country) else 0.0
            country_mismatch = (
                1.0 if (s1_country and cand_country and s1_country != cand_country) else 0.0
            )

            addr_missing_s1 = 1.0 if not s1_clean_addr else 0.0
            addr_missing_cand = 1.0 if not cand_clean_addr else 0.0
            addr_both_present = 1.0 if (s1_clean_addr and cand_clean_addr) else 0.0

            # 6. Retrieval Provenance Features
            channel_count = float(row.get("channel_count", 1.0))
            channels_str = str(row.get("channels", ""))
            ch_exact_name = 1.0 if "exact_name" in channels_str else 0.0
            ch_exact_core = 1.0 if "exact_core" in channels_str else 0.0
            ch_tfidf_name = 1.0 if "tfidf_name" in channels_str else 0.0
            ch_tfidf_word = 1.0 if "tfidf_word" in channels_str else 0.0
            ch_tfidf_addr = 1.0 if "tfidf_addr" in channels_str else 0.0
            ch_rare_token = 1.0 if "rare_token" in channels_str else 0.0

            retrieval_max_score = float(row.get("max_score", 0.0))
            retrieval_min_rank = float(row.get("min_rank", 999.0))
            score_exact_name = float(row.get("score_exact_name", 0.0))
            score_exact_core = float(row.get("score_exact_core", 0.0))
            score_tfidf_name = float(row.get("score_tfidf_name", 0.0))
            score_tfidf_word = float(row.get("score_tfidf_word", 0.0))
            score_tfidf_addr = float(row.get("score_tfidf_addr", 0.0))
            score_rare_token = float(row.get("score_rare_token", 0.0))

            features_list.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                # Exact matches
                "feat_exact_clean_name": exact_clean_name,
                "feat_exact_core_name": exact_core_name,
                "feat_exact_clean_addr": exact_clean_addr,
                "feat_exact_alias_addr": exact_alias_addr,
                # String similarities
                "feat_lev_clean_name": lev_clean_name,
                "feat_lev_core_name": lev_core_name,
                "feat_lev_clean_addr": lev_clean_addr,
                "feat_lev_alias_addr": lev_alias_addr,
                "feat_jw_clean_name": jw_clean_name,
                "feat_jw_core_name": jw_core_name,
                # Token similarities
                "feat_jaccard_name": jaccard_name,
                "feat_dice_name": dice_name,
                "feat_len_ratio_name": len_ratio_name,
                "feat_token_diff_name": token_diff_name,
                "feat_jaccard_addr": jaccard_addr,
                "feat_dice_addr": dice_addr,
                # Numeric address
                "feat_has_s1_numbers": has_s1_numbers,
                "feat_has_cand_numbers": has_cand_numbers,
                "feat_numeric_overlap_count": numeric_overlap_count,
                "feat_numeric_jaccard": numeric_jaccard,
                "feat_numeric_mismatch": numeric_mismatch,
                # Interactions & missingness
                "feat_country_missing": country_missing,
                "feat_country_match": country_match,
                "feat_country_mismatch": country_mismatch,
                "feat_addr_missing_s1": addr_missing_s1,
                "feat_addr_missing_cand": addr_missing_cand,
                "feat_addr_both_present": addr_both_present,
                # Retrieval provenance
                "feat_channel_count": channel_count,
                "feat_ch_exact_name": ch_exact_name,
                "feat_ch_exact_core": ch_exact_core,
                "feat_ch_tfidf_name": ch_tfidf_name,
                "feat_ch_tfidf_word": ch_tfidf_word,
                "feat_ch_tfidf_addr": ch_tfidf_addr,
                "feat_ch_rare_token": ch_rare_token,
                "feat_retrieval_max_score": retrieval_max_score,
                "feat_retrieval_min_rank": retrieval_min_rank,
                "feat_score_exact_name": score_exact_name,
                "feat_score_exact_core": score_exact_core,
                "feat_score_tfidf_name": score_tfidf_name,
                "feat_score_tfidf_word": score_tfidf_word,
                "feat_score_tfidf_addr": score_tfidf_addr,
                "feat_score_rare_token": score_rare_token,
            })

        if verbose and total_pairs > 0:
            print(f"\r  [Feature Extraction] Completed {total_pairs}/{total_pairs} pairs (100.0%)               ")

        return features_list

    @classmethod
    def build_features(
        cls,
        pairs_df: pd.DataFrame,
        s1_records: Dict[str, Dict[str, str]],
        cand_records: Dict[str, Dict[str, str]],
        verbose: bool = True,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Build pair features for each row in pairs_df with multi-core support.

        Parameters
        ----------
        pairs_df : pd.DataFrame with candidate pairs.
        s1_records : Dict[entity_id, normalized_field_dict].
        cand_records : Dict[entity_id, normalized_field_dict].
        verbose : bool indicating whether to print progress.
        n_jobs : number of parallel worker processes (-1 for all available cores).

        Returns
        -------
        pd.DataFrame containing feature columns plus identifiers.
        """
        total_pairs = len(pairs_df)
        if total_pairs == 0:
            return pd.DataFrame()

        n_workers = os.cpu_count() or 4 if n_jobs == -1 else n_jobs
        n_workers = max(1, min(n_workers, 8))

        if total_pairs < 50 or n_workers <= 1:
            features_list = cls._extract_features_list(
                pairs_df, s1_records, cand_records, verbose=verbose
            )
        else:
            chunk_size = (total_pairs + n_workers - 1) // n_workers
            chunks = [
                pairs_df.iloc[i : i + chunk_size]
                for i in range(0, total_pairs, chunk_size)
            ]
            if verbose:
                print(f"  [Feature Extraction (Multi-Core: {n_workers} CPU cores)] Extracting features for {total_pairs} pairs across {len(chunks)} parallel chunks...")

            ctx = mp.get_context("fork")
            
            # Set globals in parent so children inherit them via Copy-On-Write (COW) memory
            global _worker_s1_records, _worker_cand_records
            _worker_s1_records = s1_records
            _worker_cand_records = cand_records

            with ctx.Pool(processes=n_workers) as pool:
                results_nested = pool.map(_extract_chunk_worker, chunks)

            features_list = [item for sublist in results_nested for item in sublist]
            if verbose:
                print(f"\r  [Feature Extraction (Multi-Core)] Completed {total_pairs}/{total_pairs} pairs (100.0%) across {n_workers} CPU cores.    ")

        df = pd.DataFrame(features_list)
        # Strict NaN and Infinity assertions
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        if not df.empty:
            assert not df[feat_cols].isna().any().any(), "Features contain NaN values!"
            assert not np.isinf(df[feat_cols].to_numpy()).any(), "Features contain Infinite values!"

        return df

    @staticmethod
    def get_feature_column_names() -> List[str]:
        """Return canonical list of feature column names."""
        dummy_df = pd.DataFrame([{
            "source1_entity_id": "S1-1",
            "candidate_entity_id": "S2-1",
        }])
        dummy_res = PairFeatureExtractor.build_features(dummy_df, {}, {})
        return [c for c in dummy_res.columns if c.startswith("feat_")]
