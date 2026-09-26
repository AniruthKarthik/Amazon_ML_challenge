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
    """Compute Wagner-Fischer Levenshtein distance with O(min(M, N)) space and minimal allocations."""
    if s1 == s2:
        return 0
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m

    if m > n:
        s1, s2 = s2, s1
        m, n = n, m

    v0 = list(range(m + 1))
    v1 = [0] * (m + 1)
    for c2 in s2:
        v1[0] = v0[0] + 1
        for j, c1 in enumerate(s1):
            cost = 0 if c1 == c2 else 1
            del_cost = v0[j + 1] + 1
            ins_cost = v1[j] + 1
            sub_cost = v0[j] + cost
            v1[j + 1] = del_cost if del_cost < ins_cost and del_cost < sub_cost else (ins_cost if ins_cost < sub_cost else sub_cost)
        v0, v1 = v1, v0

    return v0[m]


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


def extract_postal_code(text: str, country: str = "") -> str:
    """Extract country-specific postal / zip / pin code."""
    if not text:
        return ""
    c = country.strip().lower()
    if c in ["in", "india"]:
        m = re.search(r"\b[1-9]\d{5}\b", text)
        return m.group(0) if m else ""
    elif c in ["fr", "france"]:
        m = re.search(r"\b(?:0[1-9]|[1-8]\d|9[0-8])\d{3}\b", text)
        return m.group(0) if m else ""
    elif c in ["us", "united states"]:
        m = re.search(r"\b\d{5}(?:-\d{4})?\b", text)
        return m.group(0)[:5] if m else ""
    else:
        m6 = re.search(r"\b[1-9]\d{5}\b", text)
        if m6:
            return m6.group(0)
        m5 = re.search(r"\b\d{5}\b", text)
        return m5.group(0) if m5 else ""


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
    """Worker function for backward-compatible feature extraction."""
    global _worker_s1_records, _worker_cand_records
    return PairFeatureExtractor._extract_features_list(
        chunk_df, _worker_s1_records, _worker_cand_records, verbose=False
    )


class PairFeatureExtractor:
    """Extracts leakage-free tabular feature vectors for candidate pairs with memory-safe preallocated arrays."""

    @classmethod
    def _extract_features_fast(
        cls,
        pairs_df: pd.DataFrame,
        s1_records: Dict[str, Dict[str, str]],
        cand_records: Dict[str, Dict[str, str]],
        verbose: bool = False,
    ) -> pd.DataFrame:
        total = len(pairs_df)
        if total == 0:
            return pd.DataFrame()

        s1_ids = pairs_df["source1_entity_id"].tolist()
        cand_ids = pairs_df["candidate_entity_id"].tolist()

        f_exact_clean_name = np.empty(total, dtype=np.float32)
        f_exact_core_name = np.empty(total, dtype=np.float32)
        f_exact_clean_addr = np.empty(total, dtype=np.float32)
        f_exact_alias_addr = np.empty(total, dtype=np.float32)
        f_exact_name_and_addr = np.empty(total, dtype=np.float32)

        f_lev_clean_name = np.empty(total, dtype=np.float32)
        f_lev_core_name = np.empty(total, dtype=np.float32)
        f_lev_clean_addr = np.empty(total, dtype=np.float32)
        f_lev_alias_addr = np.empty(total, dtype=np.float32)

        f_jw_clean_name = np.empty(total, dtype=np.float32)
        f_jw_core_name = np.empty(total, dtype=np.float32)
        f_jw_clean_addr = np.empty(total, dtype=np.float32)

        f_jaccard_name = np.empty(total, dtype=np.float32)
        f_dice_name = np.empty(total, dtype=np.float32)
        f_len_ratio_name = np.empty(total, dtype=np.float32)
        f_token_diff_name = np.empty(total, dtype=np.float32)
        f_token_overlap_count = np.empty(total, dtype=np.float32)

        f_name_contain_s1 = np.empty(total, dtype=np.float32)
        f_name_contain_cand = np.empty(total, dtype=np.float32)
        f_token_subset_s1 = np.empty(total, dtype=np.float32)
        f_token_subset_cand = np.empty(total, dtype=np.float32)
        f_prefix_match_4 = np.empty(total, dtype=np.float32)

        f_jaccard_addr = np.empty(total, dtype=np.float32)
        f_dice_addr = np.empty(total, dtype=np.float32)
        f_addr_len_ratio = np.empty(total, dtype=np.float32)

        f_has_s1_numbers = np.empty(total, dtype=np.float32)
        f_has_cand_numbers = np.empty(total, dtype=np.float32)
        f_numeric_overlap_count = np.empty(total, dtype=np.float32)
        f_numeric_jaccard = np.empty(total, dtype=np.float32)
        f_numeric_mismatch = np.empty(total, dtype=np.float32)
        f_first_number_match = np.empty(total, dtype=np.float32)

        f_postal_code_match = np.empty(total, dtype=np.float32)
        f_postal_code_mismatch = np.empty(total, dtype=np.float32)

        f_country_missing = np.empty(total, dtype=np.float32)
        f_country_match = np.empty(total, dtype=np.float32)
        f_country_mismatch = np.empty(total, dtype=np.float32)

        f_addr_missing_s1 = np.empty(total, dtype=np.float32)
        f_addr_missing_cand = np.empty(total, dtype=np.float32)
        f_addr_both_present = np.empty(total, dtype=np.float32)

        log_interval = max(500, total // 20) if total > 0 else 1

        for i in range(total):
            if verbose and ((i + 1) % log_interval == 0 or (i + 1) == total or (i + 1) <= 5):
                pct = 100.0 * (i + 1) / total
                print(f"\r  [Feature Extraction] {i + 1}/{total} pairs ({pct:.1f}%)", end="", flush=True)

            s1 = s1_records.get(s1_ids[i], {})
            cand = cand_records.get(cand_ids[i], {})

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

            # 1. Exact match indicators
            exact_cn = 1.0 if s1_clean_name and s1_clean_name == cand_clean_name else 0.0
            exact_cr = 1.0 if s1_core_name and s1_core_name == cand_core_name else 0.0
            exact_ca = 1.0 if s1_clean_addr and s1_clean_addr == cand_clean_addr else 0.0
            exact_aa = 1.0 if s1_alias_addr and s1_alias_addr == cand_alias_addr else 0.0
            f_exact_clean_name[i] = exact_cn
            f_exact_core_name[i] = exact_cr
            f_exact_clean_addr[i] = exact_ca
            f_exact_alias_addr[i] = exact_aa
            f_exact_name_and_addr[i] = exact_cn * exact_ca

            # 2. String similarities & token overlaps (with exact match short-circuit)
            s1_nt = set(s1_clean_name.split())
            c_nt = set(cand_clean_name.split())
            f_token_overlap_count[i] = float(len(s1_nt & c_nt))

            if exact_cn == 1.0:
                f_lev_clean_name[i] = 1.0
                f_jw_clean_name[i] = 1.0
                f_jaccard_name[i] = 1.0
                f_dice_name[i] = 1.0
                f_len_ratio_name[i] = 1.0
                f_token_diff_name[i] = 0.0
                f_name_contain_s1[i] = 1.0 if s1_clean_name else 0.0
                f_name_contain_cand[i] = 1.0 if cand_clean_name else 0.0
                f_token_subset_s1[i] = 1.0 if s1_nt else 0.0
                f_token_subset_cand[i] = 1.0 if c_nt else 0.0
                f_prefix_match_4[i] = 1.0 if len(s1_clean_name) >= 4 else 0.0
            else:
                f_lev_clean_name[i] = normalized_levenshtein_similarity(s1_clean_name, cand_clean_name)
                f_jw_clean_name[i] = jaro_winkler_similarity(s1_clean_name, cand_clean_name)
                f_jaccard_name[i] = token_jaccard(s1_nt, c_nt)
                f_dice_name[i] = token_dice(s1_nt, c_nt)
                l1, l2 = len(s1_clean_name), len(cand_clean_name)
                max_l = max(l1, l2)
                f_len_ratio_name[i] = (min(l1, l2) / max_l) if max_l > 0 else 0.0
                f_token_diff_name[i] = float(abs(len(s1_nt) - len(c_nt)))
                f_name_contain_s1[i] = 1.0 if s1_clean_name and s1_clean_name in cand_clean_name else 0.0
                f_name_contain_cand[i] = 1.0 if cand_clean_name and cand_clean_name in s1_clean_name else 0.0
                f_token_subset_s1[i] = 1.0 if s1_nt and s1_nt.issubset(c_nt) else 0.0
                f_token_subset_cand[i] = 1.0 if c_nt and c_nt.issubset(s1_nt) else 0.0
                f_prefix_match_4[i] = 1.0 if s1_clean_name[:4] == cand_clean_name[:4] and len(s1_clean_name) >= 4 else 0.0

            if exact_cr == 1.0:
                f_lev_core_name[i] = 1.0
                f_jw_core_name[i] = 1.0
            else:
                f_lev_core_name[i] = normalized_levenshtein_similarity(s1_core_name, cand_core_name)
                f_jw_core_name[i] = jaro_winkler_similarity(s1_core_name, cand_core_name)

            if exact_ca == 1.0:
                f_lev_clean_addr[i] = 1.0
                f_jw_clean_addr[i] = 1.0
                f_jaccard_addr[i] = 1.0
                f_dice_addr[i] = 1.0
                f_addr_len_ratio[i] = 1.0 if s1_clean_addr else 0.0
            else:
                f_lev_clean_addr[i] = normalized_levenshtein_similarity(s1_clean_addr, cand_clean_addr) if (s1_clean_addr and cand_clean_addr) else 0.0
                f_jw_clean_addr[i] = jaro_winkler_similarity(s1_clean_addr, cand_clean_addr) if (s1_clean_addr and cand_clean_addr) else 0.0
                s1_at = set(s1_clean_addr.split())
                c_at = set(cand_clean_addr.split())
                f_jaccard_addr[i] = token_jaccard(s1_at, c_at)
                f_dice_addr[i] = token_dice(s1_at, c_at)
                al1, al2 = len(s1_clean_addr), len(cand_clean_addr)
                max_al = max(al1, al2)
                f_addr_len_ratio[i] = (min(al1, al2) / max_al) if max_al > 0 else 0.0

            f_lev_alias_addr[i] = 1.0 if exact_aa == 1.0 else (normalized_levenshtein_similarity(s1_alias_addr, cand_alias_addr) if (s1_alias_addr and cand_alias_addr) else 0.0)

            # 4. Numeric Address & Postal Code Overlap
            s1_num_list = _NUMERIC_PATTERN.findall(s1_clean_addr)
            c_num_list = _NUMERIC_PATTERN.findall(cand_clean_addr)
            s1_num = set(s1_num_list)
            c_num = set(c_num_list)
            f_has_s1_numbers[i] = 1.0 if s1_num else 0.0
            f_has_cand_numbers[i] = 1.0 if c_num else 0.0
            f_numeric_overlap_count[i] = float(len(s1_num & c_num))
            f_numeric_jaccard[i] = token_jaccard(s1_num, c_num)
            f_numeric_mismatch[i] = 1.0 if (s1_num and c_num and not (s1_num & c_num)) else 0.0
            f_first_number_match[i] = 1.0 if s1_num_list and c_num_list and s1_num_list[0] == c_num_list[0] else 0.0

            p1 = extract_postal_code(s1_clean_addr, s1_country)
            p2 = extract_postal_code(cand_clean_addr, cand_country)
            f_postal_code_match[i] = 1.0 if p1 and p2 and p1 == p2 else 0.0
            f_postal_code_mismatch[i] = 1.0 if p1 and p2 and p1 != p2 else 0.0

            # 5. Country and missingness interactions
            f_country_missing[i] = 1.0 if (not s1_country or not cand_country) else 0.0
            f_country_match[i] = 1.0 if (s1_country and s1_country == cand_country) else 0.0
            f_country_mismatch[i] = 1.0 if (s1_country and cand_country and s1_country != cand_country) else 0.0
            f_addr_missing_s1[i] = 1.0 if not s1_clean_addr else 0.0
            f_addr_missing_cand[i] = 1.0 if not cand_clean_addr else 0.0
            f_addr_both_present[i] = 1.0 if (s1_clean_addr and cand_clean_addr) else 0.0

        # Vectorized retrieval provenance extraction directly from pairs_df (zero-copy)
        channels_s = pairs_df["channels"].fillna("").astype(str) if "channels" in pairs_df.columns else pd.Series([""] * total)
        score_name_arr = pairs_df["score_tfidf_name"].to_numpy(dtype=np.float32) if "score_tfidf_name" in pairs_df.columns else np.zeros(total, dtype=np.float32)
        score_addr_arr = pairs_df["score_tfidf_addr"].to_numpy(dtype=np.float32) if "score_tfidf_addr" in pairs_df.columns else np.zeros(total, dtype=np.float32)
        score_word_arr = pairs_df["score_tfidf_word"].to_numpy(dtype=np.float32) if "score_tfidf_word" in pairs_df.columns else np.zeros(total, dtype=np.float32)
        score_exact_n_arr = pairs_df["score_exact_name"].to_numpy(dtype=np.float32) if "score_exact_name" in pairs_df.columns else np.zeros(total, dtype=np.float32)
        score_exact_c_arr = pairs_df["score_exact_core"].to_numpy(dtype=np.float32) if "score_exact_core" in pairs_df.columns else np.zeros(total, dtype=np.float32)

        feat_score_name_x_addr = score_name_arr * score_addr_arr
        feat_max_retrieval_score = np.maximum.reduce([score_exact_n_arr, score_exact_c_arr, score_name_arr, score_word_arr, score_addr_arr])

        out_dict = {
            "source1_entity_id": s1_ids,
            "candidate_entity_id": cand_ids,
            # Exact match features
            "feat_exact_clean_name": f_exact_clean_name,
            "feat_exact_core_name": f_exact_core_name,
            "feat_exact_clean_addr": f_exact_clean_addr,
            "feat_exact_alias_addr": f_exact_alias_addr,
            "feat_exact_name_and_addr": f_exact_name_and_addr,
            # Levenshtein string similarities
            "feat_lev_clean_name": f_lev_clean_name,
            "feat_lev_core_name": f_lev_core_name,
            "feat_lev_clean_addr": f_lev_clean_addr,
            "feat_lev_alias_addr": f_lev_alias_addr,
            # Jaro-Winkler similarities (Name & Address)
            "feat_jw_clean_name": f_jw_clean_name,
            "feat_jw_core_name": f_jw_core_name,
            "feat_jw_clean_addr": f_jw_clean_addr,
            # Token-level similarities
            "feat_jaccard_name": f_jaccard_name,
            "feat_dice_name": f_dice_name,
            "feat_len_ratio_name": f_len_ratio_name,
            "feat_token_diff_name": f_token_diff_name,
            "feat_token_overlap_count": f_token_overlap_count,
            "feat_name_contain_s1": f_name_contain_s1,
            "feat_name_contain_cand": f_name_contain_cand,
            "feat_token_subset_s1": f_token_subset_s1,
            "feat_token_subset_cand": f_token_subset_cand,
            "feat_prefix_match_4": f_prefix_match_4,
            # Address token similarities & length ratios
            "feat_jaccard_addr": f_jaccard_addr,
            "feat_dice_addr": f_dice_addr,
            "feat_addr_len_ratio": f_addr_len_ratio,
            # Numeric & Postal code address features
            "feat_has_s1_numbers": f_has_s1_numbers,
            "feat_has_cand_numbers": f_has_cand_numbers,
            "feat_numeric_overlap_count": f_numeric_overlap_count,
            "feat_numeric_jaccard": f_numeric_jaccard,
            "feat_numeric_mismatch": f_numeric_mismatch,
            "feat_first_number_match": f_first_number_match,
            "feat_postal_code_match": f_postal_code_match,
            "feat_postal_code_mismatch": f_postal_code_mismatch,
            # Country and missingness interactions
            "feat_country_missing": f_country_missing,
            "feat_country_match": f_country_match,
            "feat_country_mismatch": f_country_mismatch,
            "feat_addr_missing_s1": f_addr_missing_s1,
            "feat_addr_missing_cand": f_addr_missing_cand,
            "feat_addr_both_present": f_addr_both_present,
            # Retrieval provenance & cross-scores
            "feat_channel_count": pairs_df["channel_count"].to_numpy(dtype=np.float32) if "channel_count" in pairs_df.columns else np.ones(total, dtype=np.float32),
            "feat_ch_exact_name": channels_s.str.contains("exact_name", regex=False).to_numpy(dtype=np.float32),
            "feat_ch_exact_core": channels_s.str.contains("exact_core", regex=False).to_numpy(dtype=np.float32),
            "feat_ch_tfidf_name": channels_s.str.contains("tfidf_name", regex=False).to_numpy(dtype=np.float32),
            "feat_ch_tfidf_word": channels_s.str.contains("tfidf_word", regex=False).to_numpy(dtype=np.float32),
            "feat_ch_tfidf_addr": channels_s.str.contains("tfidf_addr", regex=False).to_numpy(dtype=np.float32),
            "feat_ch_rare_token": channels_s.str.contains("rare_token", regex=False).to_numpy(dtype=np.float32),
            "feat_retrieval_max_score": pairs_df["max_score"].to_numpy(dtype=np.float32) if "max_score" in pairs_df.columns else np.zeros(total, dtype=np.float32),
            "feat_retrieval_min_rank": pairs_df["min_rank"].to_numpy(dtype=np.float32) if "min_rank" in pairs_df.columns else np.full(total, 999.0, dtype=np.float32),
            "feat_score_exact_name": score_exact_n_arr,
            "feat_score_exact_core": score_exact_c_arr,
            "feat_score_tfidf_name": score_name_arr,
            "feat_score_tfidf_word": score_word_arr,
            "feat_score_tfidf_addr": score_addr_arr,
            "feat_score_rare_token": pairs_df["score_rare_token"].to_numpy(dtype=np.float32) if "score_rare_token" in pairs_df.columns else np.zeros(total, dtype=np.float32),
            "feat_score_name_x_addr": feat_score_name_x_addr,
            "feat_max_retrieval_score": feat_max_retrieval_score,
        }
        return pd.DataFrame(out_dict)

    @classmethod
    def _extract_features_list(
        cls,
        pairs_df: pd.DataFrame,
        s1_records: Dict[str, Dict[str, str]],
        cand_records: Dict[str, Dict[str, str]],
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        df = cls._extract_features_fast(pairs_df, s1_records, cand_records, verbose=verbose)
        return df.to_dict(orient="records")

    @classmethod
    def build_features(
        cls,
        pairs_df: pd.DataFrame,
        s1_records: Dict[str, Dict[str, str]],
        cand_records: Dict[str, Dict[str, str]],
        verbose: bool = True,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Build pair features for each row in pairs_df with memory-safe preallocated arrays.

        Parameters
        ----------
        pairs_df : pd.DataFrame with candidate pairs.
        s1_records : Dict[entity_id, normalized_field_dict].
        cand_records : Dict[entity_id, normalized_field_dict].
        verbose : bool indicating whether to print progress.
        n_jobs : number of parallel worker processes (kept for API compatibility).

        Returns
        -------
        pd.DataFrame containing feature columns plus identifiers.
        """
        total_pairs = len(pairs_df)
        if total_pairs == 0:
            return pd.DataFrame()

        if verbose:
            print(f"  [Feature Extraction] Extracting features for {total_pairs} pairs...")

        df = cls._extract_features_fast(pairs_df, s1_records, cand_records, verbose=verbose)

        # Strict NaN and Infinity assertions
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        if not df.empty:
            assert not df[feat_cols].isna().any().any(), "Features contain NaN values!"
            assert not np.isinf(df[feat_cols].to_numpy()).any(), "Features contain Infinite values!"

        if verbose and total_pairs > 0:
            print(f"\r  [Feature Extraction] Completed {total_pairs}/{total_pairs} pairs (100.0%).")

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
