"""Evaluation metrics, retrieval benchmarking, and error taxonomy for business entity resolution.

Implements:
1. Exact competition Macro F_0.5 metric (including proper singleton scoring).
2. Oracle Macro F_0.5 evaluation (retrieval upper bound).
3. Micro candidate recall, entity complete recall, any-hit recall.
4. Channel diversity matrix & Unique GT Recovered per channel.
5. Explicit Retrieval Miss Rate and Ranking Failure Rate.
6. Error taxonomy analysis (token reordering, semantic/linguistic, typos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


@dataclass
class ChannelDiversityStats:
    channel_name: str
    total_retrieved: int
    gt_recovered: int
    unique_gt_recovered: int  # GT matches recovered ONLY by this channel


@dataclass
class RetrievalBenchmarkReport:
    total_s1_entities: int
    total_gt_pairs: int
    retrieved_gt_pairs: int
    micro_recall: float
    entity_any_hit_recall: float
    entity_complete_recall: float
    oracle_macro_f05: float
    retrieval_miss_rate: float
    ranking_failure_rate: float
    channel_stats: List[ChannelDiversityStats]
    error_taxonomy_counts: Dict[str, int]


class MetricsEvaluator:
    """Computes exact competition metrics and diagnostic error benchmarks."""

    @staticmethod
    def calculate_single_entity_f05(
        predicted: Set[str],
        truth: Set[str],
    ) -> float:
        """Calculate F_0.5 score for a single Source 1 entity.

        Formula:
        F_0.5 = (1.25 * P * R) / (0.25 * P + R)
        Singletons:
          truth is empty:
            pred is empty -> 1.0
            pred is non-empty -> 0.0
        Non-singletons:
          TP == 0 -> 0.0
        """
        # Case 1: Singleton (no true matches)
        if len(truth) == 0:
            return 1.0 if len(predicted) == 0 else 0.0

        # Case 2: Non-singleton with empty prediction
        if len(predicted) == 0:
            return 0.0

        tp = len(predicted & truth)
        if tp == 0:
            return 0.0

        precision = tp / len(predicted)
        recall = tp / len(truth)

        denom = 0.25 * precision + recall
        if denom == 0.0:
            return 0.0

        return (1.25 * precision * recall) / denom

    @classmethod
    def compute_macro_f05(
        cls,
        predictions: Dict[str, Set[str]],
        ground_truth: Dict[str, Set[str]],
        all_s1_ids: Set[str],
    ) -> float:
        """Compute exact macro-averaged F_0.5 across ALL Source 1 entities."""
        if not all_s1_ids:
            return 0.0

        scores = [
            cls.calculate_single_entity_f05(
                predicted=predictions.get(s1_id, set()),
                truth=ground_truth.get(s1_id, set()),
            )
            for s1_id in all_s1_ids
        ]
        return float(np.mean(scores))

    @classmethod
    def compute_oracle_f05(
        cls,
        candidates: Dict[str, Set[str]],
        ground_truth: Dict[str, Set[str]],
        all_s1_ids: Set[str],
    ) -> float:
        """Compute Oracle Macro F_0.5 upper-bound from candidate generation.

        Assumes a perfect ranker/classifier that selects exactly (candidates ∩ truth)
        for every entity. Singletons predict empty set (score 1.0).
        For non-singletons, precision is 1.0, recall = TP / |truth|.
        """
        oracle_preds: Dict[str, Set[str]] = {}
        for s1_id in all_s1_ids:
            gt_set = ground_truth.get(s1_id, set())
            cand_set = candidates.get(s1_id, set())
            oracle_preds[s1_id] = cand_set & gt_set

        return cls.compute_macro_f05(oracle_preds, ground_truth, all_s1_ids)

    @classmethod
    def benchmark_retrieval(
        cls,
        retrieval_results: Dict[str, Dict[str, Any]],  # s1_id -> target_id -> CandidateProvenance
        ground_truth: Dict[str, Set[str]],
        all_s1_ids: Set[str],
        s1_names: Optional[Dict[str, str]] = None,
        target_names: Optional[Dict[str, str]] = None,
        ranking_cutoff_rank: int = 10,
    ) -> RetrievalBenchmarkReport:
        """Run comprehensive retrieval benchmarking and bottleneck analysis."""
        total_s1 = len(all_s1_ids)
        all_gt_pairs: Set[Tuple[str, str]] = {
            (s1, target) for s1, targets in ground_truth.items() for target in targets
        }
        total_gt = len(all_gt_pairs)

        retrieved_gt_pairs: Set[Tuple[str, str]] = set()
        channel_retrieved_gt: Dict[str, Set[Tuple[str, str]]] = {}
        channel_total_retrieved: Dict[str, int] = {}

        candidate_map: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}
        ranked_below_cutoff_gt: Set[Tuple[str, str]] = set()

        for s1_id, target_map in retrieval_results.items():
            for target_id, prov in target_map.items():
                candidate_map[s1_id].add(target_id)
                pair = (s1_id, target_id)
                is_gt = pair in all_gt_pairs
                if is_gt:
                    retrieved_gt_pairs.add(pair)
                    # Check ranking cutoff
                    if prov.min_rank > ranking_cutoff_rank:
                        ranked_below_cutoff_gt.add(pair)

                for ch in prov.channels:
                    channel_total_retrieved[ch] = channel_total_retrieved.get(ch, 0) + 1
                    if is_gt:
                        if ch not in channel_retrieved_gt:
                            channel_retrieved_gt[ch] = set()
                        channel_retrieved_gt[ch].add(pair)

        # Compute Unique GT Recovered per channel
        channel_stats: List[ChannelDiversityStats] = []
        for ch, gt_set in channel_retrieved_gt.items():
            other_channels_gt = set()
            for other_ch, other_gt_set in channel_retrieved_gt.items():
                if other_ch != ch:
                    other_channels_gt |= other_gt_set
            unique_gt = len(gt_set - other_channels_gt)
            channel_stats.append(
                ChannelDiversityStats(
                    channel_name=ch,
                    total_retrieved=channel_total_retrieved.get(ch, 0),
                    gt_recovered=len(gt_set),
                    unique_gt_recovered=unique_gt,
                )
            )

        # Micro recall & miss rate
        micro_recall = len(retrieved_gt_pairs) / total_gt if total_gt > 0 else 1.0
        retrieval_miss_rate = (total_gt - len(retrieved_gt_pairs)) / total_gt if total_gt > 0 else 0.0

        # Ranking failure rate
        ranking_failure_rate = (
            len(ranked_below_cutoff_gt) / len(retrieved_gt_pairs)
            if len(retrieved_gt_pairs) > 0
            else 0.0
        )

        # Entity-level recalls
        non_singleton_s1 = [s1 for s1 in all_s1_ids if len(ground_truth.get(s1, set())) > 0]
        any_hit_count = 0
        complete_hit_count = 0

        for s1 in non_singleton_s1:
            gt_set = ground_truth[s1]
            cands = candidate_map.get(s1, set())
            hits = len(gt_set & cands)
            if hits > 0:
                any_hit_count += 1
            if hits == len(gt_set):
                complete_hit_count += 1

        entity_any_hit_recall = any_hit_count / len(non_singleton_s1) if non_singleton_s1 else 1.0
        entity_complete_recall = (
            complete_hit_count / len(non_singleton_s1) if non_singleton_s1 else 1.0
        )

        # Oracle Macro F_0.5
        oracle_macro_f05 = cls.compute_oracle_f05(candidate_map, ground_truth, all_s1_ids)

        # Qualitative Error Taxonomy on Missed GT Pairs
        missed_gt_pairs = all_gt_pairs - retrieved_gt_pairs
        error_taxonomy_counts = cls._classify_missed_gt(
            missed_gt_pairs, s1_names=s1_names, target_names=target_names
        )

        return RetrievalBenchmarkReport(
            total_s1_entities=total_s1,
            total_gt_pairs=total_gt,
            retrieved_gt_pairs=len(retrieved_gt_pairs),
            micro_recall=micro_recall,
            entity_any_hit_recall=entity_any_hit_recall,
            entity_complete_recall=entity_complete_recall,
            oracle_macro_f05=oracle_macro_f05,
            retrieval_miss_rate=retrieval_miss_rate,
            ranking_failure_rate=ranking_failure_rate,
            channel_stats=channel_stats,
            error_taxonomy_counts=error_taxonomy_counts,
        )

    @staticmethod
    def _classify_missed_gt(
        missed_pairs: Set[Tuple[str, str]],
        s1_names: Optional[Dict[str, str]] = None,
        target_names: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        """Classify missed ground truth links into error taxonomy categories."""
        counts = {
            "token_reordering": 0,
            "semantic_linguistic": 0,
            "severe_typo": 0,
            "generic_lexical_miss": 0,
        }

        if not s1_names or not target_names:
            counts["generic_lexical_miss"] = len(missed_pairs)
            return counts

        for s1, target in missed_pairs:
            name1 = s1_names.get(s1, "").lower()
            name2 = target_names.get(target, "").lower()

            if not name1 or not name2:
                counts["generic_lexical_miss"] += 1
                continue

            tokens1 = set(name1.split())
            tokens2 = set(name2.split())
            overlap = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            jaccard = overlap / union if union > 0 else 0.0

            if jaccard >= 0.5:
                # Same or shared tokens, but in different order or extra tokens
                counts["token_reordering"] += 1
            elif jaccard == 0:
                counts["semantic_linguistic"] += 1
            else:
                counts["severe_typo"] += 1

        return counts
