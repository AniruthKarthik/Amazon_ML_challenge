"""Entity-level aggregation, set decision logic, and robust threshold optimization.

Implements:
1. Entity-level pair score aggregation & meta-feature extraction.
2. Parametric entity set decision policies (pair threshold, entity threshold, score gap).
3. Exact competition macro F_0.5 evaluation on OOF predictions.
4. Joint threshold optimization over (pair_thresh, entity_thresh, gap_thresh).
5. Threshold sensitivity analysis (plateau testing, fold dispersion).
6. Simulated country/domain shift robustness evaluation (Policy A, B, C).
7. Official matching_results.tsv generation.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from chimera_submission.code.business_entity_resolution.src.evaluation import MetricsEvaluator


@dataclass
class EntityDecisionPolicy:
    pair_threshold: float = 0.50
    entity_threshold: float = 0.55
    gap_threshold: float = 0.15
    max_matches: int = 10
    policy_name: str = "global_robust"


@dataclass
class ThresholdOptimizationReport:
    best_policy: EntityDecisionPolicy
    best_macro_f05: float
    raw_vs_calibrated_comparison: Dict[str, float]
    stability_plateau_f05: Dict[str, float]  # scores at theta +- delta
    fold_f05_scores: Dict[int, float]
    country_f05_scores: Dict[str, float]
    policy_comparison: Dict[str, Dict[str, float]]  # Policy A vs B vs C metrics


class EntityAggregator:
    """Aggregates candidate pair predictions to entity-level decisions."""

    @staticmethod
    def aggregate_entity_candidates(
        scored_pairs_df: pd.DataFrame,
        all_s1_ids: Set[str],
        score_col: str = "raw_score",
    ) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, float]]]]:
        """Group scored pairs by S1 entity and compute entity meta-features.

        Returns
        -------
        Tuple of:
          1. pd.DataFrame of entity meta-features (one row per S1 entity).
          2. Dict[s1_id, List[(cand_id, score)]] sorted descending by score.
        """
        # Map s1_id -> list of (cand_id, score)
        candidates_by_s1: Dict[str, List[Tuple[str, float]]] = {
            s1_id: [] for s1_id in all_s1_ids
        }

        for _, row in scored_pairs_df.iterrows():
            s1 = row["source1_entity_id"]
            cand = row["candidate_entity_id"]
            score = float(row[score_col])
            if s1 in candidates_by_s1:
                candidates_by_s1[s1].append((cand, score))

        # Sort candidate lists descending by score
        for s1_id in candidates_by_s1:
            candidates_by_s1[s1_id].sort(key=lambda x: x[1], reverse=True)

        meta_rows = []
        for s1_id in sorted(all_s1_ids):
            cands = candidates_by_s1[s1_id]
            scores = [s[1] for s in cands]
            n_cands = len(scores)

            max_score = scores[0] if n_cands > 0 else 0.0
            second_max = scores[1] if n_cands > 1 else 0.0
            gap = max_score - second_max
            mean_score = float(np.mean(scores)) if n_cands > 0 else 0.0
            std_score = float(np.std(scores)) if n_cands > 0 else 0.0

            n_above_05 = sum(1 for s in scores if s >= 0.5)
            n_above_07 = sum(1 for s in scores if s >= 0.7)
            n_above_09 = sum(1 for s in scores if s >= 0.9)

            meta_rows.append({
                "source1_entity_id": s1_id,
                "meta_candidate_count": float(n_cands),
                "meta_max_score": max_score,
                "meta_second_max_score": second_max,
                "meta_score_gap": gap,
                "meta_mean_score": mean_score,
                "meta_std_score": std_score,
                "meta_n_above_05": float(n_above_05),
                "meta_n_above_07": float(n_above_07),
                "meta_n_above_09": float(n_above_09),
            })

        meta_df = pd.DataFrame(meta_rows)
        return meta_df, candidates_by_s1

    @staticmethod
    def apply_policy(
        candidates_by_s1: Dict[str, List[Tuple[str, float]]],
        policy: EntityDecisionPolicy,
        all_s1_ids: Set[str],
        country_policies: Optional[Dict[str, EntityDecisionPolicy]] = None,
        s1_country_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Set[str]]:
        """Apply decision thresholds to select final matching sets for each S1 entity."""
        predictions: Dict[str, Set[str]] = {}

        for s1_id in all_s1_ids:
            # Check country-specific policy override if applicable
            active_policy = policy
            if country_policies and s1_country_map:
                country = s1_country_map.get(s1_id, "")
                if country in country_policies:
                    active_policy = country_policies[country]

            cands = candidates_by_s1.get(s1_id, [])
            if not cands:
                predictions[s1_id] = set()
                continue

            max_cand, max_score = cands[0]

            # Entity-level gate: top score must clear entity threshold
            if max_score < active_policy.entity_threshold:
                predictions[s1_id] = set()
                continue

            # Candidate selection
            selected: Set[str] = set()
            for cand_id, score in cands:
                if len(selected) >= active_policy.max_matches:
                    break
                # Pair threshold check
                if score < active_policy.pair_threshold:
                    continue
                # Gap threshold check relative to top match
                if (max_score - score) > active_policy.gap_threshold:
                    continue
                selected.add(cand_id)

            predictions[s1_id] = selected

        return predictions

    @staticmethod
    def to_matching_results_tsv(
        predictions: Dict[str, Set[str]],
        all_s1_ids: Set[str],
    ) -> pd.DataFrame:
        """Export predictions to official matching_results.tsv format."""
        rows = []
        for s1_id in sorted(all_s1_ids):
            target_ids = sorted(list(predictions.get(s1_id, set())))
            rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(target_ids) if target_ids else "",
            })
        return pd.DataFrame(rows)


def _evaluate_combo_worker(
    args: Tuple[Tuple[float, float, float], Dict[str, List[Tuple[str, float]]], Dict[str, Set[str]], Set[str]]
) -> Tuple[EntityDecisionPolicy, float]:
    """Worker function for multi-core threshold combination evaluation."""
    (p_th, e_th, g_th), cands_by_s1, ground_truth, all_s1_ids = args
    pol = EntityDecisionPolicy(
        pair_threshold=p_th,
        entity_threshold=e_th,
        gap_threshold=g_th,
    )
    preds = EntityAggregator.apply_policy(cands_by_s1, pol, all_s1_ids)
    score = MetricsEvaluator.compute_macro_f05(preds, ground_truth, all_s1_ids)
    return pol, score


class ThresholdOptimizer:
    """Jointly optimizes and stress-tests decision threshold policies."""

    @classmethod
    def optimize(
        cls,
        scored_pairs_df: pd.DataFrame,
        ground_truth: Dict[str, Set[str]],
        all_s1_ids: Set[str],
        fold_map: Dict[str, int],
        s1_country_map: Optional[Dict[str, str]] = None,
        pair_threshold_grid: Optional[List[float]] = None,
        entity_threshold_grid: Optional[List[float]] = None,
        gap_threshold_grid: Optional[List[float]] = None,
        verbose: bool = True,
        n_jobs: int = -1,
    ) -> ThresholdOptimizationReport:
        """Jointly optimize pair, entity, and gap thresholds on OOF predictions."""
        # 1. Compare Raw vs Calibrated performance
        has_calibrated = "calibrated_score" in scored_pairs_df.columns
        score_cols_to_check = ["raw_score"]
        if has_calibrated:
            score_cols_to_check.append("calibrated_score")

        raw_vs_cal: Dict[str, float] = {}
        best_overall_score = -1.0
        best_overall_policy: Optional[EntityDecisionPolicy] = None
        best_score_col = "raw_score"
        best_candidates_by_s1: Optional[Dict[str, List[Tuple[str, float]]]] = None

        p_grid = pair_threshold_grid or [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        e_grid = entity_threshold_grid or [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        g_grid = gap_threshold_grid or [0.08, 0.12, 0.15, 0.20, 0.25]

        valid_combos = [
            (p, e, g) for p in p_grid for e in e_grid if e >= p for g in g_grid
        ]
        total_combos = len(valid_combos)
        n_workers = os.cpu_count() or 4 if n_jobs == -1 else n_jobs
        n_workers = max(1, min(n_workers, 32))

        for sc_idx, sc in enumerate(score_cols_to_check, start=1):
            _, cands_by_s1 = EntityAggregator.aggregate_entity_candidates(
                scored_pairs_df, all_s1_ids, score_col=sc
            )

            best_sc_f05 = -1.0
            best_sc_policy = None

            if verbose:
                print(f"  [Threshold Search {sc_idx}/{len(score_cols_to_check)}: {sc}] Evaluating {total_combos} combinations sequentially (memory safe)...")
            
            combo_results = []
            for combo in valid_combos:
                combo_results.append(_evaluate_combo_worker((combo, cands_by_s1, ground_truth, all_s1_ids)))

            for pol, score in combo_results:
                if score > best_sc_f05:
                    best_sc_f05 = score
                    best_sc_policy = pol

                if verbose:
                    print(f"\r  [Threshold Search {sc_idx}/{len(score_cols_to_check)}: {sc}] Completed {total_combos}/{total_combos} combos | Best F0.5: {best_sc_f05:.4f}")
            else:
                for idx, (p_th, e_th, g_th) in enumerate(valid_combos, start=1):
                    if verbose and (idx % 15 == 0 or idx == total_combos or idx <= 5):
                        pct = 100.0 * idx / total_combos
                        print(
                            f"\r  [Threshold Search {sc_idx}/{len(score_cols_to_check)}: {sc}] Combo {idx}/{total_combos} ({pct:.1f}%) | Best F0.5: {best_sc_f05:.4f}",
                            end="",
                            flush=True,
                        )
                    pol = EntityDecisionPolicy(
                        pair_threshold=p_th,
                        entity_threshold=e_th,
                        gap_threshold=g_th,
                    )
                    preds = EntityAggregator.apply_policy(cands_by_s1, pol, all_s1_ids)
                    score = MetricsEvaluator.compute_macro_f05(preds, ground_truth, all_s1_ids)

                    if score > best_sc_f05:
                        best_sc_f05 = score
                        best_sc_policy = pol

                if verbose:
                    print(f"\r  [Threshold Search {sc_idx}/{len(score_cols_to_check)}: {sc}] Completed {total_combos}/{total_combos} combos | Best F0.5: {best_sc_f05:.4f}    ")

            raw_vs_cal[sc] = best_sc_f05
            if best_sc_f05 > best_overall_score:
                best_overall_score = best_sc_f05
                best_overall_policy = best_sc_policy
                best_score_col = sc
                best_candidates_by_s1 = cands_by_s1

        assert best_overall_policy is not None
        assert best_candidates_by_s1 is not None

        # 2. Stability Plateau Analysis (+- 0.02 perturbations)
        plateau_scores: Dict[str, float] = {}
        for delta_name, d_pair, d_ent in [
            ("baseline", 0.0, 0.0),
            ("pair_+0.02", 0.02, 0.0),
            ("pair_-0.02", -0.02, 0.0),
            ("entity_+0.02", 0.0, 0.02),
            ("entity_-0.02", 0.0, -0.02),
            ("both_+0.02", 0.02, 0.02),
            ("both_-0.02", -0.02, -0.02),
        ]:
            perturbed_pol = EntityDecisionPolicy(
                pair_threshold=max(0.01, min(0.99, best_overall_policy.pair_threshold + d_pair)),
                entity_threshold=max(0.01, min(0.99, best_overall_policy.entity_threshold + d_ent)),
                gap_threshold=best_overall_policy.gap_threshold,
            )
            p_preds = EntityAggregator.apply_policy(
                best_candidates_by_s1, perturbed_pol, all_s1_ids
            )
            plateau_scores[delta_name] = MetricsEvaluator.compute_macro_f05(
                p_preds, ground_truth, all_s1_ids
            )

        # 3. Fold Dispersion Analysis
        fold_scores: Dict[int, float] = {}
        unique_folds = sorted(list(set(fold_map.values())))
        for f in unique_folds:
            fold_s1_ids = {s1 for s1, fold_id in fold_map.items() if fold_id == f and s1 in all_s1_ids}
            if fold_s1_ids:
                preds_fold = EntityAggregator.apply_policy(
                    best_candidates_by_s1, best_overall_policy, fold_s1_ids
                )
                fold_scores[f] = MetricsEvaluator.compute_macro_f05(
                    preds_fold, ground_truth, fold_s1_ids
                )

        # 4. Domain / Country Shift Simulation & Policy Evaluation
        country_scores: Dict[str, float] = {}
        policy_comparison: Dict[str, Dict[str, float]] = {}

        if s1_country_map:
            unique_countries = sorted(list({s1_country_map.get(s1, "unknown") for s1 in all_s1_ids}))
            for c in unique_countries:
                c_s1_ids = {s1 for s1 in all_s1_ids if s1_country_map.get(s1, "unknown") == c}
                if c_s1_ids:
                    c_preds = EntityAggregator.apply_policy(
                        best_candidates_by_s1, best_overall_policy, c_s1_ids
                    )
                    country_scores[c] = MetricsEvaluator.compute_macro_f05(
                        c_preds, ground_truth, c_s1_ids
                    )

            # Evaluate Policy A: Global Threshold
            pol_a_scores = list(country_scores.values())
            policy_comparison["Policy_A_Global"] = {
                "mean_f05": float(np.mean(pol_a_scores)),
                "worst_case_f05": float(np.min(pol_a_scores)),
                "std_f05": float(np.std(pol_a_scores)),
            }

            # Evaluate Policy C: Conservative Global Threshold (more precision-heavy)
            cons_policy = EntityDecisionPolicy(
                pair_threshold=min(0.95, best_overall_policy.pair_threshold + 0.05),
                entity_threshold=min(0.95, best_overall_policy.entity_threshold + 0.05),
                gap_threshold=best_overall_policy.gap_threshold,
                policy_name="conservative_global",
            )
            cons_country_scores = []
            for c in unique_countries:
                c_s1_ids = {s1 for s1 in all_s1_ids if s1_country_map.get(s1, "unknown") == c}
                if c_s1_ids:
                    c_preds = EntityAggregator.apply_policy(
                        best_candidates_by_s1, cons_policy, c_s1_ids
                    )
                    cons_country_scores.append(
                        MetricsEvaluator.compute_macro_f05(c_preds, ground_truth, c_s1_ids)
                    )
            policy_comparison["Policy_C_Conservative"] = {
                "mean_f05": float(np.mean(cons_country_scores)),
                "worst_case_f05": float(np.min(cons_country_scores)),
                "std_f05": float(np.std(cons_country_scores)),
            }

        return ThresholdOptimizationReport(
            best_policy=best_overall_policy,
            best_macro_f05=best_overall_score,
            raw_vs_calibrated_comparison=raw_vs_cal,
            stability_plateau_f05=plateau_scores,
            fold_f05_scores=fold_scores,
            country_f05_scores=country_scores,
            policy_comparison=policy_comparison,
        )
