"""Unit and integration tests for Entity-Level Decision & Threshold Optimization (Phase 9 & 10)."""

import pandas as pd
import pytest
from chimera_submission.code.business_entity_resolution.src.entity_decision import (
    EntityAggregator,
    EntityDecisionPolicy,
    ThresholdOptimizationReport,
    ThresholdOptimizer,
)


@pytest.fixture
def synthetic_scored_dataset():
    # 5 S1 entities
    # S1-1: 2 candidates, both high score (multi-match)
    # S1-2: 1 candidate, high score (single match)
    # S1-3: 1 candidate, low score (should be rejected as singleton)
    # S1-4: 0 candidates (singleton)
    # S1-5: 2 candidates, 1 high, 1 low (top match only)
    pairs = [
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "raw_score": 0.92, "calibrated_score": 0.95},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-1", "raw_score": 0.88, "calibrated_score": 0.90},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-2", "raw_score": 0.85, "calibrated_score": 0.87},
        {"source1_entity_id": "S1-3", "candidate_entity_id": "S2-3", "raw_score": 0.30, "calibrated_score": 0.20},
        {"source1_entity_id": "S1-5", "candidate_entity_id": "S2-5", "raw_score": 0.80, "calibrated_score": 0.82},
        {"source1_entity_id": "S1-5", "candidate_entity_id": "S3-5", "raw_score": 0.40, "calibrated_score": 0.35},
    ]
    df = pd.DataFrame(pairs)
    all_s1 = {"S1-1", "S1-2", "S1-3", "S1-4", "S1-5"}
    ground_truth = {
        "S1-1": {"S2-1", "S3-1"},
        "S1-2": {"S2-2"},
        "S1-3": set(),  # true singleton
        "S1-4": set(),  # true singleton
        "S1-5": {"S2-5"},
    }
    fold_map = {"S1-1": 0, "S1-2": 0, "S1-3": 1, "S1-4": 1, "S1-5": 0}
    country_map = {"S1-1": "US", "S1-2": "US", "S1-3": "India", "S1-4": "India", "S1-5": "US"}
    return df, all_s1, ground_truth, fold_map, country_map


def test_aggregate_entity_candidates(synthetic_scored_dataset):
    df, all_s1, _, _, _ = synthetic_scored_dataset
    meta_df, cands_by_s1 = EntityAggregator.aggregate_entity_candidates(df, all_s1, "raw_score")

    assert len(meta_df) == len(all_s1)
    # Check singleton S1-4 has 0 candidates and 0 max score
    s1_4_meta = meta_df[meta_df["source1_entity_id"] == "S1-4"].iloc[0]
    assert s1_4_meta["meta_candidate_count"] == 0.0
    assert s1_4_meta["meta_max_score"] == 0.0

    # Check S1-1 has 2 candidates, max 0.92, second max 0.88, gap 0.04
    s1_1_meta = meta_df[meta_df["source1_entity_id"] == "S1-1"].iloc[0]
    assert s1_1_meta["meta_candidate_count"] == 2.0
    assert s1_1_meta["meta_max_score"] == 0.92
    assert s1_1_meta["meta_second_max_score"] == 0.88
    assert s1_1_meta["meta_score_gap"] == pytest.approx(0.04)


def test_apply_policy_logic(synthetic_scored_dataset):
    df, all_s1, _, _, _ = synthetic_scored_dataset
    _, cands_by_s1 = EntityAggregator.aggregate_entity_candidates(df, all_s1, "raw_score")

    policy = EntityDecisionPolicy(
        pair_threshold=0.50,
        entity_threshold=0.60,
        gap_threshold=0.10,
    )
    preds = EntityAggregator.apply_policy(cands_by_s1, policy, all_s1)

    # S1-1: top is 0.92, second is 0.88 (gap 0.04 <= 0.10) -> BOTH selected
    assert preds["S1-1"] == {"S2-1", "S3-1"}
    # S1-2: top is 0.85 -> S2-2 selected
    assert preds["S1-2"] == {"S2-2"}
    # S1-3: top is 0.30 < entity_threshold 0.60 -> EMPTY set (correct singleton prediction)
    assert preds["S1-3"] == set()
    # S1-4: 0 candidates -> EMPTY set
    assert preds["S1-4"] == set()
    # S1-5: top is 0.80, second is 0.40 (gap 0.40 > 0.10) -> ONLY S2-5 selected
    assert preds["S1-5"] == {"S2-5"}


def test_threshold_optimizer(synthetic_scored_dataset):
    df, all_s1, ground_truth, fold_map, country_map = synthetic_scored_dataset

    report = ThresholdOptimizer.optimize(
        scored_pairs_df=df,
        ground_truth=ground_truth,
        all_s1_ids=all_s1,
        fold_map=fold_map,
        s1_country_map=country_map,
        pair_threshold_grid=[0.40, 0.50, 0.60],
        entity_threshold_grid=[0.50, 0.60, 0.70],
        gap_threshold_grid=[0.10, 0.15],
    )

    assert isinstance(report, ThresholdOptimizationReport)
    # The optimal thresholds should achieve perfect 1.0 macro F_0.5 on this dataset
    assert report.best_macro_f05 == 1.0
    assert "raw_score" in report.raw_vs_calibrated_comparison
    assert "baseline" in report.stability_plateau_f05
    assert len(report.fold_f05_scores) > 0
    assert len(report.country_f05_scores) == 2


def test_to_matching_results_tsv():
    preds = {
        "S1-001": {"S2-001", "S3-002"},
        "S1-002": set(),
    }
    all_s1 = {"S1-001", "S1-002"}
    tsv_df = EntityAggregator.to_matching_results_tsv(preds, all_s1)

    assert len(tsv_df) == 2
    assert list(tsv_df.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert tsv_df.loc[0, "matched_entity_ids"] == "S2-001,S3-002"
    assert tsv_df.loc[1, "matched_entity_ids"] == ""
