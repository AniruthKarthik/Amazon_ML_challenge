"""Unit and integration tests for Hard-Negative Mining (Phase 11) and Meta-Model (Phase 12)."""

import pandas as pd
import pytest
from chimera_submission.code.business_entity_resolution.src.entity_decision import (
    EntityDecisionPolicy,
)
from chimera_submission.code.business_entity_resolution.src.hard_negatives import (
    HardNegativeMiner,
)
from chimera_submission.code.business_entity_resolution.src.meta_model import (
    EntityMetaModel,
    StrategyComparisonResult,
)


def test_hard_negative_miner():
    oof_pairs = pd.DataFrame([
        # True positive: should NOT be mined
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "target": 1.0, "raw_score": 0.90},
        # False positive with high score: SHOULD be mined
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-2", "target": 0.0, "raw_score": 0.75},
        # True negative with low score: below threshold, should NOT be mined
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-3", "target": 0.0, "raw_score": 0.10},
    ])

    mined = HardNegativeMiner.mine_hard_negatives(oof_pairs, score_threshold=0.50)
    assert len(mined) == 1
    assert mined.iloc[0]["candidate_entity_id"] == "S2-2"
    assert mined.iloc[0]["is_hard_negative"] == 1.0

    # Test dataset augmentation
    augmented = HardNegativeMiner.augment_dataset(oof_pairs, mined)
    assert len(augmented) == len(oof_pairs)  # deduplicated


def test_entity_meta_model_and_strategy_comparison():
    meta_rows = []
    ground_truth = {}
    fold_map = {}
    cands_by_s1 = {}

    for i in range(12):
        s1_id = f"S1-{i:02d}"
        fold_map[s1_id] = i % 3
        is_match = (i % 2 == 0)

        if is_match:
            cand_id = f"S2-{i:02d}"
            ground_truth[s1_id] = {cand_id}
            cands_by_s1[s1_id] = [(cand_id, 0.92)]
            meta_rows.append({
                "source1_entity_id": s1_id,
                "meta_candidate_count": 1.0,
                "meta_max_score": 0.92,
                "meta_second_max_score": 0.0,
                "meta_score_gap": 0.92,
                "meta_mean_score": 0.92,
                "meta_std_score": 0.0,
                "meta_n_above_05": 1.0,
                "meta_n_above_07": 1.0,
                "meta_n_above_09": 1.0,
            })
        else:
            cand_id = f"S2-fake-{i:02d}"
            ground_truth[s1_id] = set()
            cands_by_s1[s1_id] = [(cand_id, 0.25)]
            meta_rows.append({
                "source1_entity_id": s1_id,
                "meta_candidate_count": 1.0,
                "meta_max_score": 0.25,
                "meta_second_max_score": 0.0,
                "meta_score_gap": 0.25,
                "meta_mean_score": 0.25,
                "meta_std_score": 0.0,
                "meta_n_above_05": 0.0,
                "meta_n_above_07": 0.0,
                "meta_n_above_09": 0.0,
            })

    meta_df = pd.DataFrame(meta_rows)
    all_s1 = set(meta_df["source1_entity_id"])

    meta_model = EntityMetaModel(
        lgb_params={
            "objective": "binary",
            "n_estimators": 15,
            "min_child_samples": 2,
            "num_leaves": 7,
            "verbose": -1,
            "random_state": 42,
        }
    )

    oof_meta_df = meta_model.train_oof(meta_df, ground_truth, fold_map)
    assert "meta_has_match_prob" in oof_meta_df.columns
    assert len(oof_meta_df) == 12

    base_policy = EntityDecisionPolicy(pair_threshold=0.50, entity_threshold=0.60)
    comparison = EntityMetaModel.compare_strategies(
        cands_by_s1=cands_by_s1,
        entity_meta_df=oof_meta_df,
        ground_truth=ground_truth,
        all_s1_ids=all_s1,
        base_policy=base_policy,
    )

    assert isinstance(comparison, StrategyComparisonResult)
    assert comparison.strategy_b_f05 > 0.90
