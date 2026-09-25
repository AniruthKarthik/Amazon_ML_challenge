"""Unit and integration tests for LightGBM pair scorer and OOF predictions (Phase 7 & 8)."""

import numpy as np
import pandas as pd
import pytest
from chimera_submission.code.business_entity_resolution.src.pair_model import (
    OOFResult,
    PairScorer,
)


@pytest.fixture
def synthetic_pairs_dataset():
    # 20 S1 entities, 2 candidates each = 40 candidate pairs
    rows = []
    ground_truth = {}
    fold_map = {}

    rng = np.random.RandomState(42)

    for i in range(20):
        s1_id = f"S1-{i:03d}"
        cand_pos = f"S2-{i:03d}"
        cand_neg = f"S2-{(i + 100):03d}"

        # S1 matches cand_pos
        ground_truth[s1_id] = {cand_pos}
        fold_map[s1_id] = i % 4  # 4 folds

        # Positive pair: high similarity features
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_pos,
            "feat_exact_clean_name": 1.0,
            "feat_lev_clean_name": 0.95 + rng.uniform(-0.05, 0.05),
            "feat_jw_clean_name": 0.98 + rng.uniform(-0.02, 0.02),
            "feat_jaccard_name": 0.90 + rng.uniform(-0.05, 0.05),
            "feat_channel_count": 3.0,
            "feat_retrieval_max_score": 1.0,
        })

        # Negative pair: low similarity features
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_neg,
            "feat_exact_clean_name": 0.0,
            "feat_lev_clean_name": 0.20 + rng.uniform(-0.1, 0.1),
            "feat_jw_clean_name": 0.30 + rng.uniform(-0.1, 0.1),
            "feat_jaccard_name": 0.10 + rng.uniform(-0.05, 0.05),
            "feat_channel_count": 1.0,
            "feat_retrieval_max_score": 0.4,
        })

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    return df, feature_cols, fold_map, ground_truth


def test_train_oof_and_calibration(synthetic_pairs_dataset):
    df, feature_cols, fold_map, ground_truth = synthetic_pairs_dataset

    scorer = PairScorer(
        lgb_params={
            "objective": "binary",
            "n_estimators": 20,
            "min_child_samples": 2,
            "num_leaves": 7,
            "verbose": -1,
            "random_state": 42,
        }
    )

    result = scorer.train_oof(df, feature_cols, fold_map, ground_truth)

    assert isinstance(result, OOFResult)
    assert len(result.models) == 4
    assert "raw_score" in result.oof_pairs_df.columns
    assert "calibrated_score" in result.oof_pairs_df.columns

    # High discrimination expected on synthetic data
    assert result.pair_auc > 0.90
    assert result.pair_pr_auc > 0.90

    # Ensure all predictions are in [0.0, 1.0]
    raw_scores = result.oof_pairs_df["raw_score"].to_numpy()
    cal_scores = result.oof_pairs_df["calibrated_score"].to_numpy()
    assert np.all((raw_scores >= 0.0) & (raw_scores <= 1.0))
    assert np.all((cal_scores >= 0.0) & (cal_scores <= 1.0))


def test_predict_ensemble(synthetic_pairs_dataset):
    df, feature_cols, fold_map, ground_truth = synthetic_pairs_dataset

    scorer = PairScorer(
        lgb_params={
            "objective": "binary",
            "n_estimators": 20,
            "min_child_samples": 2,
            "num_leaves": 7,
            "verbose": -1,
            "random_state": 42,
        }
    )

    scorer.train_oof(df, feature_cols, fold_map, ground_truth)

    # Predict on test data
    test_slice = df.head(4).copy()
    preds_raw = scorer.predict(test_slice, use_calibrated=False)
    preds_cal = scorer.predict(test_slice, use_calibrated=True)

    assert len(preds_raw) == 4
    assert len(preds_cal) == 4
    assert np.all((preds_raw >= 0.0) & (preds_raw <= 1.0))
    assert np.all((preds_cal >= 0.0) & (preds_cal <= 1.0))

    # Positive pair should have higher score than negative pair
    assert preds_raw[0] > preds_raw[1]
