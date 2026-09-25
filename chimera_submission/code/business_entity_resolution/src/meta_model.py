"""Entity-level meta-model for singleton and set size decision (Phase 12).

Implements:
1. Cross-fitted meta-classifier training on entity meta-features.
2. Prediction of singleton probability P(is_singleton) or P(has_match).
3. Empirical comparison of decision strategies:
   - Strategy A: Pair score + deterministic pair threshold
   - Strategy B: Pair score + top-1 / score-gap logic
   - Strategy C: Pair score + entity-level meta-model gate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from chimera_submission.code.business_entity_resolution.src.entity_decision import (
    EntityAggregator,
    EntityDecisionPolicy,
)
from chimera_submission.code.business_entity_resolution.src.evaluation import MetricsEvaluator


@dataclass
class StrategyComparisonResult:
    strategy_a_f05: float
    strategy_b_f05: float
    strategy_c_f05: float
    winning_strategy: str
    meta_model_accepted: bool


class EntityMetaModel:
    """Predicts entity-level matching properties (has_match vs singleton)."""

    def __init__(self, lgb_params: Optional[Dict] = None):
        self.lgb_params = lgb_params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": 60,
            "learning_rate": 0.08,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 10,
            "random_state": 42,
            "verbose": -1,
        }
        self.models: List[lgb.LGBMClassifier] = []
        self.meta_feature_cols: List[str] = []

    def train_oof(
        self,
        entity_meta_df: pd.DataFrame,
        ground_truth: Dict[str, Set[str]],
        fold_map: Dict[str, int],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Train cross-fitted entity meta-model to predict P(has_match).

        Parameters
        ----------
        entity_meta_df : pd.DataFrame with meta-features (one row per S1 entity).
        ground_truth : mapping source1_entity_id -> set of matched candidate IDs.
        fold_map : mapping source1_entity_id -> fold_id.
        verbose : whether to display live training progress status.

        Returns
        -------
        pd.DataFrame with added 'meta_has_match_prob'.
        """
        df = entity_meta_df.copy()

        # Binary target: 1 if entity has at least 1 true match, 0 if singleton
        df["target_has_match"] = df["source1_entity_id"].apply(
            lambda s1: 1.0 if len(ground_truth.get(s1, set())) > 0 else 0.0
        )
        df["fold"] = df["source1_entity_id"].map(fold_map).fillna(0).astype(int)

        self.meta_feature_cols = [
            c for c in df.columns if c.startswith("meta_") and not c.startswith("meta_has_match_prob")
        ]

        unique_folds = sorted(df["fold"].unique())
        n_folds = len(unique_folds)
        oof_probs = np.zeros(len(df), dtype=np.float32)

        X_all = df[self.meta_feature_cols].to_numpy(dtype=np.float32)
        y_all = df["target_has_match"].to_numpy(dtype=np.float32)

        self.models = []
        for fold_idx, fold_id in enumerate(unique_folds, start=1):
            val_idx = df["fold"] == fold_id
            train_idx = ~val_idx

            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_val = X_all[val_idx]

            model = lgb.LGBMClassifier(**self.lgb_params)
            if len(np.unique(y_train)) < 2:
                oof_probs[val_idx] = 0.5
                self.models.append(model)
                continue

            n_trees = self.lgb_params.get("n_estimators", 60)
            callbacks = []
            if verbose:
                def progress_cb(env):
                    curr = env.iteration + 1
                    total = env.end_iteration or n_trees
                    pct = 100.0 * curr / total
                    print(
                        f"\r  [Meta-Model Fold {fold_idx}/{n_folds}] Tree {curr}/{total} ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
                callbacks.append(progress_cb)

            model.fit(X_train, y_train, callbacks=callbacks)
            if verbose:
                print(f"\r  [Meta-Model Fold {fold_idx}/{n_folds}] Completed {n_trees}/{n_trees} trees.            ")

            self.models.append(model)
            val_preds = model.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = val_preds

        df["meta_has_match_prob"] = oof_probs
        return df

    def predict(self, entity_meta_df: pd.DataFrame) -> np.ndarray:
        """Predict P(has_match) for unseen test entities using fold ensemble."""
        if not self.models:
            raise RuntimeError("EntityMetaModel must be trained before predicting.")

        X = entity_meta_df[self.meta_feature_cols].to_numpy(dtype=np.float32)
        preds = np.zeros(len(X), dtype=np.float64)

        valid_models = [m for m in self.models if hasattr(m, "classes_")]
        if not valid_models:
            return np.ones(len(X), dtype=np.float32)

        for m in valid_models:
            preds += m.predict_proba(X)[:, 1]
        preds /= len(valid_models)

        return preds.astype(np.float32)

    @classmethod
    def compare_strategies(
        cls,
        cands_by_s1: Dict[str, List[Tuple[str, float]]],
        entity_meta_df: pd.DataFrame,
        ground_truth: Dict[str, Set[str]],
        all_s1_ids: Set[str],
        base_policy: EntityDecisionPolicy,
        meta_gate_threshold: float = 0.50,
    ) -> StrategyComparisonResult:
        """Empirically compare Strategy A, B, and C on Macro F_0.5.

        Strategy A: Pair score + deterministic pair threshold
        Strategy B: Pair score + top-1 / score gap logic
        Strategy C: Pair score + entity-level meta-model gate
        """
        # Strategy A: simple pair threshold
        policy_a = EntityDecisionPolicy(
            pair_threshold=base_policy.pair_threshold,
            entity_threshold=base_policy.pair_threshold,
            gap_threshold=999.0,  # no gap constraint
        )
        preds_a = EntityAggregator.apply_policy(cands_by_s1, policy_a, all_s1_ids)
        score_a = MetricsEvaluator.compute_macro_f05(preds_a, ground_truth, all_s1_ids)

        # Strategy B: top-1 + score gap logic
        policy_b = base_policy
        preds_b = EntityAggregator.apply_policy(cands_by_s1, policy_b, all_s1_ids)
        score_b = MetricsEvaluator.compute_macro_f05(preds_b, ground_truth, all_s1_ids)

        # Strategy C: Strategy B gated by entity meta-model
        meta_probs = dict(
            zip(entity_meta_df["source1_entity_id"], entity_meta_df["meta_has_match_prob"])
        )
        preds_c: Dict[str, Set[str]] = {}
        for s1_id in all_s1_ids:
            if meta_probs.get(s1_id, 1.0) < meta_gate_threshold:
                preds_c[s1_id] = set()
            else:
                preds_c[s1_id] = preds_b.get(s1_id, set())

        score_c = MetricsEvaluator.compute_macro_f05(preds_c, ground_truth, all_s1_ids)

        winning = "Strategy_B"
        accepted = False
        if score_c > max(score_a, score_b) + 0.001:
            winning = "Strategy_C_MetaModel"
            accepted = True
        elif score_a > score_b:
            winning = "Strategy_A_PairThreshold"

        return StrategyComparisonResult(
            strategy_a_f05=score_a,
            strategy_b_f05=score_b,
            strategy_c_f05=score_c,
            winning_strategy=winning,
            meta_model_accepted=accepted,
        )
