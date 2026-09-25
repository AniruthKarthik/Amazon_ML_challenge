"""Pairwise ranking and classification models for business entity resolution.

Implements:
1. Leak-free cross-fitted LightGBM pair scorer using bipartite graph folds.
2. Out-of-fold (OOF) raw prediction generation.
3. Isotonic probability calibration on OOF predictions.
4. Bagged fold ensemble prediction for test inference.
5. Diagnostic pair-level evaluation (AUC, PR-AUC, feature importances).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class OOFResult:
    oof_pairs_df: pd.DataFrame
    models: List[lgb.LGBMClassifier]
    calibrator: IsotonicRegression
    feature_cols: List[str]
    pair_auc: float
    pair_pr_auc: float
    feature_importances: Dict[str, float]


class PairScorer:
    """Trains and predicts candidate pair match probabilities using LightGBM."""

    def __init__(self, lgb_params: Optional[Dict[str, Any]] = None):
        self.lgb_params = lgb_params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 120,
            "learning_rate": 0.08,
            "num_leaves": 31,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.models: List[lgb.LGBMClassifier] = []
        self.calibrator: Optional[IsotonicRegression] = None
        self.feature_cols: List[str] = []

    def train_oof(
        self,
        pairs_df: pd.DataFrame,
        feature_cols: List[str],
        fold_map: Dict[str, int],
        ground_truth: Dict[str, Set[str]],
        verbose: bool = True,
    ) -> OOFResult:
        """Perform strict leak-free out-of-fold (OOF) cross-validation.

        Parameters
        ----------
        pairs_df : pd.DataFrame containing feature columns and source1_entity_id, candidate_entity_id.
        feature_cols : list of feature column names to feed to LightGBM.
        fold_map : mapping source1_entity_id -> fold_id.
        ground_truth : mapping source1_entity_id -> set of true matched candidate IDs.
        verbose : whether to display live training progress status.

        Returns
        -------
        OOFResult with OOF predictions, trained models, calibrator, and diagnostics.
        """
        self.feature_cols = feature_cols
        self.models = []

        df = pairs_df.copy()

        # Construct ground truth binary target
        target_labels = []
        for _, row in df.iterrows():
            s1 = row["source1_entity_id"]
            cand = row["candidate_entity_id"]
            is_match = 1.0 if cand in ground_truth.get(s1, set()) else 0.0
            target_labels.append(is_match)
        df["target"] = target_labels

        # Assign fold ID per row based on source1_entity_id
        df["fold"] = df["source1_entity_id"].map(fold_map)
        if df["fold"].isna().any():
            missing_count = df["fold"].isna().sum()
            raise ValueError(f"{missing_count} pairs missing fold assignment!")
        df["fold"] = df["fold"].astype(int)

        unique_folds = sorted(df["fold"].unique())
        n_folds = len(unique_folds)

        oof_raw_scores = np.zeros(len(df), dtype=np.float32)
        feature_importances_accum = np.zeros(len(feature_cols), dtype=np.float64)

        X_all = df[feature_cols].to_numpy(dtype=np.float32)
        y_all = df["target"].to_numpy(dtype=np.float32)

        for fold_idx, fold_id in enumerate(unique_folds, start=1):
            val_idx = df["fold"] == fold_id
            train_idx = ~val_idx

            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_val, y_val = X_all[val_idx], y_all[val_idx]

            # Fit LightGBM model for this fold
            model = lgb.LGBMClassifier(**self.lgb_params)
            # Check if there are both positive and negative examples in train
            if len(np.unique(y_train)) < 2:
                # Edge case in tiny test fixtures
                oof_raw_scores[val_idx] = 0.5
                self.models.append(model)
                continue

            n_trees = self.lgb_params.get("n_estimators", 120)
            callbacks = []
            if verbose:
                def progress_cb(env):
                    curr = env.iteration + 1
                    total = env.end_iteration or n_trees
                    pct = 100.0 * curr / total
                    print(
                        f"\r  [Training Fold {fold_idx}/{n_folds}] Tree {curr}/{total} ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
                callbacks.append(progress_cb)

            model.fit(X_train, y_train, callbacks=callbacks)
            if verbose:
                print(f"\r  [Training Fold {fold_idx}/{n_folds}] Completed {n_trees}/{n_trees} trees.            ")

            self.models.append(model)

            val_preds = model.predict_proba(X_val)[:, 1]
            oof_raw_scores[val_idx] = val_preds
            feature_importances_accum += model.feature_importances_

        df["raw_score"] = oof_raw_scores

        # Fit Isotonic Calibration on out-of-fold scores
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        # Ensure at least 2 distinct values for calibration
        if len(np.unique(oof_raw_scores)) >= 2:
            self.calibrator.fit(oof_raw_scores, y_all)
            df["calibrated_score"] = self.calibrator.predict(oof_raw_scores)
        else:
            df["calibrated_score"] = oof_raw_scores

        # Diagnostic metrics
        if len(np.unique(y_all)) >= 2:
            pair_auc = float(roc_auc_score(y_all, oof_raw_scores))
            pair_pr_auc = float(average_precision_score(y_all, oof_raw_scores))
        else:
            pair_auc = 1.0
            pair_pr_auc = 1.0

        avg_importances = feature_importances_accum / max(1, len(self.models))
        feat_imp_dict = {
            col: float(imp) for col, imp in zip(feature_cols, avg_importances)
        }

        return OOFResult(
            oof_pairs_df=df,
            models=self.models,
            calibrator=self.calibrator,
            feature_cols=feature_cols,
            pair_auc=pair_auc,
            pair_pr_auc=pair_pr_auc,
            feature_importances=feat_imp_dict,
        )

    def predict(
        self,
        pairs_df: pd.DataFrame,
        use_calibrated: bool = False,
    ) -> np.ndarray:
        """Generate test predictions by ensembling across all fold models."""
        if not self.models:
            raise RuntimeError("PairScorer must be trained before predicting.")

        X = pairs_df[self.feature_cols].to_numpy(dtype=np.float32)
        if len(X) == 0:
            return np.array([], dtype=np.float32)

        # Average predictions from all fold models (bagging)
        preds = np.zeros(len(X), dtype=np.float64)
        valid_models = [m for m in self.models if hasattr(m, "classes_")]
        if not valid_models:
            return np.zeros(len(X), dtype=np.float32)

        for model in valid_models:
            preds += model.predict_proba(X)[:, 1]
        preds /= len(valid_models)

        if use_calibrated and self.calibrator is not None:
            preds = self.calibrator.predict(preds)

        return preds.astype(np.float32)
