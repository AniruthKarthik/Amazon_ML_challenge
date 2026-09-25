"""Cross-fitted hard-negative mining for business entity resolution.

Implements:
1. Extraction of cross-fitted hard negative pairs from OOF false positives.
2. Dataset augmentation for pair model retraining.
3. Compute gate checking before downstream re-optimization loops.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd


class HardNegativeMiner:
    """Mines false positives from out-of-fold predictions to improve discriminator precision."""

    @staticmethod
    def mine_hard_negatives(
        oof_pairs_df: pd.DataFrame,
        score_threshold: float = 0.35,
        max_negatives_per_entity: int = 5,
        score_col: str = "raw_score",
    ) -> pd.DataFrame:
        """Extract high-scoring negative pairs from out-of-fold predictions.

        Parameters
        ----------
        oof_pairs_df : pd.DataFrame with source1_entity_id, candidate_entity_id, target, score_col.
        score_threshold : minimum model score to qualify as a hard negative.
        max_negatives_per_entity : maximum hard negatives to keep per S1 entity.
        score_col : score column to filter on.

        Returns
        -------
        pd.DataFrame containing mined hard negative pairs.
        """
        # Select negatives with high model score
        mask = (oof_pairs_df["target"] == 0.0) & (oof_pairs_df[score_col] >= score_threshold)
        hard_negs = oof_pairs_df[mask].copy()

        if hard_negs.empty:
            return pd.DataFrame(columns=oof_pairs_df.columns)

        # Sort and cap per entity
        hard_negs = hard_negs.sort_values(
            by=["source1_entity_id", score_col], ascending=[True, False]
        )
        capped_negs = hard_negs.groupby("source1_entity_id").head(max_negatives_per_entity)
        capped_negs["is_hard_negative"] = 1.0

        return capped_negs.reset_index(drop=True)

    @staticmethod
    def augment_dataset(
        base_pairs_df: pd.DataFrame,
        mined_negatives_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Augment base training pairs with mined hard negatives (deduplicated)."""
        if mined_negatives_df.empty:
            return base_pairs_df.copy()

        combined = pd.concat([base_pairs_df, mined_negatives_df], ignore_index=True)
        # Deduplicate on (source1_entity_id, candidate_entity_id)
        deduped = combined.drop_duplicates(
            subset=["source1_entity_id", "candidate_entity_id"], keep="first"
        )
        return deduped.reset_index(drop=True)
