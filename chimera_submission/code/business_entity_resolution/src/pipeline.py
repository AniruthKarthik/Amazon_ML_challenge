"""End-to-end Business Entity Resolution pipeline.

Integrates:
1. Data loading & contract validation (Phase 1)
2. Multi-view normalization (Phase 2)
3. Multi-channel candidate retrieval (Phase 3 & 5)
4. Tabular pair feature extraction (Phase 6)
5. LightGBM pair scoring with leak-free cross-fitting (Phase 7 & 8)
6. Entity set decision & robust threshold optimization (Phase 9 & 10)
7. Ablation benchmarking & model locking (Phase 15)
8. Deterministic frozen inference & output validation (Phase 16)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
from chimera_submission.code.business_entity_resolution.src.data_contract import (
    BipartiteGraphAnalyzer,
    GroundTruthLoader,
    TSVLoader,
)
from chimera_submission.code.business_entity_resolution.src.entity_decision import (
    EntityAggregator,
    EntityDecisionPolicy,
    ThresholdOptimizationReport,
    ThresholdOptimizer,
)
from chimera_submission.code.business_entity_resolution.src.evaluation import (
    MetricsEvaluator,
    RetrievalBenchmarkReport,
)
from chimera_submission.code.business_entity_resolution.src.features import (
    PairFeatureExtractor,
)
from chimera_submission.code.business_entity_resolution.src.normalization import (
    TextNormalizer,
)
from chimera_submission.code.business_entity_resolution.src.pair_model import (
    OOFResult,
    PairScorer,
)


@dataclass
class PipelineConfig:
    k_folds: int = 5
    random_seed: int = 42
    top_k_name_tfidf: int = 25
    top_k_addr_tfidf: int = 15
    top_k_word_tfidf: int = 15
    min_tfidf_score: float = 0.25
    enable_word_tfidf: bool = False
    max_candidates_per_entity: int = 60
    lgb_n_estimators: int = 120
    lgb_learning_rate: float = 0.08
    lgb_max_depth: int = 6
    lgb_num_leaves: int = 31
    pair_threshold: float = 0.50
    entity_threshold: float = 0.55
    gap_threshold: float = 0.15
    n_jobs: int = -1
    device: str = "cpu"


class BusinessEntityResolutionPipeline:
    """Production end-to-end entity resolution pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.retriever: Optional[Any] = None
        self.pair_scorer: Optional[PairScorer] = None
        self.locked_policy: Optional[EntityDecisionPolicy] = None
        self.feature_cols: List[str] = []
        self.is_fitted: bool = False
        self.ablation_report: Dict[str, Any] = {}

    def fit(
        self,
        train_s1_df: pd.DataFrame,
        train_s2_df: pd.DataFrame,
        train_s3_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
    ) -> BusinessEntityResolutionPipeline:
        """Fit all pipeline stages on training data using leak-free cross-fitting."""
        # 1. Multi-view Normalization
        print("\n  [Stage 1/6] Multi-View Normalization...")
        s1_norm = TextNormalizer.normalize_dataframe(train_s1_df, n_jobs=self.config.n_jobs)
        s2_norm = TextNormalizer.normalize_dataframe(train_s2_df, n_jobs=self.config.n_jobs)
        s3_norm = TextNormalizer.normalize_dataframe(train_s3_df, n_jobs=self.config.n_jobs)
        target_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)
        del s2_norm, s3_norm
        import gc
        gc.collect()

        all_s1_ids = set(s1_norm["entity_id"])
        s1_country_map = dict(zip(s1_norm["entity_id"], s1_norm["country"]))

        # 2. Build Bipartite Graph & Leak-Free Folds (Phase 1)
        print("\n  [Stage 2/6] Bipartite Graph Analysis & Leak-Free Folds...")
        graph_analyzer = BipartiteGraphAnalyzer(ground_truth_df, all_s1_ids=all_s1_ids)
        fold_map = graph_analyzer.create_leak_free_folds(
            k_folds=self.config.k_folds, random_seed=self.config.random_seed
        )

        # Parse ground truth mapping
        ground_truth: Dict[str, Set[str]] = {}
        for s1, raw_targets in zip(ground_truth_df["source1_entity_id"], ground_truth_df["matched_entity_ids"]):
            targets = set([t.strip() for t in raw_targets.split(",") if t.strip()])
            ground_truth[s1] = targets

        # 3. Candidate Retrieval (Phase 3 & 5)
        print("\n  [Stage 3/6] Multi-Channel Candidate Retrieval...")
        from chimera_submission.code.business_entity_resolution.src.retrieval import (
            CandidateRetriever,
        )

        self.retriever = CandidateRetriever(
            top_k_name_tfidf=self.config.top_k_name_tfidf,
            top_k_addr_tfidf=self.config.top_k_addr_tfidf,
            min_tfidf_score=self.config.min_tfidf_score,
            enable_word_tfidf=self.config.enable_word_tfidf,
            top_k_word_tfidf=self.config.top_k_word_tfidf,
            max_candidates_per_entity=self.config.max_candidates_per_entity,
        )
        self.retriever.fit(target_norm)
        candidates_raw = self.retriever.retrieve(s1_norm, n_jobs=self.config.n_jobs)

        # For training: retain all true ground-truth targets + top hard negative candidates
        # This prevents generating millions of unneeded distant negatives that explode RAM.
        training_candidates: Dict[str, Dict[str, Any]] = {}
        for s1_id, target_map in candidates_raw.items():
            true_targets = ground_truth.get(s1_id, set())
            sorted_cands = sorted(
                target_map.items(),
                key=lambda item: (item[0] in true_targets, item[1].channel_count, item[1].max_score),
                reverse=True,
            )
            kept = {}
            neg_count = 0
            for tid, prov in sorted_cands:
                if tid in true_targets:
                    kept[tid] = prov
                elif neg_count < 8:
                    kept[tid] = prov
                    neg_count += 1
            training_candidates[s1_id] = kept

        pairs_df = self.retriever.to_dataframe(training_candidates)
        del candidates_raw, training_candidates
        import gc
        gc.collect()

        # 4. Feature Extraction (Phase 6)
        print("\n  [Stage 4/6] Tabular Pair Feature Engineering...")
        needed_target_ids = set(pairs_df["candidate_entity_id"])
        sub_target = target_norm[target_norm["entity_id"].isin(needed_target_ids)]
        s1_records = s1_norm.set_index("entity_id").to_dict(orient="index")
        target_records = sub_target.set_index("entity_id").to_dict(orient="index")
        del sub_target
        gc.collect()

        feats_df = PairFeatureExtractor.build_features(
            pairs_df,
            s1_records=s1_records,
            cand_records=target_records,
            n_jobs=self.config.n_jobs,
        )
        del s1_records, target_records
        gc.collect()
        self.feature_cols = [c for c in feats_df.columns if c.startswith("feat_")]

        # 5. Train LightGBM Pair Scorer with OOF Predictions (Phase 7 & 8)
        print("\n  [Stage 5/6] Cross-Fitting LightGBM Pair Models & OOF Inference...")
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": self.config.lgb_n_estimators,
            "learning_rate": self.config.lgb_learning_rate,
            "max_depth": self.config.lgb_max_depth,
            "num_leaves": self.config.lgb_num_leaves,
            "random_state": self.config.random_seed,
            "verbose": -1,
            "n_jobs": self.config.n_jobs,
        }
        if self.config.device in ["cuda", "gpu"]:
            lgb_params["device"] = self.config.device

        self.pair_scorer = PairScorer(lgb_params=lgb_params)
        oof_res = self.pair_scorer.train_oof(
            feats_df, self.feature_cols, fold_map=fold_map, ground_truth=ground_truth
        )

        # 6. Joint Threshold Optimization & Robust Policy Selection (Phase 9 & 10)
        print("\n  [Stage 6/6] Robust Joint Threshold Optimization & Plateau Testing...")
        opt_report = ThresholdOptimizer.optimize(
            scored_pairs_df=oof_res.oof_pairs_df,
            ground_truth=ground_truth,
            all_s1_ids=all_s1_ids,
            fold_map=fold_map,
            s1_country_map=s1_country_map,
            n_jobs=self.config.n_jobs,
        )
        self.locked_policy = opt_report.best_policy

        self.ablation_report = {
            "graph_metrics": graph_analyzer.get_metrics(),
            "pair_auc": oof_res.pair_auc,
            "pair_pr_auc": oof_res.pair_pr_auc,
            "best_macro_f05": opt_report.best_macro_f05,
            "raw_vs_calibrated": opt_report.raw_vs_calibrated_comparison,
            "stability_plateau": opt_report.stability_plateau_f05,
            "fold_scores": opt_report.fold_f05_scores,
            "country_scores": opt_report.country_f05_scores,
            "policy_comparison": opt_report.policy_comparison,
            "locked_policy": self.locked_policy,
        }

        self.is_fitted = True
        return self

    def save(self, filepath: Path | str) -> None:
        """Save fitted model weights, decision policy, and configuration to disk."""
        import pickle
        state = {
            "config": self.config,
            "feature_cols": self.feature_cols,
            "pair_scorer": self.pair_scorer,
            "locked_policy": self.locked_policy,
            "ablation_report": self.ablation_report,
            "is_fitted": self.is_fitted,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: Path | str) -> BusinessEntityResolutionPipeline:
        """Load fitted pipeline state from disk."""
        import pickle
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        pipeline = cls(config=state["config"])
        pipeline.feature_cols = state["feature_cols"]
        pipeline.pair_scorer = state["pair_scorer"]
        pipeline.locked_policy = state["locked_policy"]
        pipeline.ablation_report = state["ablation_report"]
        pipeline.is_fitted = state["is_fitted"]
        return pipeline

    def predict(
        self,
        test_s1_df: pd.DataFrame,
        test_s2_df: pd.DataFrame,
        test_s3_df: pd.DataFrame,
        matching_out: Optional[Path] = None,
        candidates_out: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run deterministic frozen inference on test data.

        Returns
        -------
        Tuple of (matching_results_df, candidate_pairs_df) matching official schema.
        """
        if not self.is_fitted or self.pair_scorer is None:
            raise RuntimeError("Pipeline must be fit before running predict.")

        import gc
        # Release training retriever memory
        self.retriever = None
        gc.collect()

        # 1. Multi-view Normalization
        print("\n  [Inference Stage 1/4] Normalizing Test Entities...")
        keep_cols = ["entity_id", "name_clean", "name_core", "address_clean", "address_alias", "country"]

        s1_norm = TextNormalizer.normalize_dataframe(test_s1_df, n_jobs=self.config.n_jobs)
        s1_norm = s1_norm[[c for c in keep_cols if c in s1_norm.columns]].copy()

        s2_norm = TextNormalizer.normalize_dataframe(test_s2_df, n_jobs=self.config.n_jobs)
        s2_norm = s2_norm[[c for c in keep_cols if c in s2_norm.columns]].copy()

        s3_norm = TextNormalizer.normalize_dataframe(test_s3_df, n_jobs=self.config.n_jobs)
        s3_norm = s3_norm[[c for c in keep_cols if c in s3_norm.columns]].copy()

        target_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)
        del s2_norm, s3_norm
        gc.collect()

        all_s1_ids = set(s1_norm["entity_id"])
        s1_country_map = dict(zip(s1_norm["entity_id"], s1_norm["country"]))

        # 2. Candidate Retrieval on Test Target Corpus
        print("\n  [Inference Stage 2/4] Multi-Channel Candidate Retrieval on Test Corpus...")
        from chimera_submission.code.business_entity_resolution.src.retrieval import (
            CandidateRetriever,
        )

        test_retriever = CandidateRetriever(
            top_k_name_tfidf=self.config.top_k_name_tfidf,
            top_k_addr_tfidf=self.config.top_k_addr_tfidf,
            min_tfidf_score=self.config.min_tfidf_score,
            enable_word_tfidf=self.config.enable_word_tfidf,
            top_k_word_tfidf=self.config.top_k_word_tfidf,
            max_candidates_per_entity=self.config.max_candidates_per_entity,
        )
        test_retriever.fit(target_norm)

        total_s1 = len(s1_norm)
        batch_size = 50000

        if total_s1 <= batch_size:
            candidates_raw = test_retriever.retrieve(s1_norm, n_jobs=self.config.n_jobs)
            candidate_pairs_df = test_retriever.to_candidate_pairs_tsv(
                candidates_raw, all_s1_ids=all_s1_ids
            )
            pairs_df = test_retriever.to_dataframe(candidates_raw)

            if pairs_df.empty:
                matching_results_df = EntityAggregator.to_matching_results_tsv({}, all_s1_ids)
                if matching_out is not None and candidates_out is not None:
                    matching_results_df.to_csv(matching_out, sep="\t", index=False, encoding="utf-8")
                    candidate_pairs_df.to_csv(candidates_out, sep="\t", index=False, encoding="utf-8")
                return matching_results_df, candidate_pairs_df

            # 3. Feature Extraction
            print("\n  [Inference Stage 3/4] Pair Feature Extraction on Test Candidates...")
            needed_targets = set(pairs_df["candidate_entity_id"])
            sub_target = target_norm[target_norm["entity_id"].isin(needed_targets)]
            target_records = sub_target.set_index("entity_id").to_dict(orient="index")
            del sub_target

            s1_records = s1_norm.set_index("entity_id").to_dict(orient="index")
            feats_df = PairFeatureExtractor.build_features(
                pairs_df,
                s1_records=s1_records,
                cand_records=target_records,
                n_jobs=self.config.n_jobs,
            )
            del s1_records, target_records
            gc.collect()

            # 4. Ensemble Pair Scoring
            print("\n  [Inference Stage 4/4] Bagged Ensemble Pair Scoring & Applying Locked Policy...")
            scores = self.pair_scorer.predict(feats_df, use_calibrated=False)
            feats_df["raw_score"] = scores

            # 5. Entity Aggregation & Locked Threshold Policy
            _, cands_by_s1 = EntityAggregator.aggregate_entity_candidates(
                feats_df, all_s1_ids, score_col="raw_score"
            )
            active_policy = self.locked_policy or EntityDecisionPolicy()
            predictions = EntityAggregator.apply_policy(
                cands_by_s1, active_policy, all_s1_ids, s1_country_map=s1_country_map
            )

            matching_results_df = EntityAggregator.to_matching_results_tsv(
                predictions, all_s1_ids=all_s1_ids
            )
            if matching_out is not None and candidates_out is not None:
                matching_results_df.to_csv(matching_out, sep="\t", index=False, encoding="utf-8")
                candidate_pairs_df.to_csv(candidates_out, sep="\t", index=False, encoding="utf-8")
            return matching_results_df, candidate_pairs_df
        else:
            # Batch-streamed inference directly to disk to preserve memory on large test sets
            n_batches = (total_s1 + batch_size - 1) // batch_size
            print(f"  [Inference Memory Management] Processing {total_s1} entities across {n_batches} batches of up to {batch_size}...")

            if matching_out is not None and matching_out.exists():
                matching_out.unlink()
            if candidates_out is not None and candidates_out.exists():
                candidates_out.unlink()

            all_candidate_tsv_parts: List[pd.DataFrame] = []
            all_matching_tsv_parts: List[pd.DataFrame] = []
            active_policy = self.locked_policy or EntityDecisionPolicy()
            total_matches_saved = 0
            total_cands_saved = 0

            for b_idx in range(n_batches):
                start = b_idx * batch_size
                end = min(start + batch_size, total_s1)
                batch_s1 = s1_norm.iloc[start:end]
                batch_s1_ids = set(batch_s1["entity_id"])
                pct = 100.0 * end / total_s1
                print(f"    Batch {b_idx + 1}/{n_batches} ({start + 1} to {end} of {total_s1}, {pct:.1f}%)...")

                b_cands_raw = test_retriever.retrieve(batch_s1, verbose=False, n_jobs=self.config.n_jobs)
                b_cand_tsv = test_retriever.to_candidate_pairs_tsv(b_cands_raw, all_s1_ids=batch_s1_ids)

                b_pairs_df = test_retriever.to_dataframe(b_cands_raw)
                if b_pairs_df.empty:
                    b_match_tsv = EntityAggregator.to_matching_results_tsv({}, batch_s1_ids)
                else:
                    b_needed_targets = set(b_pairs_df["candidate_entity_id"])
                    sub_target = target_norm[target_norm["entity_id"].isin(b_needed_targets)]
                    b_target_records = sub_target.set_index("entity_id").to_dict(orient="index")
                    del sub_target

                    b_s1_records = batch_s1.set_index("entity_id").to_dict(orient="index")
                    b_feats_df = PairFeatureExtractor.build_features(
                        b_pairs_df,
                        s1_records=b_s1_records,
                        cand_records=b_target_records,
                        verbose=False,
                        n_jobs=self.config.n_jobs,
                    )
                    b_scores = self.pair_scorer.predict(b_feats_df, use_calibrated=False)
                    b_feats_df["raw_score"] = b_scores
                    _, b_cands_by_s1 = EntityAggregator.aggregate_entity_candidates(
                        b_feats_df, batch_s1_ids, score_col="raw_score"
                    )
                    b_preds = EntityAggregator.apply_policy(
                        b_cands_by_s1, active_policy, batch_s1_ids, s1_country_map=s1_country_map
                    )
                    b_match_tsv = EntityAggregator.to_matching_results_tsv(b_preds, all_s1_ids=batch_s1_ids)

                    del b_target_records, b_s1_records, b_feats_df, b_scores, b_cands_by_s1, b_preds

                if matching_out is not None and candidates_out is not None:
                    mode = "a" if b_idx > 0 else "w"
                    header = (b_idx == 0)
                    b_cand_tsv.to_csv(candidates_out, sep="\t", index=False, mode=mode, header=header, encoding="utf-8")
                    b_match_tsv.to_csv(matching_out, sep="\t", index=False, mode=mode, header=header, encoding="utf-8")
                    total_matches_saved += len(b_match_tsv)
                    total_cands_saved += len(b_cand_tsv)
                else:
                    all_candidate_tsv_parts.append(b_cand_tsv)
                    all_matching_tsv_parts.append(b_match_tsv)

                del b_cands_raw, b_cand_tsv, b_pairs_df, b_match_tsv
                gc.collect()

            if matching_out is not None and candidates_out is not None:
                # Proxies with metadata so caller does not reload 1.7M rows into RAM
                matching_results_df = pd.DataFrame({"source1_entity_id": [f"saved_{total_matches_saved}_rows"]})
                candidate_pairs_df = pd.DataFrame({"source1_entity_id": [f"saved_{total_cands_saved}_rows"]})
                matching_results_df._actual_len = total_matches_saved
                candidate_pairs_df._actual_len = total_cands_saved
            else:
                matching_results_df = pd.concat(all_matching_tsv_parts, ignore_index=True)
                candidate_pairs_df = pd.concat(all_candidate_tsv_parts, ignore_index=True)

            return matching_results_df, candidate_pairs_df
