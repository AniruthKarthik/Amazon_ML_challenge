#!/usr/bin/env python3
"""Main entry point for Business Entity Resolution pipeline.

Runs data loading, multi-view normalization, candidate retrieval, pair feature engineering,
LightGBM cross-fitting, threshold optimization, and deterministic frozen inference.
Outputs matching_results.tsv and candidate_pairs.tsv.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

from chimera_submission.code.business_entity_resolution.src.data_contract import (
    GroundTruthLoader,
    TSVLoader,
)
from chimera_submission.code.business_entity_resolution.src.pipeline import (
    BusinessEntityResolutionPipeline,
    PipelineConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Business Entity Resolution end-to-end pipeline."
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="dataset/train",
        help="Path to folder containing train_source1/2/3.tsv and train_ground_truth.tsv",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="dataset/test",
        help="Path to folder containing test_source1/2/3.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="chimera_submission/output",
        help="Path to directory where matching_results.tsv and candidate_pairs.tsv will be saved",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-train-queries",
        type=int,
        default=40000,
        help="Maximum S1 training entities to use for model fitting and threshold optimization (default: 40000; set to 0 to use all)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of CPU cores to utilize (-1 for all cores, default: -1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume directly from model_checkpoint.pkl if available in output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "gpu"],
        help="Device to use for training (default: auto; auto-detects CUDA / GPU on Colab)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolved_device = args.device
    if resolved_device == "auto":
        try:
            import torch
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            resolved_device = "cpu"

    print("=" * 70)
    print("ML Challenge 2026: Business Entity Resolution Pipeline")
    print(f"  Train Directory:    {args.train_dir}")
    print(f"  Test Directory:     {args.test_dir}")
    print(f"  Output Directory:   {args.output_dir}")
    print(f"  Folds:              {args.k_folds}")
    print(f"  Seed:               {args.seed}")
    print(f"  Max Train Queries:  {args.max_train_queries if args.max_train_queries > 0 else 'unlimited'}")
    print(f"  Device:             {resolved_device.upper()} {'(NVIDIA T4 / CUDA)' if resolved_device in ['cuda', 'gpu'] else '(CPU Multi-Core)'}")
    print(f"  CPU Parallelism:    {'all available cores' if args.n_jobs == -1 else f'{args.n_jobs} cores'}")
    print("=" * 70)

    train_path = Path(args.train_dir)
    test_path = Path(args.test_dir)

    # Automatically resolve path if dataset was extracted into nested structure
    def _resolve_data_dir(base_dir: Path, probe_file: str) -> Path:
        if (base_dir / probe_file).is_file():
            return base_dir
        candidates = [
            base_dir / "train",
            base_dir / "test",
            base_dir / "student_resource" / "dataset" / "train",
            base_dir / "student_resource" / "dataset" / "test",
            Path("dataset/student_resource/dataset/train"),
            Path("dataset/student_resource/dataset/test"),
            Path("dataset/train"),
            Path("dataset/test"),
        ]
        for c in candidates:
            if (c / probe_file).is_file():
                return c
        return base_dir

    train_path = _resolve_data_dir(train_path, "train_source1.tsv")
    test_path = _resolve_data_dir(test_path, "test_source1.tsv")

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    checkpoint_file = out_path / "model_checkpoint.pkl"

    # 1. Load Training Data (skip if resuming from checkpoint)
    if args.resume and checkpoint_file.exists():
        print(f"\n[Step 1/5] Skipping training data loading (resuming from checkpoint {checkpoint_file})...")
        train_s1 = train_s2 = train_s3 = gt_df = None
    else:
        print("\n[Step 1/5] Loading and validating training data...")
        print(f"  Resolved Train Path: {train_path}")
        print(f"  Resolved Test Path:  {test_path}")

        train_s1, train_s2, train_s3, gt_df = TSVLoader.load_training_split(
            train_path, max_queries=args.max_train_queries, seed=args.seed
        )
        print(
            f"  Loaded Train S1: {len(train_s1)}, S2: {len(train_s2)}, S3: {len(train_s3)}, Ground Truth: {len(gt_df)}"
        )

    # 2. Fit Pipeline & Cross-Validate (or load from checkpoint)
    if args.resume and checkpoint_file.exists():
        print(f"\n[Step 2/5] Resuming from existing checkpoint: {checkpoint_file}")
        pipeline = BusinessEntityResolutionPipeline.load(checkpoint_file)
        pipeline.config.n_jobs = args.n_jobs
    else:
        print("\n[Step 2/5] Fitting pipeline, cross-fitting models & optimizing robust thresholds...")
        config = PipelineConfig(
            k_folds=args.k_folds,
            random_seed=args.seed,
            n_jobs=args.n_jobs,
            device=resolved_device,
        )
        pipeline = BusinessEntityResolutionPipeline(config=config)
        pipeline.fit(train_s1, train_s2, train_s3, gt_df)

        # Release training memory before test phase
        del train_s1, train_s2, train_s3, gt_df
        import gc
        gc.collect()

        # Checkpoint model and policy to disk immediately
        pipeline.save(checkpoint_file)
        print(f"  [Checkpoint Saved] Trained models and locked policy saved to: {checkpoint_file}")

    rep = pipeline.ablation_report
    print("\n  --- Cross-Validation Metrics & Locked Decision Policy ---")
    print(f"  OOF Pair AUC:             {rep.get('pair_auc', 0.0):.4f}")
    print(f"  OOF Pair PR-AUC:          {rep.get('pair_pr_auc', 0.0):.4f}")
    print(f"  Locked Robust Macro F0.5: {rep.get('best_macro_f05', 0.0):.4f}")
    print(f"  Locked Decision Policy:   {rep.get('locked_policy')}")

    # 3. Load Test Data
    print("\n[Step 3/5] Loading and validating test reference data...")
    test_s1_file = test_path / "test_source1.tsv"
    test_s2_file = test_path / "test_source2.tsv"
    test_s3_file = test_path / "test_source3.tsv"

    test_s1 = TSVLoader.load_source_tsv(test_s1_file, expected_prefix="S1")
    print(
        f"  Loaded Test S1: {len(test_s1)} entities across {test_s1['country'].nunique()} country partitions."
    )
    print(f"  Streaming target sources (S2 & S3) directly by country partition to keep RAM < 6GB.")

    # 4. Predict & Stream Directly to Output TSVs
    print("\n[Step 4/5] Executing frozen inference on test set (streaming country-by-country)...")
    matching_out = out_path / "matching_results.tsv"
    candidates_out = out_path / "candidate_pairs.tsv"

    matching_df, candidates_df = pipeline.predict(
        test_s1,
        test_s2_file,
        test_s3_file,
        matching_out=matching_out,
        candidates_out=candidates_out,
    )
    del test_s1
    gc.collect()

    # 5. Export Results
    print("\n[Step 5/5] Finalizing official submission TSVs...")
    actual_matches = getattr(matching_df, "_actual_len", len(matching_df))
    actual_candidates = getattr(candidates_df, "_actual_len", len(candidates_df))
    print(f"  Saved final matches to: {matching_out} ({actual_matches} rows)")
    print(f"  Saved candidate pairs to: {candidates_out} ({actual_candidates} rows)")

    print("\nPipeline execution completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
