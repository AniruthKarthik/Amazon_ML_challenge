"""End-to-end integration and submission validation tests (Phase 15 & 16)."""

import os
from pathlib import Path
import pandas as pd
import pytest
from utils.validate_submission import validate
from chimera_submission.code.business_entity_resolution.src.data_contract import (
    GroundTruthLoader,
    TSVLoader,
)
from chimera_submission.code.business_entity_resolution.src.pipeline import (
    BusinessEntityResolutionPipeline,
    PipelineConfig,
)


def test_end_to_end_pipeline_and_official_validator(tmp_path):
    # Setup synthetic training files
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    out_dir = tmp_path / "output"
    train_dir.mkdir()
    test_dir.mkdir()
    out_dir.mkdir()

    # Train files
    train_s1 = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-001\tMicrosoft Corporation\tOne Microsoft Way, Redmond\tUS\n"
        "S1-002\tTata Consultancy Services\tBKC, Mumbai\tIndia\n"
        "S1-003\tCafé de la Gare\t10 Rue de Paris, Lyon\tFrance\n"
        "S1-004\tUnmatched Singleton A\t123 Nowhere St\tUS\n"
        "S1-005\tUnmatched Singleton B\t456 Empty Rd\tIndia\n"
        "S1-006\tGoogle LLC\tMountain View, CA\tUS\n"
    )
    (train_dir / "train_source1.tsv").write_text(train_s1, encoding="utf-8")

    train_s2 = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-001\tMicrosoft Corp\tOne Microsoft Way\tUS\n"
        "S2-002\tTata Consultancy Services Ltd\tBandra Kurla Complex\tIndia\n"
        "S2-003\tGoogle Inc\t1600 Amphitheatre Pkwy\tUS\n"
    )
    (train_dir / "train_source2.tsv").write_text(train_s2, encoding="utf-8")

    train_s3 = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-001\tMicrosoft\tRedmond, WA\tUS\n"
        "S3-002\tCafe de la Gare\tLyon\tFrance\n"
    )
    (train_dir / "train_source3.tsv").write_text(train_s3, encoding="utf-8")

    train_gt = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-001,S3-001\n"
        "S1-002\tS2-002\n"
        "S1-003\tS3-002\n"
        "S1-004\t\n"
        "S1-005\t\n"
        "S1-006\tS2-003\n"
    )
    (train_dir / "train_ground_truth.tsv").write_text(train_gt, encoding="utf-8")

    # Test files (including France which is test-only in the competition)
    test_s1 = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-901\tMicrosoft Corporation\tOne Microsoft Way\tUS\n"
        "S1-902\tBoulangerie du Coin\tParis\tFrance\n"
        "S1-903\tReliance Retail Limited\tMumbai\tIndia\n"
    )
    (test_dir / "test_source1.tsv").write_text(test_s1, encoding="utf-8")

    test_s2 = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-901\tMicrosoft Corp.\tRedmond WA\tUS\n"
        "S2-902\tReliance Retail Ltd\tMumbai, Maharashtra\tIndia\n"
    )
    (test_dir / "test_source2.tsv").write_text(test_s2, encoding="utf-8")

    test_s3 = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-901\tBoulangerie du Coin SARL\tParis, France\tFrance\n"
    )
    (test_dir / "test_source3.tsv").write_text(test_s3, encoding="utf-8")

    # Load via TSVLoader
    df_train_s1 = TSVLoader.load_source_tsv(train_dir / "train_source1.tsv", expected_prefix="S1")
    df_train_s2 = TSVLoader.load_source_tsv(train_dir / "train_source2.tsv", expected_prefix="S2")
    df_train_s3 = TSVLoader.load_source_tsv(train_dir / "train_source3.tsv", expected_prefix="S3")
    df_train_gt = GroundTruthLoader.load_ground_truth(train_dir / "train_ground_truth.tsv")

    df_test_s1 = TSVLoader.load_source_tsv(test_dir / "test_source1.tsv", expected_prefix="S1")
    df_test_s2 = TSVLoader.load_source_tsv(test_dir / "test_source2.tsv", expected_prefix="S2")
    df_test_s3 = TSVLoader.load_source_tsv(test_dir / "test_source3.tsv", expected_prefix="S3")

    # Fit pipeline
    config = PipelineConfig(
        k_folds=2,
        random_seed=42,
        lgb_n_estimators=10,
        lgb_max_depth=3,
        lgb_num_leaves=7,
    )
    pipeline = BusinessEntityResolutionPipeline(config=config)
    pipeline.fit(df_train_s1, df_train_s2, df_train_s3, df_train_gt)

    assert pipeline.is_fitted
    assert pipeline.locked_policy is not None

    # Predict
    matching_df, candidate_df = pipeline.predict(df_test_s1, df_test_s2, df_test_s3)

    matching_file = out_dir / "matching_results.tsv"
    candidate_file = out_dir / "candidate_pairs.tsv"

    matching_df.to_csv(matching_file, sep="\t", index=False, encoding="utf-8")
    candidate_df.to_csv(candidate_file, sep="\t", index=False, encoding="utf-8")

    # Check official validator!
    errors, warnings = validate(
        matching_path=str(matching_file),
        candidate_path=str(candidate_file),
        test_dir=str(test_dir),
        check_ids=True,
    )

    assert len(errors) == 0, f"Validator found errors: {errors}"
