"""Unit and integration tests for data_contract.py."""

import io
import pytest
import pandas as pd
from chimera_submission.code.business_entity_resolution.src.data_contract import (
    TSVLoader,
    GroundTruthLoader,
    BipartiteGraphAnalyzer,
    DataIntegrityError,
    GraphMetrics,
)


def test_load_source_tsv_valid():
    tsv_content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tAcme Corp\t123 Main St, Springfield\tUS\n"
        "S1-00002\tCafé de la Gare\t\tFrance\n"
        "S1-00003\tSharma Enterprises\tNear SBI ATM, Mumbai\tIndia\n"
    )
    df = TSVLoader.load_source_tsv(io.StringIO(tsv_content), expected_prefix="S1")
    assert len(df) == 3
    assert list(df.columns) == ["entity_id", "business_name", "business_address", "country"]
    assert df.loc[1, "business_address"] == ""
    assert df.loc[1, "country"] == "France"
    assert df.loc[1, "business_name"] == "Café de la Gare"


def test_load_source_tsv_preserves_nas_as_empty_strings():
    tsv_content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-00001\tNA Traders\tNA\t\n"
        "S2-00002\tnull logistics\t\tIndia\n"
    )
    df = TSVLoader.load_source_tsv(io.StringIO(tsv_content), expected_prefix="S2")
    assert len(df) == 2
    # 'NA' should NOT be converted to NaN
    assert df.loc[0, "business_name"] == "NA Traders"
    assert df.loc[0, "business_address"] == "NA"
    assert df.loc[0, "country"] == ""
    assert df.loc[1, "business_name"] == "null logistics"
    assert df.loc[1, "business_address"] == ""


def test_load_source_tsv_missing_columns():
    tsv_content = (
        "entity_id\tbusiness_name\tcountry\n"
        "S1-00001\tAcme Corp\tUS\n"
    )
    with pytest.raises(DataIntegrityError, match="Missing required columns"):
        TSVLoader.load_source_tsv(io.StringIO(tsv_content))


def test_load_source_tsv_duplicate_ids():
    tsv_content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tAcme Corp\t123 Main St\tUS\n"
        "S1-00001\tAcme Corp 2\t456 Oak St\tUS\n"
    )
    with pytest.raises(DataIntegrityError, match="duplicate entity_id"):
        TSVLoader.load_source_tsv(io.StringIO(tsv_content))


def test_load_source_tsv_empty_business_name():
    tsv_content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\t\t123 Main St\tUS\n"
    )
    with pytest.raises(DataIntegrityError, match="missing/empty business_name"):
        TSVLoader.load_source_tsv(io.StringIO(tsv_content))


def test_load_source_tsv_invalid_prefix():
    tsv_content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-00001\tAcme Corp\t123 Main St\tUS\n"
    )
    with pytest.raises(DataIntegrityError, match="prefix 'S1-'"):
        TSVLoader.load_source_tsv(io.StringIO(tsv_content), expected_prefix="S1")


def test_load_ground_truth_valid():
    gt_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-00001\tS2-00047,S3-00812\n"
        "S1-00002\tS3-00004\n"
        "S1-00003\t\n"
    )
    df = GroundTruthLoader.load_ground_truth(io.StringIO(gt_content))
    assert len(df) == 3
    assert df.loc[0, "matched_entity_ids"] == "S2-00047,S3-00812"
    assert df.loc[2, "matched_entity_ids"] == ""


def test_load_ground_truth_self_match_rejected():
    gt_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-00001\tS1-00002\n"
    )
    with pytest.raises(DataIntegrityError, match="Self-match error"):
        GroundTruthLoader.load_ground_truth(io.StringIO(gt_content))


def test_load_ground_truth_duplicate_targets_rejected():
    gt_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-00001\tS2-00047,S2-00047\n"
    )
    with pytest.raises(DataIntegrityError, match="duplicate targets"):
        GroundTruthLoader.load_ground_truth(io.StringIO(gt_content))


def test_load_ground_truth_invalid_prefix():
    gt_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-00001\tX9-00047\n"
    )
    with pytest.raises(DataIntegrityError, match="Invalid target prefix"):
        GroundTruthLoader.load_ground_truth(io.StringIO(gt_content))


def test_load_ground_truth_coverage_validation():
    gt_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-00001\tS2-00047\n"
    )
    valid_s1 = {"S1-00001", "S1-00002"}
    with pytest.raises(DataIntegrityError, match="Source 1 entities missing from ground truth"):
        GroundTruthLoader.load_ground_truth(io.StringIO(gt_content), valid_s1_ids=valid_s1)


def test_bipartite_graph_analyzer_components_and_metrics():
    gt_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\tS2-1\n"
        "S1-3\tS3-2\n"
        "S1-4\t\n"
        "S1-5\t\n"
    )
    df = GroundTruthLoader.load_ground_truth(io.StringIO(gt_content))
    analyzer = BipartiteGraphAnalyzer(df)

    metrics = analyzer.get_metrics()
    assert metrics.num_s1_entities == 5
    assert metrics.num_unique_matched_targets == 3  # S2-1, S3-1, S3-2
    assert metrics.num_total_ground_truth_links == 4  # (S1-1, S2-1), (S1-1, S3-1), (S1-2, S2-1), (S1-3, S3-2)
    assert metrics.num_singletons == 2  # S1-4, S1-5
    assert metrics.singleton_ratio == pytest.approx(2 / 5)

    s1_comps = analyzer.get_s1_components()
    # S1-1 and S1-2 share S2-1, so they must be in the same component
    comp_1 = None
    comp_2 = None
    for root, s1_set in s1_comps.items():
        if "S1-1" in s1_set:
            comp_1 = s1_set
        if "S1-2" in s1_set:
            comp_2 = s1_set
    assert comp_1 is not None
    assert comp_1 == comp_2
    assert "S1-1" in comp_1 and "S1-2" in comp_1


def test_bipartite_graph_analyzer_leak_free_folds():
    gt_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1\n"
        "S1-2\tS2-1\n"
        "S1-3\tS3-1\n"
        "S1-4\tS3-1\n"
        "S1-5\tS2-2\n"
        "S1-6\t\n"
        "S1-7\t\n"
        "S1-8\t\n"
        "S1-9\t\n"
        "S1-10\t\n"
    )
    df = GroundTruthLoader.load_ground_truth(io.StringIO(gt_content))
    analyzer = BipartiteGraphAnalyzer(df)

    k_folds = 3
    folds = analyzer.create_leak_free_folds(k_folds=k_folds, random_seed=42)

    assert len(folds) == 10
    assert all(0 <= f < k_folds for f in folds.values())

    # CRITICAL LEAKAGE TEST:
    # S1-1 and S1-2 are connected through S2-1 -> MUST have identical fold
    assert folds["S1-1"] == folds["S1-2"]
    # S1-3 and S1-4 are connected through S3-1 -> MUST have identical fold
    assert folds["S1-3"] == folds["S1-4"]

    # Folds must be deterministic with same seed
    folds_repeat = analyzer.create_leak_free_folds(k_folds=k_folds, random_seed=42)
    assert folds == folds_repeat


def test_load_training_split_and_filtered_tsv(tmp_path):
    # Create synthetic directory structure
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    s1_lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry\n"]
    gt_lines = ["source1_entity_id\tmatched_entity_ids\n"]
    for i in range(50):
        s1_lines.append(f"S1-{i}\tAcme Corp {i}\t123 Main St\tUS\n")
        if i < 10:
            gt_lines.append(f"S1-{i}\tS2-{i},S3-{i}\n")
        elif i < 30:
            gt_lines.append(f"S1-{i}\tS2-{i}\n")
        else:
            gt_lines.append(f"S1-{i}\t\n")

    s2_lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry\n"]
    s3_lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry\n"]
    for i in range(100):
        s2_lines.append(f"S2-{i}\tAcme Corp {i}\t123 Main St Suite {i}\tUS\n")
        s3_lines.append(f"S3-{i}\tAcme Corp {i}\t123 Main St Fl {i}\tUS\n")

    (train_dir / "train_source1.tsv").write_text("".join(s1_lines), encoding="utf-8")
    (train_dir / "train_source2.tsv").write_text("".join(s2_lines), encoding="utf-8")
    (train_dir / "train_source3.tsv").write_text("".join(s3_lines), encoding="utf-8")
    (train_dir / "train_ground_truth.tsv").write_text("".join(gt_lines), encoding="utf-8")

    # Test load_training_split with max_queries constraint
    s1, s2, s3, gt = TSVLoader.load_training_split(train_dir, max_queries=20, seed=42)
    assert len(gt) == 20
    assert len(s1) == 20
    assert len(s2) > 0
    assert len(s3) > 0
    # Must preserve multi-match entities
    multi_count = sum(1 for m in gt["matched_entity_ids"] if "," in m)
    assert multi_count == 10  # All 10 multi-match entities retained

