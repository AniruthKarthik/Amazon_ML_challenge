"""Unit and integration tests for evaluation metrics and retrieval benchmark (Phase 4)."""

import pytest
from chimera_submission.code.business_entity_resolution.src.evaluation import (
    MetricsEvaluator,
    RetrievalBenchmarkReport,
)
from chimera_submission.code.business_entity_resolution.src.retrieval import CandidateProvenance


def test_single_entity_f05_competition_example():
    # Example from competition README:
    # GT: [S2-00047, S3-00812]
    # Pred: [S2-00047, S2-00193, S3-00812]
    # Expected F_0.5 = 0.7142857...
    truth = {"S2-00047", "S3-00812"}
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    score = MetricsEvaluator.calculate_single_entity_f05(pred, truth)
    assert score == pytest.approx(0.7142857, abs=1e-4)


def test_single_entity_f05_singletons():
    # Empty truth: predicting empty -> 1.0
    assert MetricsEvaluator.calculate_single_entity_f05(set(), set()) == 1.0
    # Empty truth: predicting any candidate -> 0.0 (penalizing false merge)
    assert MetricsEvaluator.calculate_single_entity_f05({"S2-001"}, set()) == 0.0


def test_macro_f05_mixed():
    ground_truth = {
        "S1-1": {"S2-1"},       # perfect match -> 1.0
        "S1-2": {"S2-2"},       # missed match -> 0.0
        "S1-3": set(),          # singleton correct -> 1.0
        "S1-4": set(),          # singleton false positive -> 0.0
    }
    predictions = {
        "S1-1": {"S2-1"},
        "S1-2": set(),
        "S1-3": set(),
        "S1-4": {"S2-3"},
    }
    all_s1 = {"S1-1", "S1-2", "S1-3", "S1-4"}
    macro = MetricsEvaluator.compute_macro_f05(predictions, ground_truth, all_s1)
    # Average of 1.0, 0.0, 1.0, 0.0 = 0.5
    assert macro == pytest.approx(0.5)


def test_oracle_f05():
    ground_truth = {
        "S1-1": {"S2-1", "S2-2"},
        "S1-2": set(),
    }
    # Candidate set only retrieves S2-1 for S1-1 (retrieves 1 of 2 GTs)
    candidates = {
        "S1-1": {"S2-1", "S3-99"},  # S3-99 is false candidate
        "S1-2": {"S3-99"},          # false candidate for singleton
    }
    all_s1 = {"S1-1", "S1-2"}

    # Oracle retains S2-1 for S1-1 (Recall = 0.5, Precision = 1.0)
    # S1-1 Oracle F_0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
    # S1-2 Oracle retains nothing for singleton -> 1.0
    # Mean = (0.8333 + 1.0) / 2 = 0.91666...
    oracle = MetricsEvaluator.compute_oracle_f05(candidates, ground_truth, all_s1)
    expected_s1 = (1.25 * 1.0 * 0.5) / (0.25 + 0.5)
    expected_mean = (expected_s1 + 1.0) / 2.0
    assert oracle == pytest.approx(expected_mean, abs=1e-4)


def test_benchmark_retrieval_and_diversity():
    ground_truth = {
        "S1-1": {"S2-1", "S2-2"},
        "S1-2": {"S3-1"},
        "S1-3": set(),  # singleton
    }
    all_s1 = {"S1-1", "S1-2", "S1-3"}

    # Mock retrieval results with provenance
    prov_1_1 = CandidateProvenance(s1_id="S1-1", target_id="S2-1")
    prov_1_1.channels.add("exact_name")
    prov_1_1.ranks["exact_name"] = 1

    prov_1_2 = CandidateProvenance(s1_id="S1-1", target_id="S2-2")
    prov_1_2.channels.add("tfidf_name")
    prov_1_2.ranks["tfidf_name"] = 5

    # S1-2's match S3-1 is missed completely (not retrieved)
    retrieval_results = {
        "S1-1": {"S2-1": prov_1_1, "S2-2": prov_1_2},
        "S1-2": {},
        "S1-3": {},
    }

    report = MetricsEvaluator.benchmark_retrieval(
        retrieval_results,
        ground_truth,
        all_s1,
        s1_names={"S1-2": "Tata Motors"},
        target_names={"S3-1": "Motors Tata"},
    )

    assert report.total_s1_entities == 3
    assert report.total_gt_pairs == 3
    assert report.retrieved_gt_pairs == 2
    assert report.micro_recall == pytest.approx(2 / 3)
    assert report.retrieval_miss_rate == pytest.approx(1 / 3)

    # Check unique GT recovered per channel
    exact_stat = next(s for s in report.channel_stats if s.channel_name == "exact_name")
    tfidf_stat = next(s for s in report.channel_stats if s.channel_name == "tfidf_name")
    assert exact_stat.unique_gt_recovered == 1
    assert tfidf_stat.unique_gt_recovered == 1

    # Check error taxonomy: "Tata Motors" vs "Motors Tata" has same tokens -> token_reordering
    assert report.error_taxonomy_counts["token_reordering"] == 1
