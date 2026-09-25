"""Multi-positive OOF ranking diagnostics keep retrieval misses separate."""

import unittest

from src.oof_ranking import evaluate_oof_ranking


class OOFRankingTests(unittest.TestCase):
    def test_retrieved_true_pairs_and_multi_positive_entities(self):
        report = evaluate_oof_ranking(
            ["S1-a", "S1-a", "S1-a", "S1-b", "S1-b"],
            ["S2-x", "S2-a", "S3-a", "S2-b", "S3-x"],
            [0.9, 0.8, 0.7, 0.6, 0.4],
            {
                "S1-a": frozenset({"S2-a", "S3-a"}),
                "S1-b": frozenset({"S2-b"}),
                "S1-c": frozenset({"S2-c"}),
            },
        )
        self.assertEqual(report["ground_truth_pairs"], 4)
        self.assertEqual(report["retrieved_true_pairs"], 3)
        self.assertEqual(report["context_retrieval_misses"], 1)
        self.assertEqual(report["top1_pair_ranking_failures"], 2)
        self.assertAlmostEqual(report["top1_pair_ranking_failure_rate"], 2 / 3)
        self.assertAlmostEqual(report["retrieved_pair_recall_at_5"], 1.0)
        self.assertAlmostEqual(report["retrieved_pair_recall_at_10"], 1.0)
        self.assertEqual(report["entity_any_true_at_1"], 0.5)
        self.assertEqual(report["retrieved_true_pairs_with_false_positive_above"], 2)
        self.assertAlmostEqual(report["mean_false_positives_above_retrieved_true"], 2 / 3)

    def test_ties_are_deterministic_and_no_retrieved_true_is_not_ranked(self):
        report = evaluate_oof_ranking(
            ["S1-a", "S1-a"], ["S2-b", "S2-a"], [0.5, 0.5],
            {"S1-a": frozenset({"S2-a"}), "S1-b": frozenset({"S2-c"})},
        )
        self.assertEqual(report["top1_pair_ranking_failures"], 0)
        self.assertEqual(report["context_retrieval_misses"], 1)
        empty = evaluate_oof_ranking(
            ["S1-a"], ["S2-x"], [0.9],
            {"S1-a": frozenset({"S2-a"})},
        )
        self.assertIsNone(empty["top1_pair_ranking_failure_rate"])
        self.assertEqual(empty["context_retrieval_misses"], 1)

    def test_bad_scores_and_duplicate_pairs_are_rejected(self):
        truth = {"S1-a": frozenset({"S2-a"})}
        with self.assertRaises(ValueError):
            evaluate_oof_ranking(["S1-a"], ["S2-a"], [float("nan")], truth)
        with self.assertRaises(ValueError):
            evaluate_oof_ranking(
                ["S1-a", "S1-a"], ["S2-a", "S2-a"], [0.9, 0.8], truth
            )
        with self.assertRaises(ValueError):
            evaluate_oof_ranking(["S1-missing"], ["S2-a"], [0.8], truth)
        with self.assertRaisesRegex(ValueError, "sorted"):
            evaluate_oof_ranking(
                ["S1-b", "S1-a"], ["S2-b", "S2-a"], [0.9, 0.8],
                {"S1-a": frozenset(), "S1-b": frozenset()},
            )

    def test_recall_at_ten_excludes_true_pair_ranked_eleven(self):
        targets = [f"S2-false-{index:02d}" for index in range(10)] + ["S2-true"]
        report = evaluate_oof_ranking(
            ["S1-a"] * len(targets), targets,
            [1 - index * 0.05 for index in range(len(targets))],
            {"S1-a": frozenset({"S2-true"})},
        )
        self.assertEqual(report["retrieved_pair_recall_at_5"], 0.0)
        self.assertEqual(report["retrieved_pair_recall_at_10"], 0.0)
        self.assertEqual(report["mean_false_positives_above_retrieved_true"], 10.0)


if __name__ == "__main__":
    unittest.main()
