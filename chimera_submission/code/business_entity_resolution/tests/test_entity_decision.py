"""Phase 9 exact entity metric and score-distribution tests."""

import math
import unittest

from src.entity_decision import (
    compare_oof_score_paths, entity_f05, macro_f05,
    summarize_entity_scores,
)


class EntityDecisionTests(unittest.TestCase):
    def test_exact_competition_f05_and_singletons(self):
        truth = frozenset({"S2-a", "S3-a"})
        prediction = frozenset({"S2-a", "S3-a", "S2-x"})
        self.assertAlmostEqual(entity_f05(truth, prediction), 2.5 / 3.5)
        self.assertEqual(entity_f05(frozenset(), frozenset()), 1.0)
        self.assertEqual(entity_f05(frozenset(), frozenset({"S2-x"})), 0.0)
        self.assertEqual(entity_f05(truth, frozenset()), 0.0)
        self.assertEqual(entity_f05(truth, frozenset({"S2-x"})), 0.0)

    def test_macro_includes_missing_predictions_as_empty(self):
        truth = {
            "S1-a": frozenset({"S2-a", "S3-a"}),
            "S1-b": frozenset(),
            "S1-c": frozenset({"S2-c"}),
        }
        predictions = {"S1-a": frozenset({"S2-a"})}
        expected = (1.25 / 1.5 + 1 + 0) / 3
        self.assertAlmostEqual(macro_f05(truth, predictions), expected)
        with self.assertRaises(ValueError):
            macro_f05({}, {})
        with self.assertRaises(ValueError):
            macro_f05(truth, {"S1-unknown": frozenset()})

    def test_score_summary_covers_distribution_and_ties(self):
        summary = summarize_entity_scores(
            "S1-a", [("S2-b", 0.8), ("S2-a", 0.8), ("S3-c", 0.2)],
            threshold=0.5,
        )
        self.assertEqual(summary.candidate_count, 3)
        self.assertEqual(summary.top1_target_id, "S2-a")
        self.assertEqual(summary.max_score, 0.8)
        self.assertEqual(summary.second_score, 0.8)
        self.assertEqual(summary.score_gap, 0.0)
        self.assertEqual(summary.count_above_threshold, 2)
        self.assertAlmostEqual(summary.mean_score, 0.6)
        self.assertAlmostEqual(summary.score_std, math.sqrt(0.08))
        self.assertAlmostEqual(summary.score_concentration, 0.8 / 1.8)
        empty = summarize_entity_scores("S1-b", [])
        self.assertEqual(empty.candidate_count, 0)
        self.assertIsNone(empty.top1_target_id)
        self.assertEqual(empty.max_score, 0.0)
        with self.assertRaises(ValueError):
            summarize_entity_scores("S1-a", [("S2-a", 0.2), ("S2-a", 0.3)])
        with self.assertRaises(ValueError):
            summarize_entity_scores("S1-a", [("S2-a", float("nan"))])

    def test_compare_raw_and_calibrated_oof_paths_without_selection(self):
        report = compare_oof_score_paths(
            ["S1-a", "S1-a", "S1-b"],
            ["S2-a", "S3-a", "S2-x"],
            [0.8, 0.3, 0.7],
            [0.6, 0.7, 0.2],
            {
                "S1-a": frozenset({"S2-a", "S3-a"}),
                "S1-b": frozenset(),
                "S1-c": frozenset({"S2-c"}),
            },
            threshold=0.5,
        )
        self.assertEqual(report["entities"], 3)
        self.assertAlmostEqual(report["raw_macro_f05"], (1.25 / 1.5) / 3)
        self.assertAlmostEqual(report["calibrated_macro_f05"], 2 / 3)
        self.assertNotIn("keep_calibration", report)

    def test_compare_rejects_unsorted_unknown_and_duplicate_pairs(self):
        truth = {"S1-a": frozenset(), "S1-b": frozenset()}
        with self.assertRaises(ValueError):
            compare_oof_score_paths(
                ["S1-b", "S1-a"], ["S2-b", "S2-a"],
                [0.1, 0.2], [0.1, 0.2], truth,
            )
        with self.assertRaises(ValueError):
            compare_oof_score_paths(
                ["S1-a", "S1-a"], ["S2-a", "S2-a"],
                [0.1, 0.2], [0.1, 0.2], truth,
            )
        with self.assertRaises(ValueError):
            compare_oof_score_paths(
                ["S1-z"], ["S2-a"], [0.1], [0.1], truth,
            )


if __name__ == "__main__":
    unittest.main()
