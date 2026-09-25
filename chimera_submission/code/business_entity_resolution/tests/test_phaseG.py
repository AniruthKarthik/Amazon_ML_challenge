"""Phase G tests (stdlib only)."""

import unittest

from src.entity_metrics import cardinality_bucket, per_cardinality_f05, prediction_histogram
from src.threshold_policy import fine_threshold_grid, predict_match_set_nofallback
from src.threshold_search import OOFEntity


class PhaseGTests(unittest.TestCase):
    def test_nofallback(self):
        # Gate passes but nothing passes pair threshold -> empty (no coercion).
        self.assertEqual(
            predict_match_set_nofallback(["b", "a"], [0.7, 0.65],
                                         pair_threshold=0.8, entity_threshold=0.5),
            [],
        )
        self.assertEqual(
            predict_match_set_nofallback(["b", "a"], [0.9, 0.85],
                                         pair_threshold=0.8, entity_threshold=0.5),
            ["b", "a"],
        )
        self.assertEqual(
            predict_match_set_nofallback([], [], pair_threshold=0.5, entity_threshold=0.5),
            [],
        )
        self.assertEqual(
            predict_match_set_nofallback(["b", "a"], [0.4, 0.3],
                                         pair_threshold=0.5, entity_threshold=0.5),
            [],
        )

    def test_max_matches_cap(self):
        ids = ["a", "b", "c"]
        scores = [0.9, 0.85, 0.84]
        self.assertEqual(
            predict_match_set_nofallback(ids, scores, 0.5, 0.5, max_matches=2),
            ["a", "b"],
        )
        with self.assertRaises(ValueError):
            predict_match_set_nofallback(ids, scores, 0.5, 0.5, max_matches=0)

    def test_fine_grid(self):
        self.assertEqual(fine_threshold_grid(0.5, 0.5, 1), (0.5,))
        grid = fine_threshold_grid(0.0, 1.0, 5)
        self.assertEqual(grid, (0.0, 0.25, 0.5, 0.75, 1.0))
        with self.assertRaises(ValueError):
            fine_threshold_grid(0.8, 0.2, 3)

    def test_per_cardinality(self):
        entities = [
            OOFEntity("S1-a", ("S2-a",), (0.9,), frozenset(), 0, "US"),
            OOFEntity("S1-b", ("S2-b", "S2-c"), (0.9, 0.8),
                      frozenset({"S2-b", "S2-c"}), 0, "US"),
        ]
        preds = {"S1-a": [], "S1-b": ["S2-b", "S2-c"]}
        report = per_cardinality_f05(entities, preds)
        self.assertAlmostEqual(report["0"]["macro_f05"], 1.0)
        self.assertAlmostEqual(report["2-3"]["macro_f05"], 1.0)
        hist = prediction_histogram(preds)
        self.assertEqual(hist["0"], 1)
        self.assertEqual(hist["2-3"], 1)

    def test_buckets(self):
        self.assertEqual(cardinality_bucket(0), "0")
        self.assertEqual(cardinality_bucket(4), "4+")


if __name__ == "__main__":
    unittest.main()
