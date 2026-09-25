"""Synthetic branch coverage for the frozen Phase 10 decision function."""

import json
import tempfile
import unittest
from pathlib import Path

from src.threshold_policy import (
    FrozenDecisionConfig, load_frozen_config, predict_match_set,
    predict_policy, save_frozen_config,
)


class ThresholdPolicyTests(unittest.TestCase):
    def predict(self, ids, scores, pair=0.8, entity=0.5, gap=0.2):
        return predict_match_set(ids, scores, pair, entity, gap)

    def test_zero_candidates(self):
        self.assertEqual(self.predict([], []), [])

    def test_entity_gate_below_and_equal(self):
        self.assertEqual(self.predict(["a"], [0.49]), [])
        self.assertEqual(self.predict(["a"], [0.5]), ["a"])

    def test_one_candidate_has_infinite_decision_gap(self):
        self.assertEqual(self.predict(["a"], [0.6], pair=0.99, gap=1), ["a"])

    def test_gap_equal_selects_top_one(self):
        self.assertEqual(self.predict(["b", "a"], [0.9, 0.7], gap=0.2), ["b"])

    def test_tied_top_scores_have_zero_gap(self):
        self.assertEqual(self.predict(["b", "a"], [0.9, 0.9], gap=0.1), ["a", "b"])
        self.assertEqual(self.predict(["b", "a"], [0.9, 0.9], gap=0), ["a"])

    def test_multi_branch_pair_gate_equal_and_all_matches(self):
        self.assertEqual(self.predict(["c", "b", "a"], [0.79, 0.8, 0.85]), ["a", "b"])

    def test_multi_branch_top_one_fallback(self):
        self.assertEqual(self.predict(["b", "a"], [0.7, 0.65]), ["b"])

    def test_duplicates_and_score_tie_order_are_deterministic(self):
        self.assertEqual(self.predict(["b", "a", "b"], [0.9, 0.9, 0.6], gap=0.1),
                         ["a", "b"])

    def test_baselines_and_threshold_independence(self):
        ids, scores = ["a", "b"], [0.7, 0.68]
        self.assertEqual(predict_policy("A", ids, scores, 0.8), [])
        self.assertEqual(predict_policy("B", ids, scores, 0.8, 0.7), ["a"])
        self.assertEqual(predict_policy("B", ids, scores, 0.1, 0.8), [])
        self.assertEqual(predict_policy("C", ids, scores, 0.1, 0.7, 0.1), ids)
        self.assertEqual(predict_policy("C", ids, scores, 0.1, 0.7, 0), ["a"])

    def test_invalid_inputs(self):
        for ids, scores in [(["a"], []), (["a"], [float("nan")]), ([""], [0.5])]:
            with self.assertRaises(ValueError):
                self.predict(ids, scores)
        with self.assertRaises(ValueError):
            self.predict(["a"], [0.5], gap=-0.1)

    def test_frozen_config_roundtrip_and_no_overwrite(self):
        config = FrozenDecisionConfig("C", 0.8, 0.5, 0.2, "raw")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.json"
            save_frozen_config(path, config)
            self.assertEqual(load_frozen_config(path), config)
            self.assertEqual(load_frozen_config(path).predict(["a"], [0.6]), ["a"])
            with self.assertRaises(FileExistsError):
                save_frozen_config(path, config)
            payload = json.loads(path.read_text())
            payload["extra"] = 1
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_frozen_config(path)


if __name__ == "__main__":
    unittest.main()
