"""Tiny Phase 7 checks; these are not project-data model-training runs."""

import unittest

import numpy as np

from src.pair_model import (
    BaselineConfig, assign_component_folds, train_pair_baseline,
)


class PairModelTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(17)
        self.train_x = random.normal(size=(80, 3)).astype(np.float32)
        self.valid_x = random.normal(size=(20, 3)).astype(np.float32)
        self.train_y = (self.train_x[:, 0] + self.train_x[:, 1] > 0).astype(int)
        self.valid_y = (self.valid_x[:, 0] + self.valid_x[:, 1] > 0).astype(int)
        self.train_groups = np.repeat(np.arange(10), 8)
        self.valid_groups = np.repeat(np.arange(10, 12), 10)
        self.names = ("name_levenshtein", "name_token_jaccard", "country_match")
        self.config = BaselineConfig(
            num_boost_round=30, early_stopping_rounds=5,
            num_leaves=7, min_data_in_leaf=2, num_threads=1,
        )

    def test_phase1_components_keep_linked_entities_in_one_fold(self):
        components = (
            ("S1-a", "S1-b", "S2-link"),
            ("S1-c", "S3-c"),
            ("S1-d",),
            ("S1-e", "S2-e"),
        )
        ids = ["S1-e", "S1-c", "S1-a", "S1-d", "S1-b"]
        folds = assign_component_folds(components, ids, n_splits=3)
        self.assertEqual(folds["S1-a"], folds["S1-b"])
        self.assertEqual(set(folds), set(ids))
        self.assertEqual(folds, assign_component_folds(components, reversed(ids), 3))
        self.assertEqual(len(set(folds.values())), 3)

    def test_invalid_component_folds_fail_closed(self):
        with self.assertRaises(ValueError):
            assign_component_folds((("S1-a",),), ["S1-a", "S1-b"], 2)
        with self.assertRaises(ValueError):
            assign_component_folds((("S1-a",), ("S1-a",)), ["S1-a"], 2)
        with self.assertRaises(ValueError):
            assign_component_folds((("S1-a", "S1-b"),), ["S1-a", "S1-b"], 2)
        with self.assertRaises(ValueError):
            assign_component_folds((("S2-a",),), ["S2-a"], 2)

    def test_tiny_lightgbm_fit_and_pair_diagnostics(self):
        result = train_pair_baseline(
            self.train_x, self.train_y, self.valid_x, self.valid_y,
            self.names, self.train_groups, self.valid_groups, self.config,
        )
        self.assertEqual(result.validation_scores.shape, (20,))
        self.assertTrue(np.isfinite(result.validation_scores).all())
        self.assertGreaterEqual(result.pair_diagnostics["roc_auc"], 0.5)
        self.assertEqual(result.pair_diagnostics["pairs"], 20)
        self.assertEqual(result.feature_names, self.names)
        self.assertIn("binary_logloss", result.evaluation_history["validation"])
        self.assertLessEqual(len(result.evaluation_history["validation"]["binary_logloss"]), 30)

    def test_single_class_validation_has_no_fabricated_auc(self):
        result = train_pair_baseline(
            self.train_x, self.train_y, self.valid_x,
            np.zeros(len(self.valid_x), dtype=int), self.names,
            self.train_groups, self.valid_groups, self.config,
        )
        self.assertIsNone(result.pair_diagnostics["roc_auc"])
        self.assertEqual(result.pair_diagnostics["recall_at_0_5"], 0.0)

    def test_training_rejects_leakage_and_invalid_arrays(self):
        with self.assertRaisesRegex(ValueError, "share a graph component"):
            train_pair_baseline(
                self.train_x, self.train_y, self.valid_x, self.valid_y,
                self.names, self.train_groups, np.zeros(20), self.config,
            )
        invalid_x = self.train_x.copy()
        invalid_x[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            train_pair_baseline(
                invalid_x, self.train_y, self.valid_x, self.valid_y,
                self.names, self.train_groups, self.valid_groups, self.config,
            )
        with self.assertRaisesRegex(ValueError, "both positive and negative"):
            train_pair_baseline(
                self.train_x, np.zeros(len(self.train_y)),
                self.valid_x, self.valid_y, self.names,
                self.train_groups, self.valid_groups, self.config,
            )


if __name__ == "__main__":
    unittest.main()
