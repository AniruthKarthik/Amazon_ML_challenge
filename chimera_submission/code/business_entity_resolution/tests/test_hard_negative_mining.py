"""Small synthetic Phase 11 checks, without project-data training."""

import unittest

import numpy as np

from src.hard_negative_mining import (
    HardNegativeConfig, compare_raw_oof_hard_negatives,
    generate_hard_negative_oof_predictions,
    mine_hard_negatives,
)
from src.pair_model import BaselineConfig, assign_component_folds, source_component_ids
from src.threshold_search import ThresholdGrid


class HardNegativeMiningTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(37)
        self.sources = [f"S1-{index}" for index in range(9)]
        components = tuple((source, f"S2-{index}")
                           for index, source in enumerate(self.sources))
        self.fold_map = assign_component_folds(components, self.sources, 3)
        self.component_map = source_component_ids(components, self.sources)
        self.row_sources = [source for source in self.sources for _ in range(8)]
        self.row_targets = [f"S2-candidate-{index}" for index in range(72)]
        self.features = random.normal(size=(72, 3)).astype(np.float32)
        self.labels = (self.features[:, 0] + self.features[:, 1] > 0).astype(int)
        self.names = ("name", "address", "country")
        self.model_config = BaselineConfig(
            num_boost_round=12, early_stopping_rounds=3,
            num_leaves=7, min_data_in_leaf=2, num_threads=1,
        )
        self.mining_config = HardNegativeConfig(0.0, 2, 1.0)

    def run_oof(self, labels):
        return generate_hard_negative_oof_predictions(
            self.features, labels, self.row_sources, self.row_targets,
            self.fold_map, self.component_map, self.names,
            self.mining_config, self.model_config,
        )

    def test_miner_selects_negatives_with_cutoff_cap_and_stable_ties(self):
        selected = mine_hard_negatives(
            ["s2", "s1", "s1", "s1", "s2"],
            ["a", "b", "a", "c", "b"],
            [0, 0, 0, 1, 0],
            [0.8, 0.7, 0.7, 0.99, 0.79],
            HardNegativeConfig(0.7, 1, 1.0),
        )
        self.assertEqual(selected, (2, 0))
        with self.assertRaises(ValueError):
            mine_hard_negatives(["s", "s"], ["t", "t"], [0, 0], [0.9, 0.8],
                                self.mining_config)
        with self.assertRaises(ValueError):
            mine_hard_negatives(["s"], ["t"], [0], [float("nan")],
                                self.mining_config)

    def test_nested_oof_only_mines_outer_training_negatives(self):
        result = self.run_oof(self.labels)
        self.assertEqual(result.raw_scores.shape, (72,))
        self.assertTrue(np.isfinite(result.raw_scores).all())
        self.assertEqual(set(result.mined_indices_by_fold), {0, 1, 2})
        self.assertTrue(any(result.mined_indices_by_fold.values()))
        for fold, indices in result.mined_indices_by_fold.items():
            self.assertLessEqual(len(indices), 2 * 6)  # six eligible sources
            self.assertTrue(all(self.labels[index] == 0 for index in indices))
            self.assertTrue(all(result.fold_ids[index] != fold for index in indices))
            self.assertEqual(result.fold_diagnostics[fold]["mined_negatives"], len(indices))

    def test_heldout_labels_cannot_change_its_mining_or_scores(self):
        first = self.run_oof(self.labels)
        heldout = np.asarray([self.fold_map[source] == 0 for source in self.row_sources])
        changed_labels = self.labels.copy()
        changed_labels[heldout] = 1 - changed_labels[heldout]
        second = self.run_oof(changed_labels)
        self.assertEqual(first.mined_indices_by_fold[0], second.mined_indices_by_fold[0])
        np.testing.assert_array_equal(first.raw_scores[heldout], second.raw_scores[heldout])

    def test_invalid_config_and_component_crossing(self):
        for values in ((-0.1, 2, 1), (0.5, 0, 1), (0.5, 2, 0)):
            with self.assertRaises(ValueError):
                HardNegativeConfig(*values)
        bad_components = dict(self.component_map)
        other = next(source for source in self.sources
                     if self.fold_map[source] != self.fold_map[self.sources[0]])
        bad_components[other] = bad_components[self.sources[0]]
        with self.assertRaisesRegex(ValueError, "crosses validation folds"):
            generate_hard_negative_oof_predictions(
                self.features, self.labels, self.row_sources, self.row_targets,
                self.fold_map, bad_components, self.names,
                self.mining_config, self.model_config,
            )
        duplicated_targets = list(self.row_targets)
        duplicated_targets[1] = duplicated_targets[0]
        with self.assertRaisesRegex(ValueError, "unique, nonempty IDs"):
            generate_hard_negative_oof_predictions(
                self.features, self.labels, self.row_sources,
                duplicated_targets, self.fold_map, self.component_map,
                self.names, self.mining_config, self.model_config,
            )

    def test_raw_oof_acceptance_comparison_reoptimizes_both_paths(self):
        comparison = compare_raw_oof_hard_negatives(
            ["s1", "s2"], ["a", "a"], [0, 1],
            [0.2, 0.2], [0.9, 0.1],
            {"s1": frozenset({"a"}), "s2": frozenset()},
            {"s1": 0, "s2": 1}, {"s1": "US", "s2": "IN"},
            ThresholdGrid((0.5,), (0.5,), (0.1,)),
        )
        self.assertTrue(comparison.improves_crossfit_macro)
        self.assertGreater(comparison.crossfit_macro_delta, 0)
        self.assertEqual(comparison.baseline.policies["A"].frozen_config.score_path,
                         "raw")
        with self.assertRaises(ValueError):
            compare_raw_oof_hard_negatives(
                ["s1", "s2"], ["a", "a"], [0, 1],
                [0.2, 0.2], [float("nan"), 0.1],
                {"s1": frozenset({"a"}), "s2": frozenset()},
                {"s1": 0, "s2": 1}, {"s1": "US", "s2": "IN"},
                ThresholdGrid((0.5,), (0.5,), (0.1,)),
            )


if __name__ == "__main__":
    unittest.main()
