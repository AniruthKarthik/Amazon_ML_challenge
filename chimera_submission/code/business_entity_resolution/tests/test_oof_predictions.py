"""Small cross-fitting checks without project-data training."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.oof_predictions import generate_oof_predictions, write_oof_artifact
from src.pair_model import BaselineConfig, assign_component_folds, source_component_ids


class OOFTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(23)
        self.sources = [f"S1-{index}" for index in range(9)]
        self.components = tuple((identifier, f"S2-{index}")
                                for index, identifier in enumerate(self.sources))
        self.fold_map = assign_component_folds(self.components, self.sources, 3)
        self.component_map = source_component_ids(self.components, self.sources)
        self.row_sources = [identifier for identifier in self.sources for _ in range(10)]
        self.features = random.normal(size=(90, 3)).astype(np.float32)
        self.labels = (self.features[:, 0] + self.features[:, 1] > 0).astype(int)
        self.names = ("name_similarity", "address_similarity", "country_match")
        self.config = BaselineConfig(
            num_boost_round=20, early_stopping_rounds=4,
            num_leaves=7, min_data_in_leaf=2, num_threads=1,
        )

    def test_raw_and_calibrated_scores_are_cross_fitted(self):
        result = generate_oof_predictions(
            self.features, self.labels, self.row_sources,
            self.fold_map, self.component_map, self.names, self.config,
        )
        self.assertEqual(len(result.raw_scores), 90)
        self.assertEqual(len(result.calibrated_scores), 90)
        self.assertEqual(len(result.fold_models), 3)
        self.assertEqual(len(result.calibration_models), 3)
        self.assertTrue(np.isfinite(result.raw_scores).all())
        self.assertTrue(np.isfinite(result.calibrated_scores).all())
        self.assertTrue(np.all((0 <= result.calibrated_scores) &
                               (result.calibrated_scores <= 1)))
        self.assertEqual(set(result.pair_f05_at_0_5),
                         {"raw_at_0_5", "crossfit_calibrated_at_0_5"})
        for source_id, fold in zip(self.row_sources, result.fold_ids):
            self.assertEqual(fold, self.fold_map[source_id])

    def test_calibration_excludes_held_out_labels(self):
        held_out = np.asarray([
            self.fold_map[source] == 0 for source in self.row_sources
        ])
        first = generate_oof_predictions(
            self.features, self.labels, self.row_sources,
            self.fold_map, self.component_map, self.names, self.config,
        )
        changed = self.labels.copy()
        changed[held_out] = 1 - changed[held_out]
        second = generate_oof_predictions(
            self.features, changed, self.row_sources,
            self.fold_map, self.component_map, self.names, self.config,
        )
        np.testing.assert_array_equal(
            first.raw_scores[held_out], second.raw_scores[held_out]
        )
        np.testing.assert_array_equal(
            first.calibrated_scores[held_out], second.calibrated_scores[held_out]
        )

    def test_split_component_and_bad_input_are_rejected(self):
        bad_folds = dict(self.fold_map)
        bad_components = dict(self.component_map)
        bad_components[self.sources[1]] = bad_components[self.sources[0]]
        if bad_folds[self.sources[1]] == bad_folds[self.sources[0]]:
            different = next(source for source in self.sources
                             if bad_folds[source] != bad_folds[self.sources[0]])
            bad_components[different] = bad_components[self.sources[0]]
        with self.assertRaisesRegex(ValueError, "crosses validation folds"):
            generate_oof_predictions(
                self.features, self.labels, self.row_sources,
                bad_folds, bad_components, self.names, self.config,
            )
        with self.assertRaisesRegex(ValueError, "must have a fold"):
            generate_oof_predictions(
                self.features, self.labels, self.row_sources,
                {}, self.component_map, self.names, self.config,
            )

    def test_oof_artifact_is_label_free_and_not_overwritten(self):
        result = generate_oof_predictions(
            self.features, self.labels, self.row_sources,
            self.fold_map, self.component_map, self.names, self.config,
        )
        target_ids = [f"S2-target-{index}" for index in range(len(self.row_sources))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oof.tsv"
            write_oof_artifact(path, self.row_sources, target_ids, result)
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 90)
            self.assertNotIn("label", rows[0])
            self.assertIn("raw_oof_score", rows[0])
            with self.assertRaises(FileExistsError):
                write_oof_artifact(path, self.row_sources, target_ids, result)


if __name__ == "__main__":
    unittest.main()
