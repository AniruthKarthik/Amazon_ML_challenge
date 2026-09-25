"""Small training-only integration checks; no project dataset is touched."""

import json
import tempfile
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.cpu_training import (
    CPUTrainingConfig, oof_threshold_grid, select_source_sequences,
    train_cpu_baseline,
)
from src.pair_model import BaselineConfig
from src.phase4_store import build_store, candidate_array, open_store
from src.pipeline_store import DiskCandidateStore
from src.retrieval import CHANNELS
from src.threshold_policy import load_frozen_config
from src.threshold_search import OOFEntity


class CPUTrainingTests(unittest.TestCase):
    def _fixture(self, root):
        for source in (1, 2, 3):
            path = root / f"train_source{source}.tsv"
            rows = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
            for index in range(9):
                name = (f"Shop {index}" if source != 3 else f"Other {index}")
                rows.append(f"S{source}-{index}\t{name}\tRoad {index}\tUS")
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        (root / "train_ground_truth.tsv").write_text(
            "source1_entity_id\tmatched_entity_ids\n" +
            "".join(f"S1-{index}\tS2-{index}\n" for index in range(9)),
            encoding="utf-8",
        )
        build_store(root, root / "store.sqlite")
        work = root / "retrieval"
        work.mkdir()
        for channel in CHANNELS:
            array = candidate_array(work / f"{channel}.int32", 9, 2, create=True)
            if channel == "exact_name":
                for index in range(9):
                    array[index, 0] = index
            elif channel == "char_name":
                for index in range(9):
                    array[index, 0] = (index + 1) % 9
            array.flush()
        for channel in ("char_name", "char_address"):
            scores = np.memmap(work / f"{channel}.float32", dtype=np.float32,
                               mode="w+", shape=(9, 2))
            scores[:] = -np.inf
            if channel == "char_name":
                scores[:, 0] = 0.4
            scores.flush()
        return work

    def test_sampling_and_oof_grid(self):
        self.assertEqual(select_source_sequences(5, 5, 7), (0, 1, 2, 3, 4))
        self.assertEqual(select_source_sequences(10, 3, 42),
                         select_source_sequences(10, 3, 42))
        with self.assertRaises(ValueError):
            select_source_sequences(5, 6, 42)
        entities = [
            OOFEntity("S1-a", ("S2-a", "S2-b"), (0.9, 0.8),
                      frozenset({"S2-a"}), 0, "US"),
            OOFEntity("S1-b", (), (), frozenset(), 1, "US"),
        ]
        grid = oof_threshold_grid(entities, 2)
        self.assertTrue(all(0 <= value <= 1 for value in grid.pair))
        self.assertEqual(len(grid.gap), 1)

    def test_tiny_end_to_end_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = self._fixture(root)
            connection = open_store(root / "store.sqlite")
            store = DiskCandidateStore(connection, work, 2, 2)
            config = CPUTrainingConfig(
                train_entities=9, max_train_pairs=100, folds=3,
                grid_points=2, feature_max_features=100,
                country_min_entities=2,
                model=BaselineConfig(num_boost_round=5,
                                     early_stopping_rounds=0,
                                     min_data_in_leaf=1, num_threads=1),
            )
            result = train_cpu_baseline(store, root / "model", config)
            self.assertEqual(result["sampled_source_entities"], 9)
            self.assertEqual(result["sampled_retrieved_positives"], 9)
            self.assertEqual(result["sampled_pair_rows"], 18)
            self.assertEqual(result["oof_retrieved_pair_ranking"]["retrieved_true_pairs"], 9)
            self.assertEqual(len(result["fold_pair_diagnostics"]), 3)
            self.assertTrue((root / "model" / "pair_model.txt").exists())
            self.assertEqual(lgb.Booster(model_file=str(
                root / "model" / "pair_model.txt")).num_feature(),
                len(store.channels) * 3 + 25)
            self.assertIn(load_frozen_config(
                root / "model" / "decision_config.json").policy,
                ("A", "B", "C"))
            self.assertEqual(json.loads((root / "model" / "threshold_search.json").read_text())
                             ["search"]["score_source"],
                             "leakage-safe training OOF pair predictions")
            with self.assertRaises(FileExistsError):
                train_cpu_baseline(store, root / "model", config)
            connection.close()

    def test_meta_comparison_is_explicit_and_can_be_rejected(self):
        with self.assertRaises(ValueError):
            CPUTrainingConfig(train_entities=9, evaluate_meta=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = self._fixture(root)
            connection = open_store(root / "store.sqlite")
            result = train_cpu_baseline(
                DiskCandidateStore(connection, work, 2, 2), root / "meta_model",
                CPUTrainingConfig(
                    train_entities=9, max_train_pairs=100, folds=3,
                    grid_points=2, feature_max_features=100,
                    country_min_entities=2, evaluate_meta=True,
                    max_worst_fold_drop=0, max_fold_std_increase=0,
                    model=BaselineConfig(num_boost_round=5,
                                         early_stopping_rounds=0,
                                         min_data_in_leaf=1, num_threads=1),
                ),
            )
            comparison = json.loads((root / "meta_model" /
                                     "meta_comparison.json").read_text())
            self.assertEqual(result["selected_entity_decision"],
                             "meta" if comparison["phase10_comparison"]["keep_meta"]
                             else "deterministic")
            self.assertEqual((root / "meta_model" / "meta_model.joblib").exists(),
                             comparison["phase10_comparison"]["keep_meta"])
            connection.close()


if __name__ == "__main__":
    unittest.main()
