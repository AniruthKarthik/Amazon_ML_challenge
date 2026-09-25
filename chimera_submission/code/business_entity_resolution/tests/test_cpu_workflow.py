"""One-command provisional workflow and safe reuse on tiny synthetic TSVs."""

import fcntl
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.cpu_training import CPUTrainingConfig
from src.cpu_workflow import CPUWorkflowConfig, run_cpu_workflow
from src.main import main
from src.pair_model import BaselineConfig
from tests.test_cpu_training import CPUTrainingTests


class CPUWorkflowTests(unittest.TestCase):
    def _config(self, root, *, allow=True):
        return CPUWorkflowConfig(
            train_dir=root / "train", test_dir=root / "test",
            work_root=root / "work", output_dir=root / "output",
            training=CPUTrainingConfig(
                train_entities=9, max_train_pairs=100, folds=3,
                grid_points=2, feature_max_features=100,
                country_min_entities=2,
                model=BaselineConfig(num_boost_round=5,
                                     early_stopping_rounds=0,
                                     min_data_in_leaf=1, num_threads=1),
            ),
            top_k=2, max_candidates=10, shard_size=4, test_shard_size=4,
            threads=1, batch_entities=3, allow_exploratory=allow,
        )

    def _inputs(self, root):
        train = root / "train"
        train.mkdir()
        CPUTrainingTests()._fixture(train)
        test = root / "test"
        test.mkdir()
        for source in (1, 2, 3):
            shutil.copyfile(train / f"train_source{source}.tsv",
                            test / f"test_source{source}.tsv")
        self.assertFalse((test / "test_ground_truth.tsv").exists())

    def test_run_resume_and_input_change_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._inputs(root)
            with self.assertRaisesRegex(ValueError, "explicit"):
                run_cpu_workflow(self._config(root, allow=False))
            self.assertFalse((root / "work").exists())
            first = run_cpu_workflow(self._config(root))
            self.assertFalse(first["phase4_reused"])
            self.assertFalse(first["model_reused"])
            self.assertFalse(first["output_reused"])
            self.assertEqual(first["validated_output"]["source1_rows"], 9)
            self.assertIn("provisional", first["scope"])
            self.assertTrue((root / "output" / "output_manifest.json").exists())
            second = run_cpu_workflow(self._config(root))
            self.assertTrue(second["phase4_reused"])
            self.assertTrue(second["model_reused"])
            self.assertTrue(second["output_reused"])
            self.assertEqual(second["validated_output"]["source1_rows"], 9)
            model_report = json.loads((root / "work" / "model" /
                                       "training_report.json").read_text())
            self.assertEqual(model_report["sampled_source_entities"], 9)
            main([
                "run", "--train-dir", str(root / "train"),
                "--test-dir", str(root / "test"),
                "--work-root", str(root / "work"),
                "--output-dir", str(root / "output"),
                "--train-entities", "9", "--max-train-pairs", "100",
                "--folds", "3", "--grid-points", "2",
                "--feature-max-features", "100", "--boost-rounds", "5",
                "--country-min-entities", "2",
                "--min-data-in-leaf", "1", "--top-k", "2",
                "--max-candidates", "10", "--shard-size", "4",
                "--test-shard-size", "4",
                "--threads", "1", "--batch-entities", "3",
                "--allow-provisional",
            ])
            marker_path = root / "work" / "phase4-work" / "char_name.complete"
            original_marker = marker_path.read_text(encoding="utf-8")
            changed_marker = json.loads(original_marker)
            changed_marker["retrieval_config"]["max_token_df"] = 1
            marker_path.write_text(json.dumps(changed_marker), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "marker mismatch"):
                run_cpu_workflow(self._config(root))
            marker_path.write_text(original_marker, encoding="utf-8")
            with (root / "train" / "train_source1.tsv").open("a") as handle:
                handle.write("S1-extra\tNew Shop\t\tUS\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_cpu_workflow(self._config(root))

    def test_second_run_rejects_locked_work_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            with (work / ".pipeline.lock").open("a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "another CPU pipeline"):
                    run_cpu_workflow(self._config(root))

    def test_parallel_workflow_matches_serial_fixture_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._inputs(root)
            serial = self._config(root)
            serial_result = run_cpu_workflow(serial)
            parallel = replace(
                serial, work_root=root / "parallel-work",
                output_dir=root / "parallel-output", threads=2,
                training=replace(
                    serial.training,
                    model=replace(serial.training.model, num_threads=2),
                ),
            )
            parallel_result = run_cpu_workflow(parallel)
            self.assertEqual(serial_result["validated_output"],
                             parallel_result["validated_output"])
            for name in ("matching_results.tsv", "candidate_pairs.tsv"):
                self.assertEqual((serial.output_dir / name).read_bytes(),
                                 (parallel.output_dir / name).read_bytes())
            report = json.loads((parallel.work_root / "model" /
                                 "training_report.json").read_text())
            self.assertEqual(report["config"]["model"]["num_threads"], 2)


if __name__ == "__main__":
    unittest.main()
