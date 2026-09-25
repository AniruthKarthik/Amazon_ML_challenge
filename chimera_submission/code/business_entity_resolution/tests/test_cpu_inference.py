"""Tiny frozen-inference integration tests using unlabeled synthetic test TSVs."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.cpu_inference import (
    prepare_test_retrieval, predict_test, validate_outputs,
    verify_inference_output,
)
from src.cpu_training import CPUTrainingConfig, train_cpu_baseline
from src.entity_meta_model import generate_meta_oof_decisions, save_meta_artifact
from src.main import main
from src.pair_model import BaselineConfig
from src.phase4_store import open_store
from src.pipeline_store import DiskCandidateStore
from src.pipeline_provenance import model_code_sha256
from tests.test_cpu_training import CPUTrainingTests
from tests.test_entity_meta_model import fixture as meta_fixture


class CPUInferenceTests(unittest.TestCase):
    def test_prepare_predict_and_validate_without_test_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            train.mkdir()
            train_work = CPUTrainingTests()._fixture(train)
            train_connection = open_store(train / "store.sqlite")
            train_cpu_baseline(
                DiskCandidateStore(train_connection, train_work, 2, 2),
                root / "model",
                CPUTrainingConfig(
                    train_entities=9, max_train_pairs=100, folds=3,
                    grid_points=2, feature_max_features=100,
                    country_min_entities=2,
                    model=BaselineConfig(num_boost_round=5,
                                         early_stopping_rounds=0,
                                         min_data_in_leaf=1, num_threads=1),
                ),
            )
            train_connection.close()
            test_dir = root / "test"
            test_dir.mkdir()
            for source in (1, 2, 3):
                shutil.copyfile(train / f"train_source{source}.tsv",
                                test_dir / f"test_source{source}.tsv")
            self.assertFalse((test_dir / "test_ground_truth.tsv").exists())
            work = root / "test_work"
            counts = prepare_test_retrieval(test_dir, work, 2, 2,
                                            shard_size=4, threads=1)
            self.assertEqual(counts, {"S1": 9, "S2": 9, "S3": 9})
            self.assertEqual(prepare_test_retrieval(test_dir, work, 2, 2,
                                                    shard_size=4, threads=1), counts)
            with self.assertRaises(ValueError):
                predict_test(root / "model", work, root / "refused")
            result = predict_test(root / "model", work, root / "output",
                                  batch_entities=3, allow_exploratory=True)
            self.assertEqual(result["source1_rows"], 9)
            connection = open_store(work / "store.sqlite")
            validated = validate_outputs(
                connection, root / "output" / "matching_results.tsv",
                root / "output" / "candidate_pairs.tsv",
            )
            self.assertEqual(validated["source1_rows"], 9)
            connection.close()
            self.assertEqual(verify_inference_output(
                root / "model", work, root / "output")
                ["source1_rows"], 9)
            main(["validate", "--test-work", str(work),
                  "--output-dir", str(root / "output")])
            official = subprocess.run([
                sys.executable,
                str(Path(__file__).resolve().parents[4] / "utils" / "validate_submission.py"),
                "--matching", str(root / "output" / "matching_results.tsv"),
                "--candidate", str(root / "output" / "candidate_pairs.tsv"),
                "--test-dir", str(test_dir), "--check-ids",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(official.returncode, 0, official.stdout + official.stderr)
            report_path = root / "model" / "training_report.json"
            report = json.loads(report_path.read_text())
            report["model_code_sha256"] = "invalid"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different scoring"):
                verify_inference_output(root / "model", work, root / "output")
            report["model_code_sha256"] = model_code_sha256()
            report["selected_entity_decision"] = "meta"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match"):
                verify_inference_output(root / "model", work, root / "output")
            meta = generate_meta_oof_decisions(meta_fixture(), (0.5, 0.8))
            save_meta_artifact(root / "model" / "meta_model.joblib", meta)
            predict_test(root / "model", work, root / "meta_output",
                         batch_entities=3, allow_exploratory=True)
            connection = open_store(work / "store.sqlite")
            self.assertEqual(validate_outputs(
                connection, root / "meta_output" / "matching_results.tsv",
                root / "meta_output" / "candidate_pairs.tsv",
            )["source1_rows"], 9)
            connection.close()
            self.assertEqual(verify_inference_output(
                root / "model", work, root / "meta_output")
                ["source1_rows"], 9)
            with (root / "meta_output" / "matching_results.tsv").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_inference_output(root / "model", work, root / "meta_output")
            with self.assertRaises(FileExistsError):
                predict_test(root / "model", work, root / "output",
                             allow_exploratory=True)

    def test_validator_rejects_missing_or_unknown_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = CPUTrainingTests()._fixture(root)
            connection = open_store(root / "store.sqlite")
            matching = root / "matching.tsv"
            candidate = root / "candidate.tsv"
            matching.write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-0\tS2-0\n",
                encoding="utf-8",
            )
            candidate.write_text(
                "source1_entity_id\tcandidate_entity_ids\nS1-0\tS2-missing\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "subset"):
                validate_outputs(connection, matching, candidate)
            matching.write_text(
                "source1_entity_id\tmatched_entity_ids\n" +
                "".join(f"S1-{index}\t\n" for index in range(9)),
                encoding="utf-8",
            )
            candidate.write_text(
                "source1_entity_id\tcandidate_entity_ids\n" +
                "S1-0\tS2-missing\n" +
                "".join(f"S1-{index}\t\n" for index in range(1, 9)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not exist"):
                validate_outputs(connection, matching, candidate)
            connection.close()


if __name__ == "__main__":
    unittest.main()
