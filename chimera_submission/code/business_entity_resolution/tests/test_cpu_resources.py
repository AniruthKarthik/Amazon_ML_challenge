"""CPU-core selection and Makefile command wiring without project-data runs."""

import argparse
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cpu_resources import available_cpu_count, parse_thread_count
from src.main import _parser, _training_config, main


class CPUResourcesTests(unittest.TestCase):
    def test_affinity_and_fallback(self):
        with patch("src.cpu_resources.os.sched_getaffinity", return_value={2, 4, 6}):
            self.assertEqual(available_cpu_count(), 3)
            self.assertEqual(parse_thread_count("auto"), 3)
        with patch("src.cpu_resources.os.sched_getaffinity", side_effect=OSError), \
             patch("src.cpu_resources.os.cpu_count", return_value=6):
            self.assertEqual(available_cpu_count(), 6)
        with patch("src.cpu_resources.os.sched_getaffinity", return_value=set()), \
             patch("src.cpu_resources.os.cpu_count", return_value=None):
            self.assertEqual(available_cpu_count(), 1)

    def test_explicit_override_and_invalid_counts(self):
        self.assertEqual(parse_thread_count("2"), 2)
        self.assertEqual(parse_thread_count("AUTO"), available_cpu_count())
        for value in ("0", "-1", "many", "1.5"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_thread_count(value)

    def test_workflow_cli_passes_one_resolved_count_to_training(self):
        base = ["run", "--train-dir", "train", "--test-dir", "test",
                "--work-root", "work", "--output-dir", "output",
                "--train-entities", "9", "--allow-provisional"]
        with patch("src.cpu_resources.os.sched_getaffinity", return_value={0, 1, 2}):
            auto = _parser().parse_args(base + ["--threads", "auto"])
            default = _parser().parse_args(base)
        self.assertEqual(auto.threads, 3)
        self.assertEqual(default.threads, 3)
        self.assertEqual(_training_config(auto).model.num_threads, 3)
        self.assertEqual(_training_config(auto).model.device_type, "cpu")
        self.assertEqual(_training_config(auto).model.max_bin, 255)
        explicit = _parser().parse_args(base + ["--threads", "1"])
        self.assertEqual(_training_config(explicit).model.num_threads, 1)
        gpu = _parser().parse_args(base + [
            "--device", "gpu", "--gpu-platform-id", "1",
            "--gpu-device-id", "2", "--max-bin", "31",
        ])
        model = _training_config(gpu).model
        self.assertEqual((model.device_type, model.gpu_platform_id,
                          model.gpu_device_id, model.max_bin),
                         ("gpu", 1, 2, 31))

    def test_makefile_exposes_configurable_safe_run(self):
        project = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["make", "-n", "run", "THREADS=auto", "TRAIN_ENTITIES=9",
             "PHASE4_WORK=old-work", "PHASE4_REPORT=old-report"],
            cwd=project, capture_output=True, text=True, check=True,
        )
        self.assertIn("--threads \"auto\"", result.stdout)
        self.assertIn("--train-entities \"9\"", result.stdout)
        self.assertIn("--allow-provisional", result.stdout)
        self.assertIn('--phase4-work "old-work"', result.stdout)
        self.assertIn('--phase4-report "old-report"', result.stdout)
        self.assertIn('--device "auto"', result.stdout)
        self.assertNotIn("--official-check-ids", result.stdout)

    def test_makefile_gpu_profile_is_fail_closed_and_32gb_bounded(self):
        project = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["make", "-n", "gpu-run"], cwd=project,
            capture_output=True, text=True, check=True,
        )
        self.assertIn('--device "gpu"', result.stdout)
        self.assertIn('--train-entities "10000"', result.stdout)
        self.assertIn('--max-train-pairs "2500000"', result.stdout)
        self.assertIn('--feature-max-features "50000"', result.stdout)
        self.assertIn('--batch-entities "512"', result.stdout)

    def test_fresh_explicit_gpu_run_probes_before_workflow(self):
        arguments = [
            "run", "--train-dir", "train", "--test-dir", "test",
            "--work-root", "unused-work", "--output-dir", "output",
            "--train-entities", "9", "--device", "gpu",
            "--allow-provisional",
        ]
        with patch("src.main.resolve_training_device",
                   side_effect=RuntimeError("GPU probe failed")) as probe, \
             patch("src.main.run_cpu_workflow") as workflow:
            with self.assertRaisesRegex(RuntimeError, "GPU probe failed"):
                main(arguments)
        probe.assert_called_once()
        workflow.assert_not_called()


if __name__ == "__main__":
    unittest.main()
