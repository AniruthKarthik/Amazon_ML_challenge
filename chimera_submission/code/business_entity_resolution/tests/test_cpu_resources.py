"""CPU-core selection and Makefile command wiring without project-data runs."""

import argparse
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cpu_resources import available_cpu_count, parse_thread_count
from src.main import _parser, _training_config


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
        explicit = _parser().parse_args(base + ["--threads", "1"])
        self.assertEqual(_training_config(explicit).model.num_threads, 1)

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
        self.assertNotIn("--official-check-ids", result.stdout)


if __name__ == "__main__":
    unittest.main()
