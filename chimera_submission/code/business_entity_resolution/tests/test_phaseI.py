"""Phase I tests (stdlib only)."""

import tempfile
import unittest
from pathlib import Path

from src.cpu_parallel import feature_workers, resolve_workers, shard_ranges
from src.freeze_check import REQUIRED_DETERMINISTIC, check_frozen_dir


class PhaseITests(unittest.TestCase):
    def test_workers(self):
        self.assertGreaterEqual(resolve_workers(), 1)
        self.assertEqual(resolve_workers(4), 4)
        with self.assertRaises(ValueError):
            resolve_workers(0)
        self.assertEqual(feature_workers(8), 4)
        self.assertEqual(feature_workers(1), 1)

    def test_shards(self):
        self.assertEqual(shard_ranges(10, 3), [(0, 4), (4, 7), (7, 10)])
        self.assertEqual(shard_ranges(0, 4), [])
        self.assertEqual(sum(stop - start for start, stop in shard_ranges(100, 7)), 100)
        with self.assertRaises(ValueError):
            shard_ranges(10, 0)

    def test_freeze_check(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            report = check_frozen_dir(model)
            self.assertFalse(report["ready"])
            self.assertEqual(len(report["missing"]), len(REQUIRED_DETERMINISTIC))
            for name in REQUIRED_DETERMINISTIC:
                (model / name).write_text("x")
            report = check_frozen_dir(model)
            self.assertTrue(report["ready"])


if __name__ == "__main__":
    unittest.main()
