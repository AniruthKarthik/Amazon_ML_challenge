"""Phase F tests (stdlib only)."""

import unittest

from src.score_calibration import max_calibration_gap, reliability_bins


class ScoreCalibrationTests(unittest.TestCase):
    def test_bins(self):
        scores = [0.05, 0.15, 0.85, 0.95]
        labels = [0, 0, 1, 1]
        bins = reliability_bins(scores, labels, num_bins=2)
        self.assertEqual(len(bins), 2)
        self.assertEqual(bins[0]["count"], 2)
        self.assertEqual(bins[1]["count"], 2)
        self.assertAlmostEqual(bins[0]["empirical_rate"], 0.0)
        self.assertAlmostEqual(bins[1]["empirical_rate"], 1.0)
        gap = max_calibration_gap(bins)
        self.assertIsNotNone(gap)
        self.assertGreaterEqual(gap, 0.0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            reliability_bins([0.5], [0, 1])
        with self.assertRaises(ValueError):
            reliability_bins([0.5], [2])
        with self.assertRaises(ValueError):
            reliability_bins([0.5], [1], num_bins=0)


if __name__ == "__main__":
    unittest.main()
