"""Synthetic tests for the stratified pilot harness (stdlib only)."""

import unittest

from src.pilot import (
    cardinality_bucket,
    pilot_manifest,
    stratum_key,
    stratified_sample,
)


class PilotTests(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(cardinality_bucket(0), "0")
        self.assertEqual(cardinality_bucket(1), "1")
        self.assertEqual(cardinality_bucket(2), "2-3")
        self.assertEqual(cardinality_bucket(3), "2-3")
        self.assertEqual(cardinality_bucket(4), "4+")
        self.assertEqual(stratum_key("US", 2), "us|2-3")
        self.assertEqual(stratum_key("", 0), "missing|0")

    def test_stratified_deterministic_and_proportional(self):
        ids = [f"S1-{i:04d}" for i in range(100)]
        countries = {sid: ("US" if i < 60 else "India") for i, sid in enumerate(ids)}
        sizes = {sid: (i % 5) for i, sid in enumerate(ids)}
        first = stratified_sample(ids, countries, sizes, 20, seed=7)
        second = stratified_sample(ids, countries, sizes, 20, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        # Both countries represented.
        got = {countries[s] for s in first}
        self.assertEqual(got, {"US", "India"})

    def test_minimal_coverage(self):
        ids = ["S1-a", "S1-b", "S1-c", "S1-d"]
        countries = {"S1-a": "US", "S1-b": "US", "S1-c": "India", "S1-d": "France"}
        sizes = {"S1-a": 0, "S1-b": 5, "S1-c": 1, "S1-d": 2}
        sampled = stratified_sample(ids, countries, sizes, 4, seed=1)
        self.assertEqual(sorted(sampled), sorted(ids))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            stratified_sample([], {}, {}, 5)
        with self.assertRaises(ValueError):
            stratified_sample(["S1-a"], {"S1-a": "US"}, {"S1-a": 0}, 0)
        with self.assertRaises(ValueError):
            stratified_sample(["S1-a"], {"S1-a": "US"}, {"S1-a": 0}, 2)

    def test_manifest_hash_stable(self):
        sampled = ["S1-a", "S1-b"]
        m1 = pilot_manifest(sampled, {"S1-a": "US", "S1-b": "US"},
                            {"S1-a": 0, "S1-b": 3}, seed=42)
        m2 = pilot_manifest(sampled, {"S1-a": "US", "S1-b": "US"},
                            {"S1-a": 0, "S1-b": 3}, seed=42)
        self.assertEqual(m1, m2)
        self.assertEqual(m1["pilot_size"], 2)


if __name__ == "__main__":
    unittest.main()
