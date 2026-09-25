"""Phase B tests: numeric blocker (opt-in), truncation audit, cap sweep."""

import unittest

from src.data_contract import BusinessRecord
from src.normalization import normalize_source
from src.retrieval import (
    NUMERIC_CHANNEL,
    RetrievalConfig,
    cap_sweep_counts,
    exact_truncation_stats,
    generate_candidates,
)


def source(rows):
    return normalize_source({
        identifier: BusinessRecord(identifier, name, address, country)
        for identifier, name, address, country in rows
    })


class PhaseBTests(unittest.TestCase):
    def test_defaults_unchanged(self):
        config = RetrievalConfig()
        self.assertIsNone(config.numeric_top_k)
        self.assertEqual(config.sweep_caps, ())
        s1 = source([("S1-a", "Acme Corp", "12 Main Rd", "US")])
        s2 = source([("S2-a", "Acme Corp", "12 Main Rd", "US")])
        cands = list(generate_candidates(s1, s2, {}, config))
        self.assertTrue(all(NUMERIC_CHANNEL not in c.channels for c in cands))

    def test_numeric_channel_opt_in(self):
        s1 = source([("S1-a", "Totally Different Name", "221B Baker Street", "US")])
        s2 = source([("S2-a", "Unrelated Business Name Xyz", "221B Baker Street Apt 5", "US")])
        s3 = source([("S3-a", "Another Name Entirely", "99 Other Road", "US")])
        config = RetrievalConfig(top_k=5, numeric_top_k=5)
        cands = list(generate_candidates(s1, s2, s3, config))
        by_pair = {(c.source1_entity_id, c.candidate_entity_id): c for c in cands}
        self.assertIn(("S1-a", "S2-a"), by_pair)
        self.assertIn(NUMERIC_CHANNEL, by_pair[("S1-a", "S2-a")].channels)

    def test_truncation_stats(self):
        s2 = source([(f"S2-{n:03d}", "Same Name", "", "US") for n in range(10)])
        stats = exact_truncation_stats(s2, "business_name_clean", top_k=3)
        self.assertEqual(stats["keys"], 1)
        self.assertEqual(stats["truncated_keys"], 1)
        self.assertEqual(stats["truncated_hits"], 7)
        self.assertAlmostEqual(stats["truncation_rate"], 0.7)

    def test_cap_sweep_counts(self):
        self.assertEqual(cap_sweep_counts(["a", "b", "c"], (1, 2, 5)), {1: 1, 2: 2, 5: 3})

    def test_invalid_numeric_config(self):
        with self.assertRaises(ValueError):
            RetrievalConfig(numeric_top_k=0)
        with self.assertRaises(ValueError):
            RetrievalConfig(sweep_caps=(10, 5))


if __name__ == "__main__":
    unittest.main()
