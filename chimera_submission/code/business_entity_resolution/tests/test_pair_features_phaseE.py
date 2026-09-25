"""Phase E tests: v2 helpers (stdlib only, no numpy needed)."""

import unittest

from src.text_sim import (
    V2_FEATURE_NAMES,
    _acronym_match,
    _containment,
    _length_ratio,
    strip_country_features,
)


class PhaseEHelperTests(unittest.TestCase):
    def test_containment(self):
        self.assertEqual(_containment("acme corporation", "acme"), 1.0)
        self.assertEqual(_containment("", "acme"), 0.0)
        self.assertAlmostEqual(_containment("one two", "two three"), 0.5)

    def test_acronym(self):
        self.assertEqual(_acronym_match("international business machines", "ibm"), 1.0)
        self.assertEqual(_acronym_match("ibm", "international business machines"), 1.0)
        self.assertEqual(_acronym_match("acme corp", "acme corp"), 1.0)
        self.assertEqual(_acronym_match("", "acme"), 0.0)

    def test_length_ratio(self):
        self.assertEqual(_length_ratio("acme", "acme"), 1.0)
        self.assertEqual(_length_ratio("", "acme"), 0.0)
        self.assertAlmostEqual(_length_ratio("ab", "abcd"), 0.5)

    def test_strip_country(self):
        row = {"country_match": 1.0, "name_exact": 0.0, "country_missing": 0.0}
        stripped = strip_country_features(row)
        self.assertEqual(stripped, {"name_exact": 0.0})

    def test_v2_names_stable(self):
        self.assertIn("name_containment", V2_FEATURE_NAMES)
        self.assertIn("address_number_conflict", V2_FEATURE_NAMES)
        self.assertIn("is_s2_target", V2_FEATURE_NAMES)
        self.assertEqual(len(set(V2_FEATURE_NAMES)), len(V2_FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
