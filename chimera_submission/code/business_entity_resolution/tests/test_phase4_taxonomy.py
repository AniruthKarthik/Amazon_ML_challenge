"""Ensure miss taxonomy remains deterministic and keeps retrieval distinct."""

import unittest

from src.phase4_taxonomy import classify_retrieval_miss


class Phase4TaxonomyTests(unittest.TestCase):
    def test_token_reordering(self):
        result = classify_retrieval_miss("Blue Star", "12 Main Road",
                                         "Star Blue", "12 Main Rd")
        self.assertEqual(result["primary"], "token_reordering")

    def test_spelling_and_missing_address_indicators(self):
        result = classify_retrieval_miss("Acme Bakery", "",
                                         "Acne Bakery", "Main Road")
        self.assertIn("missing_fields", result["indicators"])
        self.assertIn("spelling_corruption_or_variation", result["indicators"])

    def test_script_change_is_not_called_a_ranking_failure(self):
        result = classify_retrieval_miss("Ram Traders", "Delhi",
                                         "राम ट्रेडर्स", "दिल्ली")
        self.assertIn("transliteration_or_script_change", result["indicators"])
        self.assertNotIn("ranking", str(result))

    def test_exact_channel_cap_has_priority(self):
        result = classify_retrieval_miss("Acme Corp", "Road",
                                         "Acme Corp", "Street")
        self.assertEqual(result["primary"], "exact_name_channel_cap_or_tie")


if __name__ == "__main__":
    unittest.main()
