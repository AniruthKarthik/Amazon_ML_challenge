"""Unicode, idempotence, and raw-preservation tests for text views."""

import unittest

from src.data_contract import BusinessRecord
from src.normalization import (
    alias_address,
    clean_text,
    collision_rates,
    core_name,
    fold_accents,
    normalize_record,
    normalize_source,
)


class NormalizationTests(unittest.TestCase):
    def test_clean_nfkc_casefold_punctuation_and_spacing(self):
        self.assertEqual(clean_text("  ＡＣＭＥ & Co., LTD  "), "acme and co ltd")
        self.assertEqual(clean_text("Straße—Café"), "strasse café")
        self.assertEqual(clean_text(""), "")

    def test_fold_latin_accents_without_transliterating_other_scripts(self):
        self.assertEqual(fold_accents("café São"), "cafe Sao")
        self.assertEqual(fold_accents("हिन्दी Ελληνικά"), "हिन्दी Ελληνικά")
        self.assertEqual(fold_accents(""), "")

    def test_core_removes_only_trailing_documented_suffixes(self):
        self.assertEqual(core_name("acme pvt ltd"), "acme")
        self.assertEqual(core_name("acme private limited corporation"), "acme")
        self.assertEqual(core_name("corporation road"), "corporation road")
        self.assertEqual(core_name("ltd"), "ltd")
        self.assertEqual(core_name(""), "")

    def test_address_alias_is_separate_view(self):
        self.assertEqual(alias_address("12 st john rd"), "12 street john road")
        self.assertEqual(alias_address(""), "")

    def test_views_are_idempotent(self):
        examples = ("", "  Café & Sons, PVT. LTD. ", "१२ स्ट्रीट", "Straße")
        for example in examples:
            with self.subTest(example=example):
                clean = clean_text(example)
                self.assertEqual(clean_text(clean), clean)
                folded = fold_accents(clean)
                self.assertEqual(fold_accents(folded), folded)
                core = core_name(clean)
                self.assertEqual(core_name(core), core)
                alias = alias_address(clean)
                self.assertEqual(alias_address(alias), alias)

    def test_raw_values_are_unchanged(self):
        raw = BusinessRecord("S1-x", "  Café & Co., Pvt Ltd ", "5 St. Rd", "France")
        normalized = normalize_record(raw)
        self.assertIs(normalized.raw, raw)
        self.assertEqual(normalized.raw.business_name, "  Café & Co., Pvt Ltd ")
        self.assertEqual(normalized.business_name_clean, "café and co pvt ltd")
        self.assertEqual(normalized.business_name_folded, "cafe and co pvt ltd")
        self.assertEqual(normalized.business_name_core, "café and co")
        self.assertEqual(normalized.business_address_clean, "5 st rd")
        self.assertEqual(normalized.business_address_alias, "5 street road")

    def test_normalize_source_keeps_identity_and_order(self):
        source = {
            "S1-b": BusinessRecord("S1-b", "Beta", "", "France"),
            "S1-a": BusinessRecord("S1-a", "Alpha", "", "India"),
        }
        normalized = normalize_source(source)
        self.assertEqual(list(normalized), list(source))
        for identifier in source:
            self.assertIs(normalized[identifier].raw, source[identifier])

    def test_collision_rates_ignore_empty_values(self):
        records = (
            normalize_record(BusinessRecord("S1-a", "Café", "", "US")),
            normalize_record(BusinessRecord("S1-b", "Cafe", "", "India")),
            normalize_record(BusinessRecord("S1-c", "Other", "Main Rd", "France")),
        )
        rates = collision_rates(records)
        self.assertEqual(rates["business_name_clean"], 0.0)
        self.assertAlmostEqual(rates["business_name_folded"], 2 / 3)
        self.assertAlmostEqual(rates["business_name_core"], 0.0)
        self.assertEqual(rates["business_address_clean"], 0.0)
        self.assertEqual(collision_rates([])["business_name_clean"], 0.0)


if __name__ == "__main__":
    unittest.main()
