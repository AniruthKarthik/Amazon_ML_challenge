"""Phase C tests: expanded legal suffixes and street aliases."""

import unittest

from src.normalization import alias_address, core_name


class NormalizationPhaseCTests(unittest.TestCase):
    def test_expanded_suffixes(self):
        self.assertEqual(core_name("acme inc"), "acme")
        self.assertEqual(core_name("acme incorporated"), "acme")
        self.assertEqual(core_name("acme company"), "acme")
        self.assertEqual(core_name("acme llc"), "acme")
        self.assertEqual(core_name("acme sarl"), "acme")
        self.assertEqual(core_name("acme sas"), "acme")
        self.assertEqual(core_name("acme pvt limited"), "acme")
        # "co" stays ambiguous and must be preserved.
        self.assertEqual(core_name("café and co"), "café and co")
        self.assertEqual(core_name("corporation road"), "corporation road")

    def test_expanded_aliases(self):
        self.assertEqual(alias_address("12 ave main blvd"), "12 avenue main boulevard")
        self.assertEqual(alias_address("5 ln oak dr"), "5 lane oak drive")
        # Idempotence holds for new values.
        self.assertEqual(alias_address(alias_address("12 ave main blvd")),
                         "12 avenue main boulevard")


if __name__ == "__main__":
    unittest.main()
