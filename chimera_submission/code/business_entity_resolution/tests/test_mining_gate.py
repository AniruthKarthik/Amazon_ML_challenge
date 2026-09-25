"""Phase H tests (stdlib only)."""

import tempfile
import unittest
from pathlib import Path

from src.mining_gate import append_gate_log, gate_record, proxy_gate


class MiningGateTests(unittest.TestCase):
    def test_gate(self):
        self.assertTrue(proxy_gate(0.80, 0.82, 0.01))
        self.assertFalse(proxy_gate(0.80, 0.805, 0.01))
        self.assertTrue(proxy_gate(0.80, 0.80, 0.0))
        with self.assertRaises(ValueError):
            proxy_gate(0.8, 0.9, -0.1)

    def test_record_and_log(self):
        record = gate_record("E10", 0.8, 0.82, 0.01, True, "clears margin")
        self.assertTrue(record["proceed_to_full_reopt"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.jsonl"
            append_gate_log(path, record)
            append_gate_log(path, record)
            self.assertEqual(len(path.read_text().strip().split("\n")), 2)


if __name__ == "__main__":
    unittest.main()
