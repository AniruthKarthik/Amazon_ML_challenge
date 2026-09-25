"""Phase D tests: database-free file store (stdlib only)."""

import tempfile
import unittest
from pathlib import Path

from src.file_store import (
    build_exact_index,
    build_file_store,
    build_token_postings,
    load_records,
    load_truth,
    shard_exact_index,
)


def write_tsv(path, header, rows):
    path.write_text(header + "".join(rows), encoding="utf-8")


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
TRUTH_HEADER = "source1_entity_id\tmatched_entity_ids\n"


class FileStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.train = self.root / "train"
        self.train.mkdir()
        write_tsv(self.train / "train_source1.tsv", HEADER, [
            "S1-a\tAcme Corp\t9 Main Rd\tUS\n",
            "S1-b\tBlue Star\t\tIndia\n",
        ])
        write_tsv(self.train / "train_source2.tsv", HEADER, [
            "S2-a\tAcme Corporation\t9 Main Road\tUS\n",
        ])
        write_tsv(self.train / "train_source3.tsv", HEADER, [
            "S3-a\tBlue Star\tNear Park\tIndia\n",
        ])
        write_tsv(self.train / "train_ground_truth.tsv", TRUTH_HEADER, [
            "S1-a\tS2-a\n",
            "S1-b\tS3-a\n",
        ])

    def test_build_and_load(self):
        out = self.root / "store"
        counts = build_file_store(self.train, out)
        self.assertEqual(counts, {"S1": 2, "S2": 1, "S3": 1, "ground_truth_links": 2})
        s1 = load_records(out / "source1.tsv")
        tgt = load_records(out / "targets.tsv")
        self.assertEqual(len(s1), 2)
        self.assertEqual(len(tgt), 2)
        truth = load_truth(out / "truth.json")
        self.assertEqual(truth, {"S1-a": frozenset({"S2-a"}), "S1-b": frozenset({"S3-a"})})
        # No SQLite file created.
        self.assertFalse((out / "store.sqlite").exists())

    def test_exact_index_and_shards(self):
        out = self.root / "store"
        build_file_store(self.train, out)
        tgt = load_records(out / "targets.tsv")
        index = build_exact_index(tgt, "name_core")
        self.assertIn("acme", index)
        shards = shard_exact_index(index, 2, out / "shards")
        self.assertEqual(len(shards), 2)

    def test_token_postings(self):
        out = self.root / "store"
        build_file_store(self.train, out)
        tgt = load_records(out / "targets.tsv")
        postings, weights = build_token_postings(tgt, max_token_df=1000)
        self.assertIn("acme", postings)
        self.assertTrue(all(weights[t] > 0 for t in weights))

    def test_no_overwrite(self):
        out = self.root / "store"
        build_file_store(self.train, out)
        with self.assertRaises(FileExistsError):
            build_file_store(self.train, out)


if __name__ == "__main__":
    unittest.main()
