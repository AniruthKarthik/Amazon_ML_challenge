"""Synthetic disk-backed candidate and unlabeled test-store integration tests."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_contract import DataContractError
from src.phase4_store import candidate_array
from src.pipeline_store import DiskCandidateStore, build_unlabeled_store
from src.retrieval import CHANNELS


class PipelineStoreTests(unittest.TestCase):
    def write_source(self, directory, source, rows):
        path = directory / f"test_source{source}.tsv"
        lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
        lines.extend("\t".join(row) for row in rows)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def make_inputs(self, directory):
        self.write_source(directory, 1, [
            ("S1-a", "Alpha Ltd", "Main Rd", "US"),
            ("S1-b", "No Candidate", "", "India"),
        ])
        self.write_source(directory, 2, [
            ("S2-a", "Alpha Limited", "Main Road", "US"),
            ("S2-b", "Beta", "High St", "India"),
        ])
        self.write_source(directory, 3, [
            ("S3-a", "Alfa", "Main Road", "US"),
        ])

    def test_unlabeled_store_and_bounded_candidate_union(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inputs(root)
            store_path = root / "test.sqlite"
            counts = build_unlabeled_store(root, store_path)
            self.assertEqual(counts, {"S1": 2, "S2": 2, "S3": 1})
            work = root / "retrieval"
            work.mkdir()
            for channel in CHANNELS:
                values = candidate_array(work / f"{channel}.int32", 2, 2, create=True)
                if channel == "exact_name":
                    values[0, 0] = 0
                elif channel == "char_name":
                    values[0, 0] = 0
                elif channel == "rare_token":
                    values[0, 0] = 2
                values.flush()
            for channel in ("char_name", "char_address"):
                scores = np.memmap(work / f"{channel}.float32", dtype=np.float32,
                                   mode="w+", shape=(2, 2))
                scores[:] = -np.inf
                if channel == "char_name":
                    scores[0, 0] = 0.75
                scores.flush()
            rare_scores = np.memmap(work / "rare_token.float32", dtype=np.float32,
                                    mode="w+", shape=(2, 2))
            rare_scores[:] = -np.inf
            rare_scores[0, 0] = 0.42
            rare_scores.flush()
            connection = sqlite3.connect(store_path)
            self.assertIsNone(connection.execute(
                "SELECT name FROM sqlite_master WHERE name='truth_indexed'"
            ).fetchone())
            candidates = DiskCandidateStore(connection, work, 2, 2)
            first = candidates.get_query(0)
            self.assertEqual(first.source.raw.entity_id, "S1-a")
            self.assertEqual([item.candidate_entity_id for item in first.candidates],
                             ["S2-a", "S3-a"])
            self.assertEqual(first.candidates[0].channels,
                             ("exact_name", "char_name"))
            self.assertAlmostEqual(first.candidates[1].hits[0].score, 0.42, places=6)
            self.assertEqual(first.targets["S2-a"].business_name_core, "alpha")
            self.assertEqual(candidates.get_query(1).candidates, ())
            self.assertEqual(len(tuple(candidates.iter_queries())), 2)
            dense_work = root / "dense"
            dense_work.mkdir()
            dense_indices = candidate_array(dense_work / "dense.int32", 2, 1, create=True)
            dense_indices[0, 0] = 2
            dense_indices.flush()
            dense_scores = np.memmap(dense_work / "dense.float32", dtype=np.float32,
                                     mode="w+", shape=(2, 1))
            dense_scores[:] = -np.inf
            dense_scores[0, 0] = 0.8
            dense_scores.flush()
            with_dense = DiskCandidateStore(connection, work, 2, 3,
                                            dense_work_dir=dense_work, dense_top_k=1)
            self.assertTrue(any("dense_name_address" in candidate.channels
                                for candidate in with_dense.get_query(0).candidates))
            with self.assertRaises(IndexError):
                candidates.get_query(2)
            connection.close()

    def test_test_store_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_inputs(root)
            self.write_source(root, 2, [
                ("S2-a", "Alpha", "", "US"),
                ("S2-a", "Alpha 2", "", "US"),
            ])
            with self.assertRaises(DataContractError):
                build_unlabeled_store(root, root / "bad.sqlite")
            self.assertFalse((root / "bad.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
