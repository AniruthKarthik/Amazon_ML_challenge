"""Tiny Phase 13 CPU/disk tests; no model download or project-data run."""

import json
import hashlib
import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dense_retrieval import (
    DENSE_CHANNEL, DenseConfig, build_faiss_index, dense_text, encode_target_cache,
    evaluate_dense_training_store, evaluate_dense_union, merge_dense_candidates,
    query_faiss_index,
)
from src.phase4_store import candidate_array
from src.retrieval import CHANNELS, Candidate, ChannelHit


class FakeEncoder:
    dimension = 4

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.asarray([
            [len(text) % 7 + 1, text.count("a") + 1,
             text.count("b") + 1, 1]
            for text in texts
        ], dtype=np.float32)


class HashEncoder:
    dimension = 4

    def encode(self, texts):
        return np.asarray([
            [int.from_bytes(hashlib.sha256(text.encode()).digest()[offset:offset + 4],
                            "big") / 2**32 for offset in (0, 4, 8, 12)]
            for text in texts
        ], dtype=np.float32)


class DenseRetrievalTests(unittest.TestCase):
    def test_disk_encoding_resumes_without_reencoding_complete_batches(self):
        config = DenseConfig(batch_size=2, nlist=2, pq_subquantizers=2,
                             nprobe=1, train_samples=2, top_k=2, threads=1)
        rows = [(index, f"name {index}", f"address {index}") for index in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "embeddings.float16"
            checkpoint = Path(directory) / "checkpoint.json"
            first = FakeEncoder()
            encode_target_cache(rows, 5, first, cache, checkpoint, config,
                                "revision-test", "input-test")
            self.assertEqual(first.calls, 3)
            self.assertEqual(json.loads(checkpoint.read_text())["completed"], 5)
            vectors = np.memmap(cache, dtype=np.float16, mode="r", shape=(5, 4))
            np.testing.assert_allclose(np.linalg.norm(vectors.astype(np.float32), axis=1),
                                       np.ones(5), atol=1e-3)
            second = FakeEncoder()
            encode_target_cache(rows, 5, second, cache, checkpoint, config,
                                "revision-test", "input-test")
            self.assertEqual(second.calls, 0)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                encode_target_cache(rows, 5, second, cache, checkpoint, config,
                                    "new-revision", "input-test")

    def test_text_and_candidate_union_keep_provenance_and_cap(self):
        self.assertEqual(dense_text(" Café ", "  Main Rd "), "Café [SEP] Main Rd")
        lexical = [Candidate("S1-a", "S2-a", 1, 1.0,
                             (ChannelHit("exact_name", 1, 1.0),))]
        merged = merge_dense_candidates(
            "S1-a", lexical, (("S2-a", 0.9), ("S3-b", 0.8)), 2,
        )
        self.assertEqual({candidate.candidate_entity_id for candidate in merged},
                         {"S2-a", "S3-b"})
        self.assertEqual(merged[0].channels, ("exact_name", DENSE_CHANNEL))
        self.assertEqual(merged[1].channels, (DENSE_CHANNEL,))
        self.assertEqual(len(merge_dense_candidates("S1-a", lexical,
                            (("S3-b", 0.8),), 1)), 1)

    def test_training_only_dense_union_gain_is_not_ranking(self):
        result = evaluate_dense_union([
            ({"a", "b"}, {"a", "x"}, {"b"}),
            (set(), set(), {"y"}),
        ])
        self.assertEqual(result.dense_unique_truth_recovered, 1)
        self.assertEqual(result.base_candidate_recall, 0.5)
        self.assertEqual(result.union_candidate_recall, 1.0)
        self.assertGreater(result.union_oracle_macro_f05, result.base_oracle_macro_f05)
        with self.assertRaises(ValueError):
            evaluate_dense_union([])

    def test_disk_backed_training_store_union_evaluation(self):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE source1(seq INTEGER PRIMARY KEY)")
        connection.executemany("INSERT INTO source1 VALUES (?)", [(0,), (1,)])
        connection.execute("CREATE TABLE truth_indexed(source_seq INTEGER,target_seq INTEGER)")
        connection.execute("INSERT INTO truth_indexed VALUES (0,2)")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for channel in CHANNELS:
                base = candidate_array(root / f"{channel}.int32", 2, 2, create=True)
                if channel == "exact_name":
                    base[0, 0] = 1
                base.flush()
            dense = candidate_array(root / "dense.int32", 2, 2, create=True)
            dense[0, 0] = 2
            dense.flush()
            result = evaluate_dense_training_store(
                connection, root, root / "dense.int32", 2, 2,
            )
            self.assertEqual(result.dense_unique_truth_recovered, 1)
            self.assertEqual(result.union_candidate_recall, 1)
        connection.close()

    def test_invalid_settings_and_pair_ids(self):
        with self.assertRaises(ValueError):
            DenseConfig(nlist=1, nprobe=2)
        with self.assertRaises(ValueError):
            merge_dense_candidates("S1-a", (), (("", 0.8),), 1)

    @unittest.skipUnless(importlib.util.find_spec("faiss"), "optional faiss-cpu not installed")
    def test_real_faiss_index_and_resumable_query(self):
        count = 10_000
        config = DenseConfig(top_k=3, batch_size=512, nlist=16,
                             pq_subquantizers=2, nprobe=4,
                             train_samples=count, threads=1)
        encoder = HashEncoder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "target.float16"
            encode_target_cache(
                ((index, f"business {index}", "") for index in range(count)),
                count, encoder, cache, root / "target.json", config,
                "revision-test", "input-test",
            )
            index_path = root / "index.faiss"
            build_faiss_index(cache, index_path, count, encoder.dimension, config)
            queries = [(index, f"business {index}", "") for index in range(4)]
            candidate_path = root / "hits.int32"
            score_path = root / "hits.float32"
            checkpoint = root / "query.json"
            query_faiss_index(queries, 4, encoder, index_path,
                              candidate_path, score_path, checkpoint, config,
                              "input-test")
            hits = np.memmap(candidate_path, dtype=np.int32, mode="r", shape=(4, 3))
            self.assertTrue(np.all((hits >= 0) & (hits < count)))
            self.assertEqual(json.loads(checkpoint.read_text())["completed"], 4)
            # Completed query batches are skipped on a second invocation.
            query_faiss_index(queries, 4, encoder, index_path,
                              candidate_path, score_path, checkpoint, config,
                              "input-test")


if __name__ == "__main__":
    unittest.main()
