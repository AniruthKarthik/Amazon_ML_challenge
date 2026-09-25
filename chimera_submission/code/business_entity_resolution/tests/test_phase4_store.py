"""Small end-to-end checks of the disk-backed Phase 4 input and lexical channels."""

import csv
import gzip
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from src.data_contract import DataContractError, load_source
from src.normalization import normalize_source
from src.phase4_char import char_channel
from src.phase4_benchmark import run
from src.phase4_store import (
    build_store,
    candidate_array,
    exact_channel,
    open_store,
    rare_token_channel,
)
from src.retrieval import RetrievalConfig, generate_candidates
from src.workflow_progress import WorkflowProgress


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


class Phase4StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / "train"
        self.train.mkdir()
        (self.train / "train_source1.tsv").write_text(
            HEADER + "S1-a\tAcme Corp\t12 Rd\tUS\nS1-b\tBlue Star\t\tIndia\n",
            encoding="utf-8",
        )
        (self.train / "train_source2.tsv").write_text(
            HEADER + "S2-a\tAcme Corporation\t12 Road\tUS\nS2-b\tStar Blue\t\tIndia\n",
            encoding="utf-8",
        )
        (self.train / "train_source3.tsv").write_text(
            HEADER + "S3-a\tAcme Corp\t\tUS\n", encoding="utf-8",
        )
        (self.train / "train_ground_truth.tsv").write_text(
            "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a,S3-a\nS1-b\tS2-b\n",
            encoding="utf-8",
        )

    def test_store_validates_and_preserves_training_rows(self):
        path = self.root / "store.sqlite"
        counts = build_store(self.train, path)
        self.assertEqual(counts, {"S1": 2, "S2": 2, "S3": 1, "ground_truth_links": 3})
        connection = open_store(path)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM truth_indexed").fetchone()[0], 3)
        self.assertEqual(connection.execute("SELECT name_core FROM targets WHERE entity_id='S2-a'").fetchone()[0], "acme")
        self.assertEqual(connection.execute("SELECT raw_name FROM source1 WHERE entity_id='S1-a'").fetchone()[0], "Acme Corp")
        connection.close()
        with self.assertRaises(FileExistsError):
            build_store(self.train, path)

    def test_exact_and_rare_channels_match_phase3_on_fixture(self):
        path = self.root / "store.sqlite"
        build_store(self.train, path)
        connection = open_store(path)
        config = RetrievalConfig(top_k=3, max_candidates=15, max_token_df=100)
        exact_name_path = self.root / "exact_name.bin"
        exact_core_path = self.root / "exact_core.bin"
        rare_path = self.root / "rare.bin"
        exact_channel(connection, "name_clean", exact_name_path, config)
        exact_channel(connection, "name_core", exact_core_path, config)
        for view, serial_path in (("name_clean", exact_name_path),
                                  ("name_core", exact_core_path)):
            parallel_path = self.root / f"{view}.parallel.bin"
            exact_channel(connection, view, parallel_path, config, threads=2)
            np.testing.assert_array_equal(
                candidate_array(serial_path, 2, 3, create=False),
                candidate_array(parallel_path, 2, 3, create=False))
        with self.assertRaisesRegex(ValueError, "threads must be positive"):
            exact_channel(connection, "name_clean", self.root / "invalid.bin",
                          config, threads=0)
        rare_token_channel(connection, rare_path, config)
        rare_scores = np.memmap(rare_path.with_suffix(".float32"), dtype=np.float32,
                                mode="r", shape=(2, 3))
        rare_indices = candidate_array(rare_path, 2, 3, create=False)
        self.assertTrue(np.isfinite(rare_scores[rare_indices >= 0]).all())
        self.assertTrue(np.isneginf(rare_scores[rare_indices < 0]).all())
        target_ids = [row[0] for row in connection.execute("SELECT entity_id FROM targets ORDER BY seq")]
        source_ids = [row[0] for row in connection.execute("SELECT entity_id FROM source1 ORDER BY seq")]
        loaded = [normalize_source(load_source(self.train / f"train_source{n}.tsv", n)) for n in (1, 2, 3)]
        expected = {}
        for candidate in generate_candidates(*loaded, config):
            expected.setdefault((candidate.source1_entity_id, candidate.candidate_entity_id), set()).update(candidate.channels)
        for channel, output_path in (("exact_name", exact_name_path),
                                     ("exact_core", exact_core_path), ("rare_token", rare_path)):
            hits = candidate_array(output_path, 2, 3, create=False)
            for source_seq, source_id in enumerate(source_ids):
                actual = {target_ids[index] for index in hits[source_seq] if index >= 0}
                expected_ids = {target_id for (query_id, target_id), channels in expected.items()
                                if query_id == source_id and channel in channels}
                self.assertEqual(actual, expected_ids)
        connection.close()

    def test_rejects_unknown_ground_truth_target(self):
        (self.train / "train_ground_truth.tsv").write_text(
            "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-missing\nS1-b\t\n",
            encoding="utf-8",
        )
        with self.assertRaises(DataContractError):
            build_store(self.train, self.root / "bad.sqlite")

    def test_rare_token_ties_use_full_target_ids(self):
        first_id = "S2-" + "x" * 25 + "a"
        second_id = "S2-" + "x" * 25 + "b"
        (self.train / "train_source2.tsv").write_text(
            HEADER + f"{second_id}\tAcme\t\tUS\n{first_id}\tAcme\t\tUS\n",
            encoding="utf-8",
        )
        (self.train / "train_ground_truth.tsv").write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            f"S1-a\t{first_id}\nS1-b\t\n",
            encoding="utf-8",
        )
        path = self.root / "long_ids.sqlite"
        build_store(self.train, path)
        connection = open_store(path)
        output_path = self.root / "rare_long_ids.bin"
        rare_token_channel(connection, output_path,
                           RetrievalConfig(top_k=1, max_token_df=100))
        top_index = int(candidate_array(output_path, 2, 1, create=False)[0, 0])
        top_id = connection.execute(
            "SELECT entity_id FROM targets WHERE seq=?", (top_index,)
        ).fetchone()[0]
        self.assertEqual(top_id, first_id)
        connection.close()

    def test_rare_token_df_cutoff_and_scratch_cleanup(self):
        path = self.root / "store.sqlite"
        build_store(self.train, path)
        connection = open_store(path)
        output_path = self.root / "rare_df.int32"
        rare_token_channel(connection, output_path,
                           RetrievalConfig(top_k=2, max_token_df=1))
        hits = candidate_array(output_path, 2, 2, create=False)
        self.assertTrue(np.all(hits[0] == -1))
        self.assertEqual(int(hits[1, 0]), 1)
        scores = np.memmap(output_path.with_suffix(".float32"),
                           dtype=np.float32, mode="r", shape=(2, 2))
        self.assertEqual(float(scores[1, 0]), 1.0)
        self.assertEqual(list(self.root.glob("rare_postings_*")), [])
        with self.assertRaises(FileExistsError):
            rare_token_channel(connection, output_path,
                               RetrievalConfig(top_k=2, max_token_df=1))
        connection.close()

    def test_sharded_char_channels_match_phase3_on_fixture(self):
        path = self.root / "store.sqlite"
        build_store(self.train, path)
        connection = open_store(path)
        config = RetrievalConfig(top_k=2, max_candidates=10)
        loaded = [normalize_source(load_source(self.train / f"train_source{n}.tsv", n)) for n in (1, 2, 3)]
        expected = {}
        for candidate in generate_candidates(*loaded, config):
            expected.setdefault((candidate.source1_entity_id, candidate.candidate_entity_id), set()).update(candidate.channels)
        target_ids = [row[0] for row in connection.execute("SELECT entity_id FROM targets ORDER BY seq")]
        source_ids = [row[0] for row in connection.execute("SELECT entity_id FROM source1 ORDER BY seq")]
        for channel, view in (("char_name", "name_clean"),
                              ("char_address", "address_alias")):
            output_path = self.root / f"{channel}.bin"
            char_channel(connection, view, output_path, self.root / f"{channel}.scores",
                         config, shard_size=2, query_batch_size=1, threads=1)
            output = candidate_array(output_path, 2, 2, create=False)
            parallel_path = self.root / f"{channel}.parallel.bin"
            char_channel(connection, view, parallel_path,
                         self.root / f"{channel}.parallel.scores",
                         config, shard_size=2, query_batch_size=1, threads=2)
            np.testing.assert_array_equal(
                output, candidate_array(parallel_path, 2, 2, create=False))
            np.testing.assert_array_equal(
                np.memmap(self.root / f"{channel}.scores", dtype=np.float32,
                          mode="r", shape=(2, 2)),
                np.memmap(self.root / f"{channel}.parallel.scores",
                          dtype=np.float32, mode="r", shape=(2, 2)))
            for source_seq, source_id in enumerate(source_ids):
                actual = {target_ids[index] for index in output[source_seq] if index >= 0}
                expected_ids = {target_id for (query_id, target_id), channels in expected.items()
                                if query_id == source_id and channel in channels}
                self.assertEqual(actual, expected_ids)
        connection.close()

    def test_char_channel_resolves_large_top_k_ties_by_entity_id(self):
        target_rows = "".join(
            f"S2-z{number:02d}\tAcme Corp\t\tUS\n"
            for number in reversed(range(12))
        )
        (self.train / "train_source2.tsv").write_text(
            HEADER + target_rows, encoding="utf-8"
        )
        (self.train / "train_ground_truth.tsv").write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-a\tS2-z00\nS1-b\t\n", encoding="utf-8",
        )
        store_path = self.root / "ties.sqlite"
        build_store(self.train, store_path)
        connection = open_store(store_path)
        output_path = self.root / "char_ties.bin"
        char_channel(
            connection, "name_clean", output_path, self.root / "char_ties.scores",
            RetrievalConfig(top_k=2), shard_size=13,
            query_batch_size=2, threads=1,
        )
        hit_indices = candidate_array(output_path, 2, 2, create=False)[0]
        hit_ids = [
            connection.execute("SELECT entity_id FROM targets WHERE seq=?", (int(index),))
            .fetchone()[0]
            for index in hit_indices
        ]
        self.assertEqual(hit_ids, ["S2-z00", "S2-z01"])
        connection.close()

    def test_full_training_only_benchmark_writes_reproducible_artifacts(self):
        output_dir = self.root / "reports"
        work_dir = self.root / "work"
        config = RetrievalConfig(top_k=2, max_candidates=10)
        progress = WorkflowProgress(7)
        output = io.StringIO()
        with redirect_stdout(output):
            run(self.train, output_dir, work_dir, config, "all", shard_size=2,
                threads=2, progress=progress)
        progress.check_complete()
        self.assertIn("[7/7] DONE Training retrieval benchmark and report",
                      output.getvalue())
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["groups"]["overall"]["ground_truth_links"], 3)
        self.assertEqual(metrics["groups"]["overall"]["recovered_links"], 3)
        self.assertEqual(metrics["groups"]["overall"]["candidate_recall"], 1.0)
        self.assertEqual(metrics["miss_taxonomy"]["sample_size"], 0)
        self.assertIn("no test labels", metrics["manifest"]["evaluation_scope"])
        with gzip.open(output_dir / "ground_truth_link_status.tsv.gz", "rt", encoding="utf-8") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row[2] == "1" for row in rows[1:]))
        self.assertIn("No ranking-failure rate", (output_dir / "phase4_report.md").read_text(encoding="utf-8"))
        run(self.train, output_dir, work_dir, config, "metrics", shard_size=2, threads=2)
        with self.assertRaises(ValueError):
            run(self.train, output_dir, work_dir,
                RetrievalConfig(top_k=1, max_candidates=5), "exact_name",
                shard_size=2, threads=1)


if __name__ == "__main__":
    unittest.main()
