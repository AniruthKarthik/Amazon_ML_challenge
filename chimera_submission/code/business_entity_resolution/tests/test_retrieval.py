"""Synthetic tests for bounded, provenance-preserving candidate retrieval."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data_contract import BusinessRecord, load_source
from src.normalization import normalize_source
from src.retrieval import (
    CHANNELS,
    RetrievalConfig,
    channel_volumes,
    generate_candidates,
    write_candidate_artifact,
)


def source(rows):
    return normalize_source({
        identifier: BusinessRecord(identifier, name, address, country)
        for identifier, name, address, country in rows
    })


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.s1 = source([
            ("S1-a", "Café Pvt Ltd", "12 Main Rd", "France"),
            ("S1-b", "Blue Star", "", "US"),
        ])
        self.s2 = source([
            ("S2-a", "Cafe Private Limited", "12 Main Road", "France"),
            ("S2-b", "Other Name", "12 Main Rd", "India"),
        ])
        self.s3 = source([
            ("S3-a", "Café Pvt Ltd", "", "France"),
            ("S3-b", "Star Blue", "", "US"),
        ])

    def test_all_channels_and_provenance(self):
        candidates = list(generate_candidates(self.s1, self.s2, self.s3))
        by_pair = {(c.source1_entity_id, c.candidate_entity_id): c for c in candidates}
        self.assertIn("exact_name", by_pair[("S1-a", "S3-a")].channels)
        self.assertIn("exact_core", by_pair[("S1-a", "S3-a")].channels)
        self.assertIn("char_name", by_pair[("S1-a", "S2-a")].channels)
        self.assertIn("char_address", by_pair[("S1-a", "S2-b")].channels)
        self.assertIn("rare_token", by_pair[("S1-b", "S3-b")].channels)
        self.assertEqual(len(by_pair), len(candidates))
        self.assertTrue(all(candidate.channel_count == len(candidate.hits) for candidate in candidates))
        self.assertTrue(all(0 < candidate.score <= 1.000001 for candidate in candidates))
        self.assertEqual(set(channel_volumes(candidates)), set(CHANNELS))

    def test_sorted_ids_and_ranks_are_deterministic(self):
        first = list(generate_candidates(self.s1, self.s2, self.s3))
        second = list(generate_candidates(self.s1, self.s2, self.s3))
        self.assertEqual(first, second)
        self.assertEqual([c.source1_entity_id for c in first],
                         sorted(c.source1_entity_id for c in first))
        for source_id in self.s1:
            ranks = [c.rank for c in first if c.source1_entity_id == source_id]
            self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_caps_prevent_common_name_explosion(self):
        s1 = source([("S1-a", "Same Name", "", "US")])
        s2 = source([(f"S2-{n:03d}", "Same Name", "", "US") for n in range(100)])
        config = RetrievalConfig(top_k=3, max_candidates=4, query_batch_size=1)
        candidates = list(generate_candidates(s1, s2, {}, config))
        self.assertLessEqual(len(candidates), 4)
        self.assertEqual(candidates[0].candidate_entity_id, "S2-000")
        self.assertTrue(all(hit.rank <= 3 for c in candidates for hit in c.hits))

    def test_empty_targets_or_unusable_text(self):
        self.assertEqual(list(generate_candidates(self.s1, {}, {})), [])
        s1 = source([("S1-a", "?", "", "US")])
        s2 = source([("S2-a", "!", "", "US")])
        self.assertEqual(list(generate_candidates(s1, s2, {})), [])

    def test_invalid_config_and_duplicate_target_ids(self):
        with self.assertRaises(ValueError):
            RetrievalConfig(top_k=0)
        with self.assertRaises(ValueError):
            RetrievalConfig(char_ngram_range=(3, 2))
        with self.assertRaises(ValueError):
            list(generate_candidates(self.s1, self.s2, self.s2))

    def test_artifact_contains_only_candidate_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate_set.tsv"
            write_candidate_artifact(path, generate_candidates(self.s1, self.s2, self.s3))
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertTrue(rows)
        self.assertEqual(set(rows[0]), {
            "source1_entity_id", "candidate_entity_id", "rank", "score",
            "channel_count", "channels", "provenance",
        })
        self.assertEqual(int(rows[0]["channel_count"]), len(rows[0]["channels"].split(",")))
        self.assertEqual(set(json.loads(rows[0]["provenance"])),
                         set(rows[0]["channels"].split(",")))

    def test_artifact_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate_set.tsv"
            write_candidate_artifact(path, generate_candidates(self.s1, self.s2, self.s3))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_candidate_artifact(path, ())
            self.assertEqual(path.read_bytes(), original)

    def test_loader_to_normalization_to_candidate_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            (root / "s1.tsv").write_text(header + "S1-a\tAcme Corp\t9 Main Rd\tFrance\n")
            (root / "s2.tsv").write_text(header + "S2-a\tAcme Corporation\t9 Main Road\tFrance\n")
            (root / "s3.tsv").write_text(header)
            s1 = normalize_source(load_source(root / "s1.tsv", 1))
            s2 = normalize_source(load_source(root / "s2.tsv", 2))
            s3 = normalize_source(load_source(root / "s3.tsv", 3))
            path = root / "candidates.tsv"
            write_candidate_artifact(path, generate_candidates(s1, s2, s3))
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source1_entity_id"], "S1-a")
        self.assertEqual(rows[0]["candidate_entity_id"], "S2-a")
        self.assertIn("exact_core", rows[0]["channels"].split(","))


if __name__ == "__main__":
    unittest.main()
