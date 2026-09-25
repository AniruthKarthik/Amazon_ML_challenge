"""Synthetic checks of retrieval-only, multi-positive-aware Phase 4 metrics."""

import unittest

from src.phase4_metrics import RetrievalBenchmark, oracle_f05
from src.retrieval import Candidate, ChannelHit


def candidate(source, target, *channels):
    return Candidate(source, target, 1, 1.0,
                     tuple(ChannelHit(channel, 1, 1.0) for channel in channels))


class Phase4MetricsTests(unittest.TestCase):
    def test_oracle_f05_handles_multiple_positives_and_singletons(self):
        self.assertEqual(oracle_f05(0, 0), 1.0)
        self.assertEqual(oracle_f05(2, 0), 0.0)
        self.assertAlmostEqual(oracle_f05(2, 1), 1.25 / 1.5)
        self.assertEqual(oracle_f05(2, 2), 1.0)

    def test_retrieval_metrics_groups_unions_and_unique_hits(self):
        metadata = {
            "S2-a": ("S2", "US"), "S2-b": ("S2", "India"),
            "S2-c": ("S2", "US"), "S3-a": ("S3", "US"),
        }
        benchmark = RetrievalBenchmark({"all": 4, "S2": 3, "S3": 1,
                                        "country:US": 3, "country:India": 1})
        benchmark.add_query("S1-a", "US", {
            "S2-a": metadata["S2-a"], "S3-a": metadata["S3-a"],
        }, [
            candidate("S1-a", "S2-a", "exact_name", "char_name"),
            candidate("S1-a", "S3-a", "rare_token"),
            candidate("S1-a", "S2-b", "char_address"),
        ], metadata)
        benchmark.add_query("S1-b", "India", {}, [], metadata)
        benchmark.add_query("S1-c", "US", {"S2-c": metadata["S2-c"]}, [], metadata)
        report = benchmark.report()
        overall = report["groups"]["overall"]
        self.assertEqual(overall["ground_truth_links"], 3)
        self.assertEqual(overall["recovered_links"], 2)
        self.assertAlmostEqual(overall["candidate_recall"], 2 / 3)
        self.assertEqual(overall["entity_any_hit_recall"], 0.5)
        self.assertEqual(overall["entity_complete_recall"], 0.5)
        self.assertAlmostEqual(overall["oracle_entity_macro_f05"], 2 / 3)
        self.assertEqual(overall["candidates_per_query"], 1.0)
        self.assertEqual(overall["reduction_ratio"], 0.75)
        self.assertEqual(report["candidate_count_distribution"]["zero_candidate_queries"], 2)
        self.assertEqual(report["candidate_count_distribution"]["p99"], 3)
        self.assertEqual(report["groups"]["target_source:S2"]["candidate_recall"], 0.5)
        self.assertEqual(report["groups"]["target_source:S3"]["candidate_recall"], 1.0)
        self.assertEqual(report["channel_views"]["Exact ∪ Char"]["recovered_links"], 1)
        self.assertEqual(report["channel_views"]["Exact ∪ Char ∪ Rare"]["recovered_links"], 2)
        self.assertEqual(report["incremental_union_recovered_links"], {
            "Exact": 1, "Exact ∪ Char": 0, "Exact ∪ Char ∪ Rare": 1,
        })
        self.assertEqual(report["unique_ground_truth_recovered_only_by_channel"]["rare_token"], 1)
        self.assertEqual(report["unique_ground_truth_recovered_only_by_channel"]["exact_name"], 0)
        self.assertEqual(benchmark.miss_sample, [("S1-c", "S2-c")])
        self.assertEqual(report["misses_seen"], 1)
        self.assertIn("deferred", report["metric_definitions"]["ranking_metrics"])

    def test_miss_reservoir_is_bounded_and_reproducible(self):
        metadata = {f"S2-{n}": ("S2", "US") for n in range(20)}
        samples = []
        for _ in range(2):
            benchmark = RetrievalBenchmark({"all": 20, "S2": 20, "S3": 0},
                                           sample_limit=5, random_seed=7)
            benchmark.add_query("S1-a", "US", metadata, [], metadata)
            samples.append(benchmark.miss_sample)
            self.assertEqual(benchmark.misses_seen, 20)
            self.assertEqual(len(benchmark.miss_sample), 5)
        self.assertEqual(samples[0], samples[1])

    def test_rejects_cross_entity_and_duplicate_candidates(self):
        metadata = {"S2-a": ("S2", "US")}
        benchmark = RetrievalBenchmark({"all": 1, "S2": 1, "S3": 0})
        with self.assertRaises(ValueError):
            benchmark.add_query("S1-a", "US", {},
                                [candidate("S1-b", "S2-a", "exact_name")], metadata)
        with self.assertRaises(ValueError):
            benchmark.add_query("S1-a", "US", {}, [
                candidate("S1-a", "S2-a", "exact_name"),
                candidate("S1-a", "S2-a", "char_name"),
            ], metadata)


if __name__ == "__main__":
    unittest.main()
