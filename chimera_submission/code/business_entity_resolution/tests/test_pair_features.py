"""Phase 6 feature correctness, edge cases, and label isolation."""

import math
import unittest

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_contract import BusinessRecord
from src.normalization import normalize_record
from src.pair_features import (
    PairFeatureExtractor, TfidfCosineEncoder, jaro_winkler,
    normalized_levenshtein, token_jaccard,
)
from src.retrieval import Candidate, ChannelHit, RetrievalConfig, generate_candidates


def record(identifier, name, address="", country="US"):
    return normalize_record(BusinessRecord(identifier, name, address, country))


class PairFeatureTests(unittest.TestCase):
    def setUp(self):
        self.source = record("S1-a", "Acme Corp", "12 Rd, Unit 4", "US")
        self.target = record("S2-b", "Acme Corporation", "12 Road Unit 5", "us")
        self.other = record("S3-c", "Blue Star", "", "India")
        self.candidate = Candidate(
            "S1-a", "S2-b", 2, 0.9,
            (ChannelHit("exact_core", 1, 1.0),
             ChannelHit("char_name", 3, 0.8)),
        )
        records = [self.source, self.target, self.other]
        self.extractor = PairFeatureExtractor.fit_from_records(lambda: iter(records))

    def test_string_distances_and_missing_text(self):
        self.assertEqual(normalized_levenshtein("kitten", "sitting"), 1 - 3 / 7)
        self.assertEqual(normalized_levenshtein("abc", "abc"), 1.0)
        self.assertEqual(normalized_levenshtein("", ""), 0.0)
        self.assertAlmostEqual(jaro_winkler("martha", "marhta"), 0.961111111, places=6)
        self.assertEqual(jaro_winkler("", ""), 0.0)
        self.assertEqual(jaro_winkler("abc", "xyz"), 0.0)
        self.assertAlmostEqual(token_jaccard("one two", "two three"), 1 / 3)
        self.assertEqual(token_jaccard("", ""), 0.0)

    def test_streaming_tfidf_matches_sklearn_on_small_corpus(self):
        corpus = ["acme corp", "acme corporation", "blue star"]
        encoder = TfidfCosineEncoder.fit(iter(corpus), max_features=1000,
                                         cache_size=1)
        reference = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                    dtype=np.float32).fit_transform(corpus)
        expected = float((reference[0] @ reference[1].T).toarray()[0, 0])
        self.assertAlmostEqual(encoder.cosine(corpus[0], corpus[1]), expected,
                               places=5)
        self.assertEqual(encoder.cosine("", corpus[0]), 0.0)
        self.assertAlmostEqual(encoder.cosine(corpus[0], corpus[0]), 1.0,
                               places=6)
        self.assertLessEqual(len(encoder.cache), 1)
        empty_encoder = TfidfCosineEncoder.fit(iter(()))
        self.assertEqual(empty_encoder.cosine("acme", "acme"), 0.0)
        with self.assertRaises(ValueError):
            TfidfCosineEncoder.fit(corpus, max_features=0)

    def test_pair_features_cover_interactions_and_provenance(self):
        features = self.extractor.features(self.source, self.target, self.candidate)
        self.assertEqual(features["name_core_exact"], 1.0)
        self.assertEqual(features["address_shared_number_count"], 1.0)
        self.assertAlmostEqual(features["address_number_jaccard"], 1 / 3)
        self.assertEqual(features["country_match"], 1.0)
        self.assertEqual(features["country_mismatch"], 0.0)
        self.assertEqual(features["country_missing"], 0.0)
        self.assertEqual(features["channel_count"], 2.0)
        self.assertEqual(features["channel_exact_core_present"], 1.0)
        self.assertEqual(features["channel_exact_core_rank"], 1.0)
        self.assertEqual(features["channel_char_name_score"], 0.8)
        self.assertEqual(features["channel_rare_token_present"], 0.0)
        self.assertTrue(0 <= features["name_tfidf_cosine"] <= 1)
        self.assertEqual(tuple(features), self.extractor.feature_names)
        self.assertTrue(all(math.isfinite(value) for value in features.values()))

    def test_country_and_missing_address_context(self):
        candidate = Candidate("S1-a", "S3-c", 1, 0.4,
                              (ChannelHit("char_name", 1, 0.4),))
        features = self.extractor.features(self.source, self.other, candidate)
        self.assertEqual(features["country_mismatch"], 1.0)
        self.assertEqual(features["one_address_missing"], 1.0)
        self.assertEqual(features["target_address_missing"], 1.0)
        self.assertEqual(features["address_tfidf_cosine"], 0.0)
        blank_source = record("S1-empty", "Acme", "", "")
        blank_target = record("S2-empty", "Acme", "", "")
        blank_candidate = Candidate("S1-empty", "S2-empty", 1, 1.0,
                                    (ChannelHit("exact_name", 1, 1.0),))
        blank_features = self.extractor.features(
            blank_source, blank_target, blank_candidate
        )
        self.assertEqual(blank_features["country_missing"], 1.0)
        self.assertEqual(blank_features["both_addresses_missing"], 1.0)
        self.assertEqual(blank_features["address_exact"], 0.0)

    def test_transform_is_lazy_and_rejects_bad_pairs(self):
        source1 = {self.source.raw.entity_id: self.source}
        targets = {self.target.raw.entity_id: self.target}
        rows = list(self.extractor.transform([self.candidate], source1, targets))
        self.assertEqual(rows[0][0], self.candidate)
        self.assertEqual(rows[0][1]["candidate_rank"], 2.0)
        with self.assertRaises(ValueError):
            list(self.extractor.transform([self.candidate], {}, targets))
        with self.assertRaises(ValueError):
            self.extractor.features(self.other, self.target, self.candidate)
        bad_score = Candidate("S1-a", "S2-b", 1, float("nan"), ())
        with self.assertRaises(ValueError):
            self.extractor.features(self.source, self.target, bad_score)
        no_provenance = Candidate("S1-a", "S2-b", 1, 1.0, ())
        with self.assertRaises(ValueError):
            self.extractor.features(self.source, self.target, no_provenance)
        unknown_channel = Candidate("S1-a", "S2-b", 1, 1.0,
                                    (ChannelHit("word_name", 1, 0.9),))
        with self.assertRaises(ValueError):
            self.extractor.features(self.source, self.target, unknown_channel)

    def test_optional_channel_schema_can_be_configured_without_activation(self):
        extractor = PairFeatureExtractor(
            self.extractor.name_tfidf, self.extractor.address_tfidf,
            channels=("char_name", "word_name"),
        )
        candidate = Candidate("S1-a", "S2-b", 1, 0.9,
                              (ChannelHit("word_name", 1, 0.9),))
        features = extractor.features(self.source, self.target, candidate)
        self.assertEqual(features["channel_word_name_present"], 1.0)
        self.assertEqual(features["channel_char_name_present"], 0.0)

    def test_phase3_candidates_feed_phase6_features_without_labels(self):
        source1 = {self.source.raw.entity_id: self.source}
        source2 = {self.target.raw.entity_id: self.target}
        source3 = {self.other.raw.entity_id: self.other}
        candidates = generate_candidates(
            source1, source2, source3,
            RetrievalConfig(top_k=2, max_candidates=10, max_token_df=100),
        )
        rows = list(self.extractor.transform(
            candidates, source1, {**source2, **source3}
        ))
        self.assertTrue(rows)
        self.assertTrue(all(tuple(features) == self.extractor.feature_names
                            for _, features in rows))
        self.assertTrue(all("label" not in features for _, features in rows))


if __name__ == "__main__":
    unittest.main()
