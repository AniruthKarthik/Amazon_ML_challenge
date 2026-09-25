"""Unit and integration tests for pair feature engineering (Phase 6)."""

import numpy as np
import pandas as pd
import pytest
from chimera_submission.code.business_entity_resolution.src.features import (
    PairFeatureExtractor,
    extract_numbers,
    jaro_winkler_similarity,
    levenshtein_distance,
    normalized_levenshtein_similarity,
    token_dice,
    token_jaccard,
)


def test_levenshtein_and_jw():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert normalized_levenshtein_similarity("kitten", "sitting") == pytest.approx(1.0 - 3 / 7)
    assert normalized_levenshtein_similarity("same", "same") == 1.0
    assert normalized_levenshtein_similarity("", "") == 1.0

    jw_identical = jaro_winkler_similarity("martha", "martha")
    assert jw_identical == 1.0
    jw_sim = jaro_winkler_similarity("martha", "marhta")
    assert jw_sim > 0.90
    assert jaro_winkler_similarity("", "test") == 0.0


def test_token_jaccard_and_dice():
    t1 = {"apple", "inc", "cupertino"}
    t2 = {"apple", "inc", "redmond"}
    assert token_jaccard(t1, t2) == pytest.approx(2 / 4)
    assert token_dice(t1, t2) == pytest.approx(4 / 6)


def test_extract_numbers():
    assert extract_numbers("123 Main St, Apt 4B") == {"123", "4"}
    assert extract_numbers("No numbers here") == set()


def test_build_features_complete_and_finite():
    s1_dict = {
        "S1-001": {
            "name_clean": "google llc",
            "name_core": "google",
            "address_clean": "1600 amphitheatre pkwy mountain view",
            "address_alias": "1600 amphitheatre parkway mountain view",
            "country": "US",
        },
        "S1-002": {
            "name_clean": "no address inc",
            "name_core": "no address",
            "address_clean": "",
            "address_alias": "",
            "country": "India",
        },
    }

    cand_dict = {
        "S2-001": {
            "name_clean": "google inc",
            "name_core": "google",
            "address_clean": "1600 amphitheatre parkway mountain view",
            "address_alias": "1600 amphitheatre parkway mountain view",
            "country": "US",
        },
        "S3-001": {
            "name_clean": "different company",
            "name_core": "different company",
            "address_clean": "123 elm st",
            "address_alias": "123 elm street",
            "country": "France",
        },
    }

    pairs_df = pd.DataFrame([
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-001",
            "channels": "exact_core,tfidf_name",
            "channel_count": 2,
            "max_score": 0.95,
            "min_rank": 1,
            "score_exact_name": 0.0,
            "score_exact_core": 1.0,
            "score_tfidf_name": 0.95,
            "score_tfidf_word": 0.90,
            "score_tfidf_addr": 0.88,
            "score_rare_token": 0.0,
        },
        {
            "source1_entity_id": "S1-002",
            "candidate_entity_id": "S3-001",
            "channels": "rare_token",
            "channel_count": 1,
            "max_score": 0.5,
            "min_rank": 10,
            "score_exact_name": 0.0,
            "score_exact_core": 0.0,
            "score_tfidf_name": 0.0,
            "score_tfidf_word": 0.0,
            "score_tfidf_addr": 0.0,
            "score_rare_token": 0.5,
        },
    ])

    feats = PairFeatureExtractor.build_features(pairs_df, s1_dict, cand_dict)

    assert len(feats) == 2
    feat_cols = [c for c in feats.columns if c.startswith("feat_")]
    assert len(feat_cols) > 25

    # Check assertions: NO NaNs, NO Infs
    assert not feats[feat_cols].isna().any().any()
    assert np.all(np.isfinite(feats[feat_cols].to_numpy()))

    # Row 0: Google LLC vs Google Inc
    row0 = feats.iloc[0]
    assert row0["feat_exact_core_name"] == 1.0
    assert row0["feat_country_match"] == 1.0
    assert row0["feat_country_mismatch"] == 0.0
    assert row0["feat_numeric_overlap_count"] == 1.0  # both have 1600
    assert row0["feat_ch_exact_core"] == 1.0

    # Row 1: India vs France
    row1 = feats.iloc[1]
    assert row1["feat_country_match"] == 0.0
    assert row1["feat_country_mismatch"] == 1.0
    assert row1["feat_addr_missing_s1"] == 1.0
    assert row1["feat_addr_both_present"] == 0.0
