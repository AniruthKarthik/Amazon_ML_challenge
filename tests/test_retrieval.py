"""Unit and integration tests for candidate retrieval (Phase 3)."""

import pandas as pd
import pytest
from chimera_submission.code.business_entity_resolution.src.normalization import TextNormalizer
from chimera_submission.code.business_entity_resolution.src.retrieval import CandidateRetriever


@pytest.fixture
def sample_target_df():
    data = [
        {
            "entity_id": "S2-001",
            "business_name": "Microsoft Corporation",
            "business_address": "One Microsoft Way, Redmond, WA",
            "country": "US",
        },
        {
            "entity_id": "S2-002",
            "business_name": "Apple Inc.",
            "business_address": "1 Infinite Loop, Cupertino, CA",
            "country": "US",
        },
        {
            "entity_id": "S3-001",
            "business_name": "Tata Consultancy Services Limited",
            "business_address": "Bandra Kurla Complex, Mumbai",
            "country": "India",
        },
        {
            "entity_id": "S3-002",
            "business_name": "Zylophonic Quantum Bio",
            "business_address": "Near Central Station, Lyon",
            "country": "France",
        },
    ]
    df = pd.DataFrame(data)
    return TextNormalizer.normalize_dataframe(df)


@pytest.fixture
def sample_s1_df():
    data = [
        {
            # Exact clean name match
            "entity_id": "S1-001",
            "business_name": "Microsoft Corporation",
            "business_address": "One Microsoft Way, Redmond",
            "country": "US",
        },
        {
            # Core name match (different suffix)
            "entity_id": "S1-002",
            "business_name": "Tata Consultancy Services Pvt. Ltd.",
            "business_address": "BKC, Mumbai",
            "country": "India",
        },
        {
            # Typos (should match via Char TF-IDF)
            "entity_id": "S1-003",
            "business_name": "Apple Computer",
            "business_address": "1 Infinite Loop",
            "country": "US",
        },
        {
            # Distinctive rare token match
            "entity_id": "S1-004",
            "business_name": "Zylophonic Tech Labs",
            "business_address": "Lyon",
            "country": "France",
        },
        {
            # Unmatched singleton
            "entity_id": "S1-005",
            "business_name": "Completely Unrelated Brand X99",
            "business_address": "Nowhere 123",
            "country": "US",
        },
    ]
    df = pd.DataFrame(data)
    return TextNormalizer.normalize_dataframe(df)


def test_retriever_exact_and_core_channels(sample_target_df, sample_s1_df):
    retriever = CandidateRetriever(min_tfidf_score=0.20)
    retriever.fit(sample_target_df)

    candidates = retriever.retrieve(sample_s1_df)

    # S1-001 should match S2-001 via exact_name
    s1_1_cands = candidates["S1-001"]
    assert "S2-001" in s1_1_cands
    assert "exact_name" in s1_1_cands["S2-001"].channels

    # S1-002 should match S3-001 via exact_core
    s1_2_cands = candidates["S1-002"]
    assert "S3-001" in s1_2_cands
    assert "exact_core" in s1_2_cands["S3-001"].channels


def test_retriever_tfidf_typo_channel(sample_target_df, sample_s1_df):
    retriever = CandidateRetriever(min_tfidf_score=0.20)
    retriever.fit(sample_target_df)

    candidates = retriever.retrieve(sample_s1_df)

    # S1-003 ('Apple Computer') should retrieve S2-002 ('Apple Inc.') via tfidf_name
    s1_3_cands = candidates["S1-003"]
    assert "S2-002" in s1_3_cands
    assert "tfidf_name" in s1_3_cands["S2-002"].channels


def test_retriever_rare_token_channel(sample_target_df, sample_s1_df):
    retriever = CandidateRetriever(rare_token_min_len=4)
    retriever.fit(sample_target_df)

    candidates = retriever.retrieve(sample_s1_df)

    # S1-004 ('Zylophonic Tech Labs') should retrieve S3-002 ('Zylophonic Quantum Bio') via rare_token
    s1_4_cands = candidates["S1-004"]
    assert "S3-002" in s1_4_cands
    assert "rare_token" in s1_4_cands["S3-002"].channels


def test_retriever_provenance_and_capping(sample_target_df, sample_s1_df):
    retriever = CandidateRetriever(max_candidates_per_entity=2)
    retriever.fit(sample_target_df)

    candidates = retriever.retrieve(sample_s1_df)

    # Max candidates capped at 2 per S1 entity
    for s1_id, targets in candidates.items():
        assert len(targets) <= 2

    # Check tabular DataFrame export
    pairs_df = retriever.to_dataframe(candidates)
    assert "source1_entity_id" in pairs_df.columns
    assert "candidate_entity_id" in pairs_df.columns
    assert "channels" in pairs_df.columns
    assert "channel_count" in pairs_df.columns


def test_retriever_to_candidate_pairs_tsv(sample_target_df, sample_s1_df):
    retriever = CandidateRetriever()
    retriever.fit(sample_target_df)
    candidates = retriever.retrieve(sample_s1_df)

    all_s1_ids = set(sample_s1_df["entity_id"])
    tsv_df = retriever.to_candidate_pairs_tsv(candidates, all_s1_ids)

    assert len(tsv_df) == len(sample_s1_df)
    assert list(tsv_df.columns) == ["source1_entity_id", "candidate_entity_ids"]

    # Singleton S1-005 should have empty candidate_entity_ids
    s1_5_row = tsv_df[tsv_df["source1_entity_id"] == "S1-005"].iloc[0]
    assert s1_5_row["candidate_entity_ids"] == ""


def test_retriever_word_tfidf_channel():
    # Test that word TF-IDF recovers inverted/reordered multi-token names
    target_data = [
        {
            "entity_id": "S2-100",
            "business_name": "International Hospital Apollo Healthcare",
            "business_address": "Chennai",
            "country": "India",
        }
    ]
    target_df = TextNormalizer.normalize_dataframe(pd.DataFrame(target_data))

    s1_data = [
        {
            "entity_id": "S1-100",
            "business_name": "Apollo International Healthcare Hospital",
            "business_address": "Chennai",
            "country": "India",
        }
    ]
    s1_df = TextNormalizer.normalize_dataframe(pd.DataFrame(s1_data))

    # Without Word TF-IDF
    base_retriever = CandidateRetriever(enable_word_tfidf=False, min_tfidf_score=0.99)
    base_retriever.fit(target_df)
    base_cands = base_retriever.retrieve(s1_df)

    # With Word TF-IDF
    word_retriever = CandidateRetriever(
        enable_word_tfidf=True, min_word_tfidf_score=0.50, min_tfidf_score=0.99
    )
    word_retriever.fit(target_df)
    word_cands = word_retriever.retrieve(s1_df)

    assert "S2-100" in word_cands["S1-100"]
    assert "tfidf_word" in word_cands["S1-100"]["S2-100"].channels
