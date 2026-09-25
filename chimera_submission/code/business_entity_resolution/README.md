# Business Entity Resolution Pipeline

**Team:** `chimera`  
**Challenge:** ML Challenge 2026 — Business Entity Resolution  

---

## 1. Overview

This package provides a self-contained, reproducible pipeline for bipartite business entity resolution (Source 1 reference matching against Source 2 and Source 3 candidate records).

Key stages:
1. **Data Contract & Integrity:** Strict TSV parser preserving empty strings without silent drops, ID schema validation, bipartite connected-component cross-validation splitting.
2. **Multi-View Normalization:** Deterministic generation of `clean` (NFKC, casefold, safe punctuation), `folded` (diacritic removal), `core` (legal suffix stripping), and `alias` (address abbreviation normalization) views.
3. **Multi-Channel Candidate Retrieval:** Exact normalized name, exact core name, char TF-IDF name KNN, char TF-IDF address KNN, word TF-IDF KNN, and rare-token inverted index with provenance tracking.
4. **Pair Feature Engineering:** Tabular feature vectors computing Levenshtein, Jaro-Winkler, token overlaps, numeric address matching, missingness context, country match/mismatch flags, and retrieval provenance.
5. **Leak-Free Pair Scoring:** LightGBM classifier cross-fitted on bipartite graph folds to ensure no connected cluster crosses validation folds.
6. **Robust Threshold Optimization:** Joint optimization of pair, entity, and score gap thresholds maximizing Macro $F_{0.5}$ with domain shift stress testing.
7. **Deterministic Inference:** Frozen test inference generating `matching_results.tsv` and `candidate_pairs.tsv`.

---

## 2. Environment Setup

Python 3.8+ is supported. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 3. End-to-End Execution

Run the complete pipeline from repository root:

```bash
python chimera_submission/code/business_entity_resolution/src/main.py \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir chimera_submission/output \
    --k-folds 5 \
    --seed 42
```

Outputs generated:
- `chimera_submission/output/matching_results.tsv`: Final matches for all Source 1 entities in `test_source1.tsv`.
- `chimera_submission/output/candidate_pairs.tsv`: Candidate blocking set evaluated by the matching model.

---

## 4. Submission Validation

Verify output files against competition rules:

```bash
python3 utils/validate_submission.py \
    --matching chimera_submission/output/matching_results.tsv \
    --candidate chimera_submission/output/candidate_pairs.tsv \
    --test-dir dataset/test
```
