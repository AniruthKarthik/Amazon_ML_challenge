# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** chimera  
**Team Members:** chimera team  
**Submission Date:** 2026-09-25  

---

## 1. Executive Summary

Team Chimera built an evidence-driven, high-precision bipartite entity resolution system specifically optimized for the competition's macro $F_{0.5}$ metric. Our solution integrates deterministic multi-view text normalization, multi-channel lexical and character/word TF-IDF candidate retrieval with provenance tracking, tabular pair feature engineering (Levenshtein, Jaro-Winkler, token overlaps, numeric address overlap, and country interaction signals), and a leak-free cross-fitted LightGBM pair scorer partitioned on bipartite graph connected components. Set decisions and thresholds are jointly optimized on out-of-fold predictions with plateau sensitivity analysis to maintain operational robustness against open-set domain and country shifts (e.g. France in the test set).

---

## 2. Methodology

### 2.1 Problem Analysis
Business records from disparate commercial sources exhibit significant noise, typographic variations, legal suffix variations (e.g., `Pvt Ltd`, `Corporation`, `Inc.`), accent variations (`Café` vs `Cafe`), and address inconsistencies (abbreviations, municipal variations, landmark-based descriptors). Furthermore, the evaluation metric is Macro $F_{0.5}$, heavily penalizing false merges ($2\times$ weight on precision over recall) and rewarding accurate identification of singletons (entities with zero true matches score 1.0 if empty, but 0.0 on false merges). Additionally, the test set introduces `France`, an unseen country label absent in training data, necessitating open-set, domain-invariant features.

### 2.2 Solution Strategy
**Approach Type:** Multi-Channel Blocking + Leak-Free Tabular Pair Scorer + Joint Macro $F_{0.5}$ Robust Threshold Optimization.  
**Core Innovation:** Bipartite graph connected component cross-validation splitting to eliminate cross-fold data leakage, paired with a multi-view candidate retrieval engine (exact, core, char TF-IDF, word TF-IDF, rare-token) and joint parametric thresholding (pair threshold, entity max threshold, and score-gap gating) calibrated for precision and domain shift stability.

---

## 3. Candidate Generation (Blocking)
To maximize the recall ceiling without memory explosion:
- **Blocking channels used:**
  1. *Exact Clean Name:* NFKC normalized, casefolded safe punctuation exact index.
  2. *Exact Core Name:* Legal corporate suffixes stripped (`corp`, `inc`, `ltd`, `pvt ltd`, `gmbh`, `sa`, etc.).
  3. *Character TF-IDF Name KNN:* Character n-grams (3-4 chars) with cosine similarity search for typo tolerance.
  4. *Character TF-IDF Address KNN:* Character n-grams (3-5 chars) on normalized addresses.
  5. *Word TF-IDF Name KNN:* Word n-grams (1-2 words) to recover inverted and transposed multi-token names.
  6. *Rare-Token Inverted Index:* Low document-frequency distinctive tokens indexed directly.
- **Candidate pairs generated:** Deduplicated union capped at $K \le 60$ candidates per entity.
- **Recall Assurance:** Monitored through oracle entity-level $F_{0.5}$ and explicit channel diversity matrices tracking *Unique Ground Truth Recovered* per channel.

---

## 4. Matching Model

**Features used:**
- **Exact Match Indicators:** `exact_clean_name`, `exact_core_name`, `exact_clean_addr`, `exact_alias_addr`.
- **String Distance & Similarities:** Normalized Levenshtein similarities and Jaro-Winkler similarities across clean and core name views.
- **Token Overlaps:** Jaccard and Dice token similarities on names and addresses; length ratios; token count differences.
- **Numeric Address Features:** Regex digit extraction, numeric token overlap count, numeric Jaccard, and numeric mismatch flag (disjoint non-empty street/unit numbers).
- **Domain & Missingness Interactions:** Country match flag, country mismatch flag, country missing flag; address missing flags.
- **Retrieval Provenance:** Number of retrieving channels, channel indicator flags, minimum retrieval rank, maximum retrieval score, and individual channel scores.

**Model type:** LightGBM Gradient Boosted Decision Trees trained with binary log-loss across bipartite-graph connected component folds.  
**Threshold selection method:** Joint grid optimization of $(\theta_{\text{pair}}, \theta_{\text{entity}}, \theta_{\text{gap}})$ on cross-fitted out-of-fold Macro $F_{0.5}$, stress-tested with $\pm 0.02$ parameter perturbations and simulated country shift.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** Achieves high held-out validation Macro $F_{0.5}$ with near-zero false positive merges on singletons.
- **Common false positives (wrong merges):** Franchise branches with identical brand names but differing unit numbers in sparse address contexts (mitigated via numeric mismatch penalty).
- **Common false negatives (missed matches):** Drastic acronyms with zero token or character overlap where neither name nor address share lexical tokens.

---

## 6. Conclusion
The Chimera pipeline delivers a rigorous, modular, and leak-free solution that satisfies all competition constraints. By coupling multi-view text normalization with multi-channel candidate retrieval, leak-free bipartite folding, gradient-boosted pair scoring, and robust threshold optimization, our system achieves maximum precision-weighted macro $F_{0.5}$ while maintaining stability under domain shift.

---

## Appendix

### A. Code Artefacts
All code resides under `chimera_submission/code/business_entity_resolution/`:
- `src/data_contract.py`: TSV loader, schema validation, and bipartite graph connected component analyzer.
- `src/normalization.py`: Multi-view text normalizer (raw, clean, folded, core, alias).
- `src/retrieval.py`: Multi-channel candidate generator and provenance tracker.
- `src/features.py`: Tabular pair feature extractor (edit distances, token overlaps, numeric overlaps, provenance).
- `src/pair_model.py`: Cross-fitted LightGBM pair scorer and out-of-fold predictor.
- `src/entity_decision.py`: Entity set decision aggregator and robust threshold optimizer.
- `src/hard_negatives.py`: Cross-fitted hard negative mining.
- `src/meta_model.py`: Entity-level meta-model.
- `src/pipeline.py`: Production end-to-end pipeline.
- `src/main.py`: CLI entry point reproducing `matching_results.tsv` and `candidate_pairs.tsv`.
- `requirements.txt`: Pinned dependencies.
