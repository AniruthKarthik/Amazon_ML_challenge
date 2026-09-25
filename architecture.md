# Architecture — Business Entity Resolution Pipeline

> **Team:** chimera  
> **Objective:** MAXIMIZE true held-out macro F₀.₅ accuracy at the ENTITY level, under the actual competition constraints.

---

## 1. Problem Formulation

The task is an **entity-level retrieval and set-decision problem**, evaluated on macro F₀.₅ over entities. The pipeline is fundamentally broken into three accuracy bottlenecks:
1. **Candidate-generation recall:** A perfect classifier cannot recover a match that was never retrieved.
2. **Pairwise ranking/classification quality:** Given candidates, the model must rank true matches above false ones.
3. **Entity-level set decision / thresholding:** Given all candidates and their scores, the system must decide whether the source record produces zero, one, or multiple matches.

---

## 2. Data Contract

Input TSVs contain `entity_id`, `business_name`, `business_address`, and `country`.
- **Loading:** `sep="\t"`, no dropping of NA/missing values.
- **Country:** Open-set label. Cannot rely on a fixed taxonomy (e.g., France is seen only in test).
- **Target schema:** Entity matching is bipartite (S1 → S2∪S3).

---

## 3. Normalization

Normalization must remain **multi-view** to preserve discriminative signals. 
- **Views produced:** `raw`, `clean` (NFKC, casefold, safe punctuation), `folded` (accent removal), `core` (legal suffix removal), `alias` (address abbreviation formatting).
- **Constraint:** Raw fields must remain available to downstream models. Never assume ASCII transliteration is universally correct.

---

## 4. Candidate Generation

Generates candidates via immutable, reproducible, and auditable channels.
- **Base Channels:** 
  - Exact normalized name
  - Exact core name
  - Character TF-IDF (name)
  - Character TF-IDF (address)
  - Rare-token index
- **Outputs:** Canonical candidate set artifact containing S1 id, candidate id, retrieval channels, rank, score, channel count, and retrieval provenance. No leakage-dependent features are stored here.

---

## 5. Retrieval Evaluation & Bottleneck Analysis

Phase 4 evaluates channels on diversity and downstream potential. Ranking failures are measured separately after Phase 8 using OOF LightGBM scores.
- **Diversity Measurement:** Track exact overlaps and unions (e.g., R_exact ∪ R_char). Critically, track **Unique GT Recovered** to determine how many true matches a channel recovers that were missed by all other channels.
- **Explicit Error Metrics:**
  - **Retrieval miss rate** = `(GT pairs absent from candidate set) / (all GT pairs)`
  - **Post-Phase-8 OOF top-1 ranking failure rate** = `(retrieved GT pairs ranked below first) / (retrieved GT pairs)`; also report retrieved-pair Recall@5/10 and false-positive competition without mixing in retrieval misses.
- **Decision Logic:** 
  - If token/reordering misses dominate retrieval errors → Evaluate Word TF-IDF.
  - If semantic/linguistic misses dominate retrieval errors → Evaluate Dense Retrieval.
  - If post-Phase-8 OOF ranking failure rate is substantial → Evaluate Cross-Encoder.
  - Note: Word TF-IDF and Dense Retrieval are independent, parallel conditional branches.

---

## 6. Pair Feature Engineering

Tabular feature vectors describing the relationship between an S1 entity and a candidate.
- **Features:** Exact matches, Levenshtein, Jaro-Winkler, token overlap, TF-IDF cosine, numeric address overlap, missingness flags, retrieval provenance/rank.
- **Interactions:** Country relation flags (match, missing, mismatch) and missing-field interactions.

---

## 7. Pair Model

**LightGBM** serves as the primary pairwise scorer. Tabular features (edit distances, token overlap, missingness) are naturally suited for gradient boosted trees. 
- Pair scoring merely optimizes ranking among candidates. It is evaluated via pair precision/recall as diagnostics, but selected via downstream entity macro F₀.₅.

---

## 8. Entity Aggregation

Pair scores are grouped by S1 `entity_id`. This step transitions the data from pair-level to entity-level.
- **Meta-features generated:** Max score, second-highest score, score gap, candidate count, scores above thresholds, score distribution statistics, score concentration, retrieval agreement, missingness context, country/source information.

---

## 9. Entity-level Decision / Meta-model

An optional secondary model that decides the predicted match set (zero, one, or multiple matches) using the complete candidate-score distribution.
- **Evaluation:** Compare empirically:
  A. Pair score + deterministic threshold
  B. Pair score + top-1 / score-gap logic
  C. Pair score + entity-level meta-model
- **Justification:** The meta-model is kept only if it improves cross-fitted entity-level macro F₀.₅. If deterministic rules perform equally well, the simpler approach is retained.

---

## 10. Threshold Optimization & Robustness

Thresholds (pair threshold, entity threshold, gap thresholds) are jointly optimized against the exact entity-level macro F₀.₅ metric using cross-fitted OOF predictions.
- **Operational Robustness:** Evaluated via simulated domain shift (e.g., leave-one-country-out). A robust policy maximizes mean F₀.₅ while preserving acceptable worst-case fold/country performance and maintaining threshold stability against small score perturbations.
- **Country Shift Remediation Rule:** If domain shift causes meaningful degradation, identify if it stems from the threshold policy. If yes, evaluate:
  - **Policy A:** One global threshold.
  - **Policy B:** Country/source-specific thresholds (must have sufficient labeled evidence).
  - **Policy C:** A conservative global threshold chosen from the OOF-stable range.
- **Test-Set Rule (Absolute):** Test distribution may be monitored, but test distribution may NOT determine thresholds. No automatic unlabeled test overrides are permitted. Conservative thresholds must be validated on labeled OOF data.
- **Downstream Re-optimization:** Whenever an upstream change occurs (e.g., retrieval, K, pair model), thresholds must NEVER be reused. Re-optimizing thresholds on the new OOF distribution is mandatory.

---

## 11. OOF / Cross-Fitting

Leakage prevention is central to the architecture.
- **Protocol:** Models generating upstream predictions (pair model) must not be trained on the data they predict. Entity-level features and threshold optimization are fit strictly on out-of-fold (OOF) pair predictions.
- **Downstream Regeneration:** ANY material change to upstream components (retrieval channels, K, cap, pair model, features, hard-negatives, dense retrieval, cross-encoders) REQUIRES regenerating OOF predictions, rebuilding meta-features, retraining the meta-model, and reoptimizing thresholds to compute the true final entity F₀.₅.
  - **Compute Gate:** Full regeneration is expensive at this candidate scale. Before triggering it, evaluate the candidate change on a cheap proxy metric (pair AUC / sampled OOF pair F₀.₅) against the current baseline. Only commit to full OOF/meta-feature/threshold regeneration if the proxy clears a pre-defined minimum improvement margin. Rejected candidates are logged, not silently dropped.
- **Validation:** Connected-component (graph-aware) GroupKFold is mandatory.

---

## 12. Hard-Negative Mining

Improves the pair model's discriminative power.
- **Mechanism:** Must be explicitly cross-fitted. Fold A's model mines negatives for Fold B to avoid bias. 
- **Acceptance:** Kept only if it improves final entity-level macro F₀.₅. Any change here forces a full downstream re-optimization loop.

---

## 13. Optional / Conditional Components

These components are independently triggered based on specific error taxonomy evidence from Phase 4. None are mandatory.
1. **Word-level TF-IDF retrieval:** Triggered by token/reordering misses. Evaluated independently of Dense Retrieval.
2. **Dense bi-encoder retrieval:** Triggered by semantic/linguistic misses. Evaluated independently of Word TF-IDF. Accepted ONLY if it yields a net improvement in the FINAL cross-fitted entity-level macro F₀.₅, not just candidate recall.
3. **Cross-encoder pair scoring:** Triggered by high post-Phase-8 OOF ranking failure rates. Purpose: resolve difficult retrieved pairwise ranking/classification cases.
4. **Isotonic score calibration:** Evaluated as an optional transformation of pair scores if it improves downstream entity-level F₀.₅ or threshold stability.

---

## 14. Error Taxonomy

Errors are systematically categorized and quantified to drive architecture decisions:
- **Categories:** Retrieval miss, Ranking error, Pair classification error, False-positive entity, False-negative entity, Ambiguous entity, Normalization failure, Country shift, Missing-field failure.

---

## 15. Ablation / Model Selection

The final system is selected via cross-fitted entity-level macro F₀.₅. 
- **Ensemble Evaluation:** Compare LightGBM against tabular ensembles (CatBoost, LR Stacker) using identical OOF folds and feature sets. Ensure full downstream re-optimization loops are executed before comparison.
- **Reporting:** Generate a comprehensive ablation report detailing OOF entity F₀.₅ mean, variance, worst-fold, and per-country breakdown for every candidate model configuration.
- No model is added solely for complexity. The single most robust architecture achieving the most stable validated F₀.₅ wins, not just the one with the highest mean score.

---

## 16. Final Inference

A frozen, deterministic pipeline. 
- **Flow:** Load frozen config → Multi-view normalization → Retrieve base + conditional candidates → Dedup/Cap → Build features → Score → Aggregate → Apply locked threshold policy → Output TSV.
- **Validation:** Internal ID uniqueness/format checks + official `validate_submission.py --check-ids`.

---

## 17. Leakage Controls

- No entity crosses training folds.
- Target labels are strictly isolated from feature engineering.
- Candidate artifacts do not store fold-dependent labels.
- Hard negatives and meta-models rely exclusively on cross-fitted OOF scoring.
- Re-optimization requirement ensures thresholds never adapt to a mismatched score distribution.

---

## 18. Resource Constraints

- Memory usage is tracked via CSR sparse matrices, batching, and top-K limits.
- Dense/Cross-encoder models (if used) strictly enforce ≤8B parameters and MIT/Apache 2.0 licensing.

---

## 19. Reproducibility

- Exact environment locking (requirements.txt), fixed library seeds, fixed thread counts.
- Immutable experiment manifests tracking git hashes, parameters, and artifact checksums.
