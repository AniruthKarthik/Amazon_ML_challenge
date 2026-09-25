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
- **Deterministic baseline:** Group every OOF candidate by S1, sort by descending score then ascending candidate ID, and use the highest pair score as entity confidence. Deduplicate candidate IDs by their maximum score; output is always a subset of candidates. A missing candidate set produces an empty prediction.
- **Decision Policies (Phase 10):** A = all candidates with score `>= pair_threshold`; B = empty if `top1_score < entity_threshold`, otherwise all candidates with score `>= pair_threshold`, falling back to top-1 if none pass; C = B's entity gate, then top-1 only when `top1_score - top2_score >= gap_threshold`, otherwise B's pair gate and fallback. For one candidate the decision gap is `+inf`; tied top scores have gap zero. All three thresholds are independently configurable. The entity gate rejects strictly below threshold, while pair and gap gates accept equality.
- **Evaluation:** Compare empirically:
  - Decision Policies A/B/C on leakage-safe OOF pair scores.
  - Optional entity-level meta-model only after the deterministic baseline is measured.
- **Phase 12 cardinality rule:** A CPU-friendly three-class meta-model uses only candidate-score aggregations of OOF pair predictions. `ZERO` returns empty; `ONE` returns top-1; `MANY` returns all candidates with pair score `>= multi_pair_threshold`, falling back to top-1 if none pass. `MANY` never forces a second match. Tune the multi threshold on inner OOF entity predictions with exact macro F₀.₅; compare outer-fold meta predictions to the locked Phase 10 deterministic family on identical folds. Keep only for a cross-fitted F₀.₅ gain without a material worst-fold or dispersion loss. Inference uses the frozen model and threshold unchanged.
- **Justification:** The meta-model is kept only if it improves cross-fitted entity-level macro F₀.₅. If deterministic rules perform equally well, the simpler approach is retained.

---

## 10. Threshold Optimization & Robustness

Search Policies A/B/C over the supplied pair/entity/gap threshold grid against exact entity-level macro F₀.₅. For each heldout OOF fold, select thresholds on other folds only; report heldout aggregate and fold scores, mean, standard deviation, worst fold, singleton precision and false-positive singleton rate, average predicted count, and empty/top-1/multi-match percentages. Compare families by worst heldout fold, then overall heldout macro F₀.₅, lower fold dispersion, and simpler policy on exact ties. Policy C is not presumed best. Refit the chosen family's frozen thresholds on all training OOF rows, reporting this exploratory fit separately from the cross-fitted comparison. The frozen policy/configuration must use the same pure decision function at inference.
- **Operational Robustness:** Report caller-specified ± threshold sensitivity and leave-one-country-out OOF performance where labeled sample size is sufficient. Do not invent cutoffs or label the all-OOF fit an unbiased validation score.
- **Country Shift Remediation Rule:** If domain shift causes meaningful degradation, identify if it stems from the threshold policy. If yes, compare one global threshold, country/source-specific thresholds with sufficient labeled evidence, and a conservative OOF-stable global threshold. These geographic variants are distinct from Decision Policies A/B/C.
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
- **Provisional CPU baseline:** The lexical-only one-command runner may emit an explicitly acknowledged, sampled-training output for operational use before Phase 15. It records input/model/output provenance and does not claim final model selection or full-training OOF accuracy.

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
- The provisional CPU runner resolves `threads=auto` to the process's available logical cores for parallel exact/character retrieval and LightGBM operations; serial store/rare-token stages are not run concurrently on a 16 GB machine.
- Dense/Cross-encoder models (if used) strictly enforce ≤8B parameters and MIT/Apache 2.0 licensing.

---

## 19. Reproducibility

- Exact environment locking (requirements.txt), fixed library seeds, fixed thread counts.
- Immutable experiment manifests tracking git hashes, parameters, and artifact checksums.
