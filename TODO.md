# Entity Resolution Implementation TODO

> **Objective:** MAXIMIZE true held-out macro F₀.₅ accuracy at the ENTITY level, under the actual competition constraints.
> **Critical Rule:** ANY material change to upstream components (retrieval channels, pair model, features) REQUIRES a full downstream re-optimization (regenerate OOF → rebuild meta-features → retrain meta-model → reoptimize thresholds → compute true final F₀.₅).
> **Compute Gate Rule:** Before running a full downstream re-optimization loop (Phase 8→10) for any candidate change in Phase 11/12/13/14, first check a cheap proxy signal (pair-level AUC or OOF pair F₀.₅ on a fixed sample) against the current locked baseline. Only proceed to the full mandatory re-optimization loop if the proxy signal shows improvement above a pre-defined minimum margin. Log proxy-reject decisions in the ablation report for auditability.

---

## Phase 1: Data Contract + Integrity

**Objective:** Ensure strict adherence to competition data rules, schema integrity, and graph validation.
- [x] Implement TSV loader with `sep="\t"`, preserving NAs as empty strings.
- [x] Enforce schema validation (IDs must match prefix S1/S2/S3, business_name cannot be missing).
- [x] Validate `train_ground_truth.tsv` (coverage of S1 entities, uniqueness of targets, valid IDs).
- [ ] Analyze connected components in the training bipartite graph to inform folding.
  - Graph analysis and metric reporting are implemented and tested on synthetic TSVs; full-training execution is deferred to Colab.
**Metrics:** Record counts, singleton ratio, graph component sizes.
**Acceptance Criteria:** Zero silent data drops; strict TSV compliance.
**Next Decision:** Proceed to Phase 2.

## Phase 2: Normalization

**Objective:** Produce multi-view text representations safely.
- [x] Implement `business_name_clean`: NFKC + casefold + safe punctuation.
- [x] Implement `business_name_folded`: accent removal.
- [x] Implement `business_name_core`: legal suffix extraction.
- [x] Implement `business_address_clean` and `business_address_alias` formatting.
- [x] Write unit tests for idempotence, Unicode edge cases, and empty strings.
- [x] Verify raw representations are fully preserved in the output struct.
  - Collision-rate calculation is implemented; dataset-wide rates and timing are deferred to Colab.
**Metrics:** Collision rate per view; execution time.
**Acceptance Criteria:** Tests pass, raw data preserved, multi-view available.
**Next Decision:** Proceed to Phase 3.

## Phase 3: Exact/Lexical Retrieval

**Objective:** Build base candidate generation from exact and character-level channels.
- [x] Implement Exact Normalized Name and Exact Core Name channels.
- [x] Implement Character TF-IDF Name and Character TF-IDF Address KNN.
- [x] Implement Rare-token retrieval index.
- [x] Union candidates, deduplicate, and record provenance per channel.
  - Per-channel counts and bounded generation are tested on synthetic records; full-training candidate volumes are deferred to Colab.
**Metrics:** Candidate volume per channel.
**Acceptance Criteria:** Base channels retrieve candidates without exploding memory.
**Next Decision:** Proceed to Phase 4.

## Phase 4: Retrieval Benchmark + Error Taxonomy

**Objective:** Measure whether base lexical retrieval recovers ground-truth candidates, and explicitly quantify retrieval vs ranking failures.
- [x] Implement Micro candidate recall and Entity complete/any-hit recall.
- [x] Implement Oracle entity-level F₀.₅ (assumes perfect pair classification).
- [x] Implement retrieval diversity, incremental unions, and **Unique GT Recovered** per channel.
- [x] Implement **Retrieval miss rate** = `(GT pairs absent from candidate set) / (all GT pairs)`.
- [x] Implement reproducible sampling and taxonomy for up to 1000 genuine retrieval misses.
- [ ] Execute the full-training benchmark in Colab, produce the quantitative report, and make the measured lexical/Word TF-IDF/Dense decision.
- [ ] After Phases 7/8, calculate ranking failures from OOF LightGBM scores, including top-1, Recall@5/10, and false-positive competition; no Phase 4 ranking rate is claimed.
**Metrics:** Oracle F₀.₅, Unique GT Recovered per channel, Retrieval Miss Rate, Ranking Failure Rate.
**Acceptance Criteria:** Explicit quantitative error metric report produced.
**Decision Gate:** 
  - If token/reordering misses dominate retrieval errors: trigger Phase 5 (Word TF-IDF).
  - If semantic/linguistic misses dominate retrieval errors: trigger Phase 13 (Dense Retrieval).
  - (Note: Phase 5 and Phase 13 are independent conditional branches).
  - If quantitative ranking failure rate is substantial: mark Phase 14 (Cross-encoder) eligible for evaluation after Phase 10.

## Phase 5: Word TF-IDF Retrieval Experiment (Independent Conditional)

**Objective:** Test if word-level TF-IDF recovers reordered/multi-token targets missed by char TF-IDF.
- [ ] Implement Word TF-IDF KNN channel.
- [ ] Measure **Unique GT Recovered** specifically against the base Char TF-IDF union.
- [ ] Measure incremental Oracle F₀.₅ and candidate explosion.
**Decision Gate:** Keep if it measurably improves Oracle F₀.₅ and Unique GT Recovered without excessive memory cost.

## Phase 6: Pair Features

**Objective:** Generate tabular features for scoring candidate pairs.
- [x] Implement name/address edit distances (Levenshtein, Jaro-Winkler).
- [x] Implement token overlap, TF-IDF cosine, and numeric address overlap.
- [x] Implement interaction features (country flags, missingness context).
- [x] Append retrieval provenance (ranks, scores, channel counts).
  - Feature functions and the Phase 3→6 interface are tested on synthetic pairs. Full-data feature generation and build-time measurement are deferred to Colab after retrieval-channel selection.
**Metrics:** Feature build time, NaN/Infinity assertions.
**Acceptance Criteria:** Features are pure functions of candidate pairs without fold leakage.
**Next Decision:** Proceed to Phase 7.

## Phase 7: LightGBM Baseline

**Objective:** Train primary pair scorer using structural/lexical features.
- [x] Implement cross-validation folds using connected components from Phase 1.
- [x] Implement deterministic LightGBM pair-scorer training for one component-disjoint fold.
- [x] Implement pair-level precision/recall/AUC diagnostics (not entity metrics).
- [ ] Fit on the full training candidates and inspect memory use and learning curves in Colab.
**Acceptance Criteria:** Model fits within memory constraints, stable learning curves.
**Next Decision:** Proceed to Phase 8.

## Phase 8: OOF Pair Predictions

**Objective:** Generate leakage-safe predictions to drive entity-level decisions.
- [ ] Generate out-of-fold (OOF) predictions for all train candidates.
- [ ] Generate OOF predictions for raw LightGBM scores.
- [ ] Fit isotonic calibration on OOF scores and generate calibrated OOF scores.
**Metrics:** OOF Pair F₀.₅.
**Acceptance Criteria:** No entity crosses fold boundaries during generation.
**Next Decision:** Proceed to Phase 9.

## Phase 9: Entity-Level Decision + Exact F₀.₅

**Objective:** Aggregate pair scores and compute the exact competition metric.
- [ ] Implement entity-level aggregation (max score, second max, counts, score gaps).
- [ ] Implement exact macro F₀.₅ computation per competition rules.
- [ ] Compare Raw LightGBM vs Isotonic Calibrated scores on final Macro F₀.₅.
**Decision Gate:** Keep Isotonic Calibration ONLY if it improves Entity F₀.₅ or threshold stability.

## Phase 10: Threshold Optimization & Robust Policy Selection

**Objective:** Jointly optimize decision thresholds and operationally define robust policy against country shift.
- [ ] Optimize pair threshold, entity threshold, and gap thresholds jointly on OOF entity F₀.₅.
- [ ] Run threshold stability analysis (fold dispersion, ± sensitivity).
- [ ] Simulate country/domain shift (e.g., leave-one-country-out). Measure mean F₀.₅, worst-fold/country F₀.₅, and variance.
- [ ] If country shift causes meaningful degradation, execute threshold remediation rule:
  - Check if degradation is driven by threshold policy.
  - Evaluate Policy A: Global Threshold.
  - Evaluate Policy B: Country/source-specific thresholds (only if labeled validation evidence supports).
  - Evaluate Policy C: Conservative global threshold from OOF-stable range (only if validation confirms better worst-case robustness).
  - Select policy via cross-fitted validation tradeoff (mean score vs variance vs worst-case).
**Metrics:** Entity F₀.₅ mean, worst-case, variance, stability dispersion.
**Decision Gate:** Lock the robust threshold policy. Test distribution may NOT determine thresholds. No automatic test-time overrides allowed.
**Critical Rule:** Any change to upstream components invalidates these thresholds; must re-optimize.

## Phase 11: Hard-Negative Mining

**Objective:** Improve pair model discrimination through cross-fitted negative mining.
- [ ] Mine false positives using Fold A model to generate negatives for Fold B.
- [ ] Retrain pair scorer on augmented dataset.
- [ ] **MANDATORY RE-EVALUATION:** Re-run Phase 8 (OOF) → Phase 9 (Aggregation) → Phase 10 (Thresholds).
**Decision Gate:** Keep if FINAL cross-fitted entity-level F₀.₅ improves.

## Phase 12: Entity-Level Decision / Meta Model

**Objective:** Predict the final set decision (zero, one, many) using complete candidate distributions.
- [ ] Train a meta-model on OOF entity meta-features (score gap, max score, candidate count, distribution stats).
- [ ] Compare empirically:
  A. Pair score + deterministic threshold
  B. Pair score + top-1 / score-gap logic
  C. Pair score + entity-level meta-model
- [ ] **MANDATORY RE-EVALUATION:** Re-run Phase 10 (Thresholds) on meta-model output.
**Decision Gate:** Keep meta-model ONLY if Option C improves cross-fitted entity F₀.₅ over deterministic logic.

## Phase 13: Dense Bi-Encoder Retrieval (Independent Conditional)

**Objective:** Address persistent semantic/linguistic retrieval misses from Phase 4.
- [ ] Verify license/size for dense bi-encoder.
- [ ] Add dense retrieval candidates to union. Evaluate independently or together with Phase 5.
- [ ] **MANDATORY RE-EVALUATION:** Re-run Phase 6 (Features) → Phase 7 (Pair Model) → Phase 8 (OOF) → Phase 9 (Aggregation) → Phase 10 (Thresholds).
**Decision Gate:** Keep dense retrieval ONLY if the additional candidates produce a net improvement in the FINAL cross-fitted entity-level macro F₀.₅ that justifies compute cost, not just candidate recall.

## Phase 14: Cross-Encoder Pair Scoring (Conditional)

**Objective:** Address persistent pairwise ranking errors explicitly measured in Phase 4.
- [ ] If Phase 4 ranking failure rate was substantial, add OOF cross-encoder score as feature to LightGBM.
- [ ] **MANDATORY RE-EVALUATION:** Re-run Phase 7 (Pair Model) → Phase 8 (OOF) → Phase 9 (Aggregation) → Phase 10 (Thresholds).
**Decision Gate:** Keep ONLY if final entity F₀.₅ materially improves.

## Phase 15: Final Ablations

**Objective:** Measure and lock the optimal ensemble/configuration.
- [ ] Compare all accepted modules (Raw vs Calibrated, Base vs Base+Word, LightGBM vs LightGBM+Meta-model).
- [ ] Compare LightGBM against tabular ensembles (CatBoost, LR Stacker) using identical OOF folds and identical feature set.
- [ ] Ensure full downstream re-optimization loop (Phase 8→10) is run for each candidate final model before comparison.
- [ ] Generate comprehensive ablation report: OOF entity F₀.₅ mean, variance, worst-fold, per-country breakdown for every candidate.
**Acceptance Criteria:** Lock the single most robust architecture configuration based on stable OOF entity F₀.₅, not just mean.

## Phase 16: Frozen Inference + Output Validation

**Objective:** Execute the locked policy on test data without label leakage.
- [ ] Apply frozen transformations, retrieval, scoring, and locked thresholds to test set.
- [ ] Run internal ID/format validators and official validator with `--check-ids`.
**Acceptance Criteria:** Identical deterministic outputs across runs, validators PASS.
