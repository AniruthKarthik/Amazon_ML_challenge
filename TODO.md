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

**Objective:** Measure whether base lexical retrieval recovers ground-truth candidates; keep retrieval misses separate from ranking failures measured after Phase 8.
- [x] Implement Micro candidate recall and Entity complete/any-hit recall.
- [x] Implement Oracle entity-level F₀.₅ (assumes perfect pair classification).
- [x] Implement retrieval diversity, incremental unions, and **Unique GT Recovered** per channel.
- [x] Implement **Retrieval miss rate** = `(GT pairs absent from candidate set) / (all GT pairs)`.
- [x] Implement reproducible sampling and taxonomy for up to 1000 genuine retrieval misses.
- [ ] Execute the full-training benchmark in Colab, produce the quantitative report, and make the measured lexical/Word TF-IDF/Dense decision.
- [ ] After Phases 7/8, calculate ranking failures from OOF LightGBM scores, including top-1, Recall@5/10, and false-positive competition; no Phase 4 ranking rate is claimed.
**Metrics:** Oracle F₀.₅, Unique GT Recovered per channel, Retrieval Miss Rate; post-Phase-8 OOF Ranking Failure Rate.
**Acceptance Criteria:** Explicit quantitative error metric report produced.
**Decision Gate:** 
  - If token/reordering misses dominate retrieval errors: trigger Phase 5 (Word TF-IDF).
  - If semantic/linguistic misses dominate retrieval errors: trigger Phase 13 (Dense Retrieval).
  - (Note: Phase 5 and Phase 13 are independent conditional branches).
  - After Phase 8, if quantitative OOF ranking failure is substantial: mark Phase 14 (Cross-encoder) eligible for evaluation after Phase 10.

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
- [x] Implement component-disjoint OOF raw LightGBM pair predictions.
- [x] Implement nested cross-fitted isotonic calibration without held-out-label reuse.
- [x] Implement label-free OOF score artifact and multi-positive-aware ranking diagnostics.
- [ ] Run full-training OOF predictions and report pair F₀.₅ and retrieved-pair ranking failure/Recall@5/10 in Colab.
**Metrics:** OOF Pair F₀.₅.
**Acceptance Criteria:** No entity crosses fold boundaries during generation.
**Next Decision:** Proceed to Phase 9.

## Phase 9: Entity-Level Decision + Exact F₀.₅

**Objective:** Aggregate pair scores and compute the exact competition metric.
- [x] Implement entity-level score aggregation (max, second, counts, gaps, distribution).
- [x] Implement exact macro F₀.₅, including empty-list singleton scoring.
- [x] Define Phase 10 inputs: top-1 entity confidence, top-2 score/gap, zero-candidate state, and all candidate pair scores; preserve multi-positive truth for exact set scoring. The deterministic rule uses `<` for entity rejection and `>=` for gap/pair acceptance.
- [x] Implement same-threshold Raw-vs-Calibrated OOF macro F₀.₅ comparison.
- [ ] Run the full OOF entity comparison in Colab; keep calibration only if measured macro F₀.₅ or threshold stability improves.
**Decision Gate:** Keep Isotonic Calibration ONLY if it improves Entity F₀.₅ or threshold stability.

## Phase 10: Threshold Optimization & Robust Policy Selection

**Objective:** Jointly optimize decision thresholds and operationally define robust policy against country shift.
- [x] Implement the pure deterministic C rule: sort unique candidates by descending score/ascending ID; zero candidates → empty; `top1 < entity_threshold` → empty; else `gap >= gap_threshold` → top-1; else all scores `>= pair_threshold`, or top-1 fallback when none pass. One candidate has decision gap `+inf`; tied top scores have gap zero. Output is a candidate subset.
- [x] Implement comparison baselines A (pair threshold only) and B (entity gate + pair threshold + top-1 fallback), separate from C. Independently configure all three thresholds and serialize the frozen policy.
- [x] Implement OOF-only joint threshold search and cross-fitted fold evaluation with exact entity F₀.₅; report overall, fold-wise, mean/std/worst, singleton precision/false-positive singleton rate, average predicted count, and empty/top-1/multi percentages. Select C only if its heldout F₀.₅ and worst-fold robustness justify it.
- [x] Implement threshold sensitivity (caller-supplied ± perturbation) and leave-one-country-out diagnostics; use synthetic tests locally.
- [ ] Run the full OOF search, robustness diagnostics, and frozen configuration/report generation in Colab. The all-OOF frozen-threshold fit is exploratory, distinct from cross-fitted model-selection metrics.
- [ ] If country shift causes meaningful degradation, execute threshold remediation rule:
  - Check if degradation is driven by threshold policy.
  - Evaluate one global threshold, country/source-specific thresholds (only if labeled evidence supports), and a conservative OOF-stable global threshold (only if validation confirms better worst-case robustness). These geographic variants are not decision Policies A/B/C.
  - Select policy via cross-fitted validation tradeoff (mean score vs variance vs worst-case).
**Metrics:** Entity F₀.₅ mean, worst-case, variance, stability dispersion.
**Decision Gate:** Lock the robust threshold policy. Test distribution may NOT determine thresholds. No automatic test-time overrides allowed.
**Critical Rule:** Any change to upstream components invalidates these thresholds; must re-optimize.

## Phase 11: Hard-Negative Mining

**Objective:** Improve pair model discrimination through cross-fitted negative mining.
- [x] Implement nested mining: for each outer heldout fold, fit inner models only on other folds; select high-scoring known negatives from inner OOF scores, capped per S1 with deterministic ties. The mining cutoff is an explicit training-only experiment setting, never fitted from the heldout fold or test data.
- [x] Implement weighted retraining of the outer LightGBM scorer on those mined training negatives and emit raw OOF scores, fold diagnostics, and mined-row audit indices. Use fixed boosting rounds; heldout labels influence diagnostics only.
- [x] Implement an identical-grid raw OOF baseline-vs-mined Phase 9/10 comparison hook; synthetic tests verify outer-label isolation and cross-fitted entity F₀.₅ comparison.
- [ ] In Colab, run the full mined OOF experiment and mandatory Phase 8 → 9 → 10 re-evaluation. If calibration is retained, regenerate its nested OOF pathway before any acceptance decision; do not reuse the old calibrator.
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

**Objective:** Address persistent pairwise ranking errors measured after Phase 8 using OOF LightGBM scores.
- [ ] If the post-Phase-8 OOF ranking failure rate is substantial, add OOF cross-encoder score as feature to LightGBM.
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
