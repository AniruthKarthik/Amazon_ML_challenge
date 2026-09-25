# Implementation Plan — Business Entity Resolution

> **Objective:** MAXIMIZE true held-out macro F₀.₅ accuracy at the ENTITY level.
> **Principle:** MEASURE → IDENTIFY BOTTLENECK → ADD COMPLEXITY ONLY IF JUSTIFIED.

This plan executes the architecture through evidence-driven gates. No downstream complexity is added until the upstream bottleneck is proven.

---

## Phase 1: Data Contract + Integrity
- Validate input schemas, unique IDs, and ground-truth structures.
- Compute bipartite graph connected components for robust out-of-fold (OOF) splitting.
- *Dependency:* Raw TSVs.

## Phase 2: Normalization
- Construct deterministic multi-view normalizations (raw, clean, folded, core, alias).
- Verify idempotence and Unicode safety.
- *Dependency:* Phase 1 data contract.

## Phase 3: Exact/Lexical Retrieval
- Implement the baseline candidate generation channels (exact name, exact core, char TF-IDF, rare-token).
- *Dependency:* Phase 2 normalized views.

## Phase 4: Retrieval Benchmark & Bottleneck Analysis
- Evaluate Phase 3 channels using oracle entity-level F₀.₅, candidate recall, and channel diversity.
- Crucially, measure **Unique GT Recovered** to test if channels actually add new true matches.
- Measure explicit **Retrieval miss rate** here. Defer **Ranking failure rate** until Phase 8 provides OOF LightGBM scores; report top-1, Recall@5/10, and false-positive competition separately from retrieval misses.
- *Decision:* Quantitative retrieval results drive independent Phase 5 and Phase 13 branches; post-Phase-8 OOF ranking results gate Phase 14.
- *Dependency:* Phase 3 candidates + Phase 1 folds.

## Phase 5: Word TF-IDF Retrieval Experiment (Independent Conditional)
- Evaluate word-level TF-IDF *if* Phase 4 indicates token/reordering misses dominate.
- *Decision:* Keep only if it improves oracle F₀.₅ and Unique GT Recovered.
- *Dependency:* Phase 4 bottleneck analysis (independent of Phase 13).

## Phase 6: Pair Features
- Build tabular features (string distances, interactions, provenance) for the retrieved candidate pairs.
- *Dependency:* Finalized lexical candidate set from Phase 4/5/13.

## Phase 7: LightGBM Baseline
- Train LightGBM pair scorer using Phase 6 features.
- Evaluate using pair-level diagnostic metrics.
- *Dependency:* Phase 6 features + Phase 1 folds.

## Phase 8: OOF Pair Predictions
- Generate cross-fitted OOF raw score predictions.
- Fit and generate isotonic-calibrated OOF predictions.
- *Dependency:* Phase 7 models.

## Phase 9: Entity-Level Decision + Exact F₀.₅
- Group OOF pair predictions by source entity.
- Calculate the exact macro F₀.₅.
- Summarize top-1 score, second score, and gap for Phase 10's deterministic gate: reject below entity threshold; select top-1 at or above gap threshold; otherwise keep all pair scores at or above pair threshold, with top-1 fallback. One-candidate gap is `+inf` for decisions; tied top scores have gap zero.
- Compare Raw vs Calibrated pathways. Reject calibration if it provides no F₀.₅ or stability win.
- Include zero-candidate entities and multiple valid S1 links; retain exact empty-truth singleton scoring.
- *Dependency:* Phase 8 OOF predictions.

## Phase 10: Threshold Optimization & Robust Policy Selection
- Freeze a reusable deterministic rule: deduplicate candidate IDs, sort by descending pair score then ascending ID; empty candidates or `top1_score < entity_threshold` yield empty; otherwise `gap >= gap_threshold` yields top-1; otherwise retain all candidates with score `>= pair_threshold`, falling back to top-1 if none pass. One candidate has `gap = +inf`, tied top scores have gap zero. Output remains a candidate subset.
- Jointly search independent pair/entity/gap thresholds on leakage-safe training OOF predictions and exact entity macro F₀.₅. Compare decision A (pair-only), B (entity gate + pair gate + top-1 fallback), and C (full score-gap rule); never assume C wins.
- Select fold thresholds on other OOF folds, report heldout overall/fold mean/std/worst, singleton precision/false-positive singleton rate, mean prediction count, and empty/top-1/multi percentages. Fit frozen selected-family thresholds on all OOF data separately; serialize configuration and search report. No test labels or distribution for selection.
- Simulate country/domain shifts and caller-specified ± threshold perturbations. If shift degrades results, compare global, sufficiently supported country/source-specific, and conservative OOF-stable global *geographic* policies; these are separate from decision A/B/C.
- *Decision:* Lock the robust policy. Unlabeled test data may NOT determine thresholds.
- *Compute gate:* Implement and unit-test locally; execute the full OOF search and freeze measured thresholds in Colab before Phase 11.
- *Dependency:* Phase 9 entity aggregations.

## Phase 11: Hard-Negative Mining
- Use cross-fitted OOF false positives to retrain the pair model.
- **MANDATORY RE-OPTIMIZATION:** Re-run Phase 8 → 9 → 10.
- *Decision:* Keep only if the updated entity F₀.₅ improves.
- *Dependency:* Phase 8 OOF errors.

## Phase 12: Entity-Level Decision / Meta Model
- Train an explicit meta-model to make the set-decision (zero, one, many) using OOF meta-features.
- **MANDATORY RE-OPTIMIZATION:** Re-run Phase 10 Thresholds on meta-model outputs.
- *Decision:* Compare empirical performance of deterministic rules vs meta-model. Keep only if entity F₀.₅ improves.
- *Dependency:* Phase 9 meta-features.

## Phase 13: Dense Bi-Encoder Retrieval (Independent Conditional)
- Evaluate Dense Retrieval *if* Phase 4 indicates semantic/linguistic shifts dominate.
- **MANDATORY RE-OPTIMIZATION:** Re-run Phase 6 (Features) → 7 (Pair Model) → 8 → 9 → 10.
- *Decision:* Keep dense retrieval ONLY if the additional candidates produce a net improvement in the FINAL cross-fitted entity-level macro F₀.₅ that justifies compute cost.
- *Dependency:* Phase 4 bottleneck analysis (independent of Phase 5).

## Phase 14: Cross-Encoder Pair Scoring (Conditional)
- If the post-Phase-8 OOF ranking failure rate is substantial, evaluate Cross-encoders.
- **MANDATORY RE-OPTIMIZATION:** Re-run Phase 7 (Pair Model) → 8 → 9 → 10.
- *Decision:* Keep ONLY if final entity F₀.₅ materially improves.
- *Dependency:* Post-Phase-8 quantitative OOF ranking failure rate.

## Phase 15: Final Ablations
- Evaluate final ensembles (LightGBM vs CatBoost/LR Stacker).
- Lock the final experiment configuration based on stable OOF entity F₀.₅.
- *Dependency:* All accepted prior phases.

## Phase 16: Frozen Inference + Output Validation
- Run the locked policy on test TSVs.
- Execute strict internal ID/format validators and official `validate_submission.py`.
- *Dependency:* Phase 15 locked configuration.
