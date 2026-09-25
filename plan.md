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
- Measure explicit **Retrieval miss rate** and **Ranking failure rate** to identify the bottleneck.
- *Decision:* Quantitative results explicitly drive independent conditional branches (Phase 5, Phase 13, Phase 14).
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
- Compare Raw vs Calibrated pathways. Reject calibration if it provides no F₀.₅ or stability win.
- *Dependency:* Phase 8 OOF predictions.

## Phase 10: Threshold Optimization & Robust Policy Selection
- Jointly optimize pair, entity, and gap thresholds.
- Simulate country/domain shifts. Evaluate threshold policies via trade-off analysis (mean F₀.₅, worst-case fold/country, variance).
- Compare Global (A), Country-specific (B), and Conservative OOF-stable (C) policies if degradation is observed.
- *Decision:* Lock the robust policy. Unlabeled test data may NOT determine thresholds.
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
- If Phase 4 explicit ranking failure rate was substantial, evaluate Cross-encoders.
- **MANDATORY RE-OPTIMIZATION:** Re-run Phase 7 (Pair Model) → 8 → 9 → 10.
- *Decision:* Keep ONLY if final entity F₀.₅ materially improves.
- *Dependency:* Phase 4 quantitative ranking failure rate.

## Phase 15: Final Ablations
- Evaluate final ensembles (LightGBM vs CatBoost/LR Stacker).
- Lock the final experiment configuration based on stable OOF entity F₀.₅.
- *Dependency:* All accepted prior phases.

## Phase 16: Frozen Inference + Output Validation
- Run the locked policy on test TSVs.
- Execute strict internal ID/format validators and official `validate_submission.py`.
- *Dependency:* Phase 15 locked configuration.
