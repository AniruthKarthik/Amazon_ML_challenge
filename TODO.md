# TODO — Amazon ML Challenge 2026: Business Entity Resolution

> Derived strictly from `README.md` and `plan.md`.  
> Nothing is marked `[x]` — neither document states any implementation is completed.  
> The plan explicitly says: *"no pipeline or model code is implemented by this document"* and the repository assessment confirms all code files are empty/placeholder.

---

## Phase 0: Plan Acceptance & Setup

- [ ] Review and accept `plan.md` as the implementation source of truth
- [ ] Confirm team agreement on the scope, constraints, and phase order
- [ ] Verify `.gitignore` excludes `dataset/` (confirmed present, but double-check coverage)
- [ ] Clean stale `.gitattributes` LFS path (`student_resource/dataset/**`) if it causes confusion
- [ ] Remove or replace `chimera_submission/code/business_entity_resolution/src/notebook.ipynb` (zero-byte invalid file)

---

## Phase 1: Repository Audit + Data Profiling

**Objective:** Full schema/ID/graph integrity checks; distributions, missingness, lengths/scripts, collisions, label cardinality; benchmark representative reads.

### 1.1 Data Loading & Contract Validation

- [ ] Implement TSV loader using `pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False, encoding="utf-8")`
- [ ] Validate headers before expensive loading — reject extra/missing columns
- [ ] Validate row widths (exactly 4 TSV fields per source row)
- [ ] Validate UTF-8 encoding on all files
- [ ] Validate `entity_id` prefix matches source file (`S1-` in source1, `S2-` in source2, `S3-` in source3)
- [ ] Validate `entity_id` uniqueness within each source file
- [ ] Validate `entity_id` uniqueness across S2 and S3 (target sources)
- [ ] Verify no blank/null `entity_id` values
- [ ] Verify train ground truth has exactly one row for every train S1 ID (row count = 2,206,821)
- [ ] Verify train ground truth has no unknown S1 IDs
- [ ] Verify no repeated link IDs within a single ground-truth row
- [ ] Verify all S2/S3 IDs in ground truth exist in their respective source files
- [ ] Confirm dataset row counts match plan (train: S1=2,206,821; S2=5,034,616; S3=5,285,603; test: S1=1,732,544; S2=4,887,273; S3=5,082,316)

### 1.2 Data Profiling

- [ ] Record row counts, null/blank counts, distinct ID counts per file
- [ ] Record country counts per file and confirm against plan values
- [ ] Confirm missing-address counts (train S2: 168,967; train S3: 175,916; test S2: 129,408; test S3: 136,098)
- [ ] Confirm zero missing names, zero missing country across all files
- [ ] Profile business_name lengths, token counts, character distributions
- [ ] Profile business_address lengths, token counts, character distributions
- [ ] Profile script usage (Latin, Devanagari, French diacritics, etc.)
- [ ] Compute and record file hashes in a data manifest
- [ ] Benchmark representative TSV read times for memory/time budgeting

### 1.3 Ground Truth Analysis

- [ ] Parse ground-truth `matched_entity_ids` into sets and compute label statistics
- [ ] Confirm 123,247 singletons (5.5848% of train S1)
- [ ] Confirm 7,638,365 total links
- [ ] Confirm mean 3.461 links per S1
- [ ] Confirm maximum 11 links per S1
- [ ] Build bipartite graph of S1-to-S2/S3 truth links
- [ ] Compute connected components and verify graph properties
- [ ] Check whether each S2/S3 target belongs to at most one S1 (document actual properties)
- [ ] Verify global ID uniqueness and exact ID-set equality between ground truth and source files
- [ ] Compute comparison space sizes (~22.8T train pairs, ~17.3T test pairs) as a sanity check

### 1.4 Initial Tests

- [ ] Write unit tests for TSV loading contract checks
- [ ] Write tests for ground-truth parsing and set explosion

**Acceptance gate:** All TSVs load; exact schemas/coverage/uniqueness known; no unreported integrity error.

---

## Phase 2: Preprocessing (Multi-View Normalization)

**Objective:** Implement deterministic, tested, Unicode-safe, multi-view name/address/country normalization.

### 2.1 Shared Normalization Operations

- [ ] Implement missing-value handling: convert missing input to empty string + set missing flag
- [ ] Implement Unicode NFKC normalization
- [ ] Implement Unicode `casefold()` (not ASCII-only lowercasing)
- [ ] Implement typographic apostrophe/dash and punctuation normalization
- [ ] Implement `&` alias handling (preserve raw, create alias view)
- [ ] Implement repeated whitespace collapse and trim
- [ ] Implement accent-preserving and accent-folded forms (Unicode decomposition, remove combining marks only in folded view)
- [ ] Ensure non-Latin text is never fully transliterated to ASCII as the sole representation

### 2.2 Name Views

- [ ] Implement `business_name_raw`: untouched input passthrough
- [ ] Implement `business_name_clean`: NFKC + case-fold + conservative punctuation/space normalization
- [ ] Implement `business_name_folded`: accent-folded clean name
- [ ] Implement `business_name_core`: reviewed configurable legal/corporate suffix removal from terminal positions (e.g., Ltd, Limited, Corp, Corporation, Pvt, Private)
- [ ] Retain flags and counts describing removed suffixes
- [ ] Derive token multiset and character n-gram views from clean/folded forms

### 2.3 Address Views

- [ ] Implement `business_address_raw`: untouched input passthrough
- [ ] Implement `business_address_clean`: NFKC + case-fold + conservative punctuation/space normalization
- [ ] Implement `business_address_folded`: accent-folded companion
- [ ] Implement `business_address_alias`: unambiguous reviewed address aliases + stable number formatting
- [ ] Extract token sets: alphabetic, numeric, and alphanumeric tokens (generic, no country-format assumptions)
- [ ] Ensure ambiguous tokens like `st` are NOT blindly rewritten — preserve both original and expanded hypothesis if useful

### 2.4 Country Normalization

- [ ] Normalize country with NFKC, case-folding, whitespace cleanup
- [ ] Treat country as open-set string — no hard-coded vocabulary or one-hot encoding
- [ ] Set `country_missing` flag where applicable

### 2.5 Stable Internal Schema

- [ ] Implement the normalized entity record schema per plan §4.2 (all fields listed)
- [ ] Canonicalize empty fields to one internal representation (nullable string or `""`)
- [ ] Ensure literal text `"None"` never becomes an output value
- [ ] Preserve input order for output assembly; use `entity_id` as stable join key

### 2.6 Caching

- [ ] Cache normalized entity tables once per normalization version in columnar format
- [ ] Partition by split/source
- [ ] Checksum cached results against raw input

### 2.7 Normalization Tests

- [ ] Write golden unit tests for idempotence
- [ ] Test blank/empty inputs
- [ ] Test composed vs. decomposed accents
- [ ] Test French diacritics
- [ ] Test Devanagari text
- [ ] Test punctuation normalization
- [ ] Test ampersand handling
- [ ] Test apostrophe handling
- [ ] Test digit-letter tokens
- [ ] Test repeated whitespace
- [ ] Test ambiguous abbreviations
- [ ] Verify raw columns are byte-for-byte recoverable from loaded values
- [ ] Generate collision-rate report for each normalized view

**Acceptance gate:** Deterministic/idempotent tests pass; raw preserved; Unicode cases pass.

---

## Phase 3: Baseline Candidate Generation (Blocking)

**Objective:** Exact clean/core, char-name/address TF-IDF, rare tokens; batching, provenance, union/caps.

### 3.1 Exact Retrieval Channels

- [ ] Implement **exact normalized name** channel: inverted index on `business_name_clean`
- [ ] Implement **exact core name** channel: inverted index on `business_name_core` with separate cap for very common cores

### 3.2 Fuzzy Retrieval Channels

- [ ] Implement **character n-gram TF-IDF name KNN**: primary fuzzy retriever over clean/folded names
  - [ ] Tune `char` vs `char_wb` analyzer
  - [ ] Tune n-gram ranges
  - [ ] Configure `min_df`, `max_df`, sublinear TF, dtype (`float32`)
  - [ ] Store as CSR sparse matrices; never densify
- [ ] Implement **character n-gram TF-IDF address KNN**: independent recovery path; skip blank addresses safely
- [ ] Implement **rare-token inverted index**: retrieve by high-IDF name/address tokens; ignore tokens above configurable DF ceiling; cap postings

### 3.3 Retrieval Infrastructure

- [ ] Run each channel against both S2 and S3 independently
- [ ] Retain per-channel retrieval provenance (hit flags, ranks, similarities)
- [ ] Union candidates across all channels per S1 entity
- [ ] Deduplicate by `(source1_entity_id, candidate_entity_id)`
- [ ] Implement configurable top-K per field, source, and representation
- [ ] Experiment with K values: 5, 10, 20, 40
- [ ] Apply similarity floors per channel only after recall analysis (not arbitrary thresholds)
- [ ] Prevent exact-block explosion: rank large posting lists by another field; retain configured maximum
- [ ] Apply deterministic per-S1 final cap after union (preserve high-confidence exact hits first, then allocate bounded quotas across fuzzy channels)
- [ ] Process queries in batches; write partitioned long-form candidate artifacts incrementally
- [ ] Use top-N sparse similarity or benchmarked ANN — not dense all-pairs matrix
- [ ] Fit vectorizers only on permitted training partition for strict CV
- [ ] Serialize vectorizers for final inference
- [ ] Record `retrieval_channel_count`, per-channel rank/score, provenance bitset per candidate

### 3.4 Candidate Pair Schema

- [ ] Implement final candidate pair record (long form, one row per unique S1-target pair) with all fields from plan §4.2:
  - keys: `source1_entity_id`, `candidate_entity_id`, `candidate_source`
  - retrieval provenance columns
  - deterministic `candidate_order` for batching/output
  - `is_match` only in labeled training artifacts (set-membership join)
  - No raw label-derived field in inference features

**Acceptance gate:** Completes representative/full fold within resource budget; deterministic unique candidates.

---

## Phase 4: Blocking Evaluation

**Objective:** Implement blocking metrics; K/channel/cap experiments; analyze misses.

### 4.1 Blocking Metrics Implementation

- [ ] Implement **true-pair blocking recall (micro):** Σ|Cᵢ ∩ Tᵢ| / Σ|Tᵢ|
- [ ] Implement **entity complete-recall rate:** fraction of non-singletons with Tᵢ ⊆ Cᵢ
- [ ] Implement **entity any-recall rate:** fraction of non-singletons with ≥1 retrieved truth
- [ ] Implement **reduction ratio:** 1 − Σ|Cᵢ| / (|Q| × |D|)
- [ ] Implement **candidate load stats:** mean, median, P95, P99, max per S1, total pairs, zero-candidate rate
- [ ] Break all metrics down by source and country
- [ ] Implement **oracle macro F0.5 ceiling:** set Pᵢ = Cᵢ ∩ Tᵢ; true singletons score 1; missed non-singletons score 0; partially retrieved score exact F0.5 with oracle precision 1

### 4.2 Blocking Experiments

- [ ] Run blocking matrix over name/address K combinations
- [ ] Run blocking matrix over channel unions
- [ ] Run blocking matrix over final caps and per-source quotas
- [ ] Record incremental recall and candidate cost for each added channel
- [ ] Select smallest configuration whose OOF oracle F0.5 is near best observed ceiling and fits memory/runtime
- [ ] Measure oracle loss caused by final cap
- [ ] Measure lost truths from exact-block explosion handling
- [ ] Analyze blocking misses: which entities and why

**Acceptance gate:** Recall, candidate distribution, reduction ratio, and oracle F0.5 reported OOF.

---

## Phase 5: Lexical/Structural Feature Engineering

**Objective:** Implement versioned name/address/cross-field/retrieval features and benchmarks.

### 5.1 Name Features

- [ ] Exact matches on clean, folded, and core views
- [ ] Normalized Levenshtein distance
- [ ] Jaro-Winkler distance
- [ ] Token-set ratio and token-sort ratio
- [ ] Token Jaccard and directional containment
- [ ] Monge-Elkan token similarity (benchmark runtime first; compute only if justified)
- [ ] Char TF-IDF cosine similarity
- [ ] Word TF-IDF cosine similarity
- [ ] Character length, token count, absolute/relative differences, empty flags
- [ ] IDF-weighted token overlap and rare-token overlap/count
- [ ] Corporate-suffix agreement/removal flags

### 5.2 Address Features

- [ ] Normalized Levenshtein and Jaro-Winkler
- [ ] Token-set/sort ratios
- [ ] Token Jaccard and containment
- [ ] Optional Monge-Elkan (benchmark first)
- [ ] Char and word TF-IDF cosine
- [ ] Numeric-token intersection/union, exact set match, overlap ratio, subset in each direction, conflict indicators
- [ ] Alphanumeric-token overlap for unit/building strings
- [ ] Generic numeric sequence agreement (no PIN/ZIP length assumptions)
- [ ] Address missing on left/right/both flags
- [ ] Token-count/length differences
- [ ] Whether similarity is undefined due to missingness

### 5.3 Cross-Field & Retrieval Features

- [ ] Name × address similarity products and min/max/mean of primary scores
- [ ] Exact name × address score and exact address × name score
- [ ] Country relation flags (`country_exact_match`, `country_both_missing`, `country_left_missing`, `country_right_missing`, `country_mismatch`)
- [ ] Country relation × name/address evidence interactions
- [ ] Candidate source (`S2` vs `S3`) and source × similarity interactions
- [ ] Missing-field interactions (e.g., exact name with missing address)
- [ ] Retrieval channel flags, ranks, scores, reciprocal ranks, best rank, number of agreeing channels
- [ ] Agreement/conflict summaries (e.g., strong name + numeric-address conflict)

### 5.4 Feature Infrastructure

- [ ] Ensure all features are pure functions of two records + training-fitted artifacts
- [ ] Exclude pair IDs from model features
- [ ] Implement tiered computation: cheap vectorized first, expensive edit-distance only on final set if justified
- [ ] Version every feature group
- [ ] Write unit tests for each feature group
- [ ] Benchmark runtime per feature group
- [ ] Audit all features for null/finite values
- [ ] Create feature manifest with column order and definitions hash
- [ ] Use compiled RapidFuzz-style batch functions for string distances (avoid Python nested loops)

**Acceptance gate:** Finite schema-stable features; no IDs/labels; runtime acceptable.

---

## Phase 6: Baseline LightGBM Pair Model

**Objective:** Build group-safe training pairs, negatives, train and diagnose baseline.

### 6.1 Training Pair Construction

- [ ] Parse ground-truth cells into sets; explode into positive `(S1, target)` pairs
- [ ] Run exact production blocking configuration in each CV fold
- [ ] Label candidate pairs by set membership
- [ ] Ensure a target that is another true match of the same S1 is always positive — never sampled as negative
- [ ] Record blocked-out positives separately (do not inject into validation candidates)
- [ ] Optionally inject blocked-out positives into training pairs under named `positive_rescue` policy with honest retrieval features
- [ ] Split negatives into easy (low similarity/random) and hard (top-ranked false, same/near-exact name but conflicting address, etc.)
- [ ] Sample negatives per S1 with reproducible mixture retaining score/rank diversity
- [ ] Keep unsampled candidate validation set for honest threshold evaluation

### 6.2 LightGBM Training

- [ ] Train LightGBM with binary objective over candidate pairs
- [ ] Use bounded tree depth/leaves
- [ ] Implement early stopping on entity-aware validation
- [ ] Set deterministic seeds and log thread counts
- [ ] Do NOT blindly apply large `scale_pos_weight` — compare: controlled negative sampling, modest weights, no weights, optional OOF calibration
- [ ] Start with pair log-loss or average precision as training diagnostics only

### 6.3 Model Diagnostics

- [ ] Compute feature importances
- [ ] Generate SHAP diagnostics
- [ ] Review learning curves
- [ ] Verify no entity ID or label leakage in features

**Acceptance gate:** Every validation candidate scored OOF; reproducible training; no leakage.

---

## Phase 7: Exact CV + Macro F0.5 Thresholding

**Objective:** Implement the exact competition metric, complete pipeline folds, cross-fitted threshold search.

### 7.1 Exact Metric Implementation

- [ ] Implement `entity_f05(true_ids, predicted_ids)` exactly per plan §15 code
- [ ] Implement `macro_f05(truth_by_s1, predictions_by_s1)` exactly per plan §15 code
- [ ] Unit test: correct singleton (true empty, predict empty → 1.0)
- [ ] Unit test: incorrect singleton (true empty, predict non-empty → 0.0)
- [ ] Unit test: empty prediction on non-singleton → 0.0
- [ ] Unit test: zero-overlap predictions
- [ ] Unit test: duplicate-input rejection before set conversion
- [ ] Unit test: perfect match
- [ ] Unit test: false positives only
- [ ] Unit test: false negatives only
- [ ] Unit test: multi-match examples (confirm against README example: P=2/3, R=1.0, F0.5=0.714)

### 7.2 Cross-Validation Design

- [ ] Build bipartite graph from S1-to-S2/S3 truth links
- [ ] Compute connected components as fold groups
- [ ] Implement deterministic shuffled GroupKFold-style assignment balanced on entity count, singleton status, country, link cardinality
- [ ] Never randomly split pair rows
- [ ] Fit normalization vocabularies/IDF/vectorizers on training fold only
- [ ] Transform held-out records with frozen artifacts
- [ ] Run full held-out path per fold: normalize → retrieve → cap → features → pair score → entity meta → threshold → aggregate → exact metric
- [ ] Keep validation candidate population unsampled (negative sampling for model fitting only)
- [ ] Validate no entity crosses folds
- [ ] Validate no truth label, target count, or fold assignment leaks into transformations

### 7.3 Threshold Optimization

- [ ] Implement joint threshold search over OOF/cross-fitted scores:
  - Entity/singleton threshold (tₑ)
  - First-match pair threshold (t₁)
  - Optional additional-match threshold (tₐ)
  - Optional source-specific thresholds (only with repeated OOF evidence of stable benefit)
- [ ] Reconstruct prediction set for every S1 entity per threshold tuple
- [ ] Compute exact macro F0.5 for each
- [ ] Use coarse grid + local refinement over observed score quantiles
- [ ] Tie-break: simpler scheme, higher precision, fewer links, stability
- [ ] Report fold mean, dispersion, worst fold, per-country score, singleton contribution, sensitivity near optimum
- [ ] Implement cross-fitted threshold estimate: select thresholds using other OOF folds, evaluate on held-out fold
- [ ] Lock one final threshold configuration using all OOF data before test inference

### 7.4 OOF Artifact Generation (per fold)

- [ ] Generate fold manifest and group IDs
- [ ] Save fitted preprocessing/retrieval configuration
- [ ] Save final candidate pairs with provenance
- [ ] Save blocked-out truth report
- [ ] Save full pair features or version/hash reference
- [ ] Save pair labels
- [ ] Save OOF raw/calibrated scores
- [ ] Save per-S1 meta-features
- [ ] Save cross-fitted singleton score
- [ ] Save final prediction sets
- [ ] Save exact entity scores
- [ ] Save blocking/candidate statistics
- [ ] Save runtime/memory logs

**Acceptance gate:** Exact metric unit tests pass; every train S1 has one OOF decision; threshold stability recorded.

---

## Phase 8: Hard-Negative Mining

**Objective:** Mine OOF high-score false pairs and retrain with controlled sampling.

- [ ] Score all training candidates OOF
- [ ] Mine high-scoring false positives (top-ranked false candidates, same/near-exact name but conflicting address, strong address but conflicting name, highest model-scored false pairs)
- [ ] Retrain model with mined hard negatives in controlled mixture
- [ ] Prevent feedback leakage: mine from predictions made by model that did NOT train on that entity's group
- [ ] Measure cross-fitted macro-F0.5 improvement
- [ ] Accept iteration only if entity-level OOF F0.5 improves; otherwise reject and document

**Acceptance gate:** Measurable cross-fitted macro-F0.5 improvement or feature rejected.

---

## Phase 9: Singleton Optimization

**Objective:** Build OOF meta-features, cross-fitted entity-level singleton model, joint thresholds.

### 9.1 Singleton Model Features (from OOF pair-model scores only)

- [ ] Maximum and second-highest pair scores
- [ ] Top1-top2 difference (as weak feature only, never a hard rule)
- [ ] Mean, std, quantiles, entropy/concentration of top-k scores
- [ ] Counts above several fixed score levels
- [ ] Maximum/second-highest name, address, and combined lexical similarities
- [ ] Number of exact-name/core-name hits and number of retrieval channels agreeing
- [ ] Candidate count, source mix, zero-candidate flag
- [ ] Missing-field and country-relation summaries

### 9.2 Singleton Model Training

- [ ] Train second-stage binary model for yᵢ = 1[|Tᵢ| > 0]
- [ ] Build all meta-features from OOF pair-model scores (prevent leakage)
- [ ] Use nested/cross-fitted meta-model predictions for honest evaluation
- [ ] At final training: fit singleton model on all OOF meta-features, apply to features from pair model refit on all training data

### 9.3 Joint Threshold Re-optimization

- [ ] Re-run threshold optimization incorporating singleton model scores
- [ ] Report singleton confusion matrix and score contribution
- [ ] Report macro-F0.5 delta from singleton gating

**Acceptance gate:** No stack leakage; improves/stabilizes macro score and singleton errors.

---

## Phase 10: Dense Embeddings (Optional Experiment)

**Objective:** License-reviewed multilingual embedding features, then optional ANN retrieval.

- [ ] Identify candidate multilingual bi-encoder model
- [ ] **Before downloading**: verify exact model/revision, parameter count (≤8B), weight license, code license, tokenizer license, source URL
- [ ] Record license manifest with cached artifact checksum
- [ ] Proceed ONLY when all challenge requirements are unambiguously satisfied
- [ ] Encode clean name, address, and structured full-record text
- [ ] Add cosine similarities as features to LightGBM
- [ ] Measure incremental OOF macro-F0.5 gain
- [ ] If justified: implement dense ANN retrieval channel
- [ ] Query S2/S3 embedding indexes; union with lexical candidates
- [ ] Measure incremental oracle F0.5 and candidate growth
- [ ] Optional: fine-tune bi-encoder on positives + OOF hard negatives with group-safe folds and no external data
- [ ] Accept only if gain justifies latency/resource cost

**Acceptance gate:** Legal/resource gates pass and incremental OOF gain justifies cost.

---

## Phase 11: Cross-Encoder Reranking (Optional Experiment)

**Objective:** Rerank feasible candidate subset; produce OOF feature.

- [ ] Identify candidate cross-encoder model
- [ ] Verify license/parameter compliance (same checks as Phase 10)
- [ ] Concatenate tagged fields from S1 and candidate records
- [ ] Score only the final/safely prefiltered candidate set
- [ ] Add OOF cross-encoder probability/score as LightGBM feature
- [ ] Measure incremental OOF macro-F0.5 gain
- [ ] Report latency and resource impact
- [ ] Accept only for material net improvement

**Acceptance gate:** License/size compliant; no leakage; material net improvement.

---

## Phase 12: Final Ablation & Ensemble Selection

**Objective:** Compare only validated components; analyze folds/countries/resources.

- [ ] Compare all validated component combinations via OOF ablations
- [ ] Analyze per-fold, per-country performance
- [ ] Analyze resource usage (memory, runtime) for each configuration
- [ ] Select best robust configuration
- [ ] Lock `selected.yaml` configuration file
- [ ] Freeze artifact bundle (normalizers, TF-IDF vocabularies/IDF arrays, retrieval indexes, feature-column order, pair model, singleton model, calibrator, locked thresholds)
- [ ] Ensure selection is made BEFORE test labels or leaderboard feedback influence it

**Acceptance gate:** Best robust configuration chosen before test labels/leaderboard feedback; reproducible.

---

## Phase 13: Test Inference

**Objective:** Run frozen pipeline in batches and persist canonical candidates/scores/decisions.

### 13.1 Inference Pipeline (exact sequence from plan §18)

- [ ] Load selected frozen experiment config and verify artifact/data schema versions
- [ ] Load test TSVs using explicit tab parsing and run contracts
- [ ] Generate raw-preserving normalized views with frozen normalizer
- [ ] Load/finalize S2 and S3 retrieval indexes
- [ ] Retrieve candidates for S1 in batches
- [ ] Union, cap, deterministically deduplicate
- [ ] Persist canonical final long-form candidate set
- [ ] Aggregate candidate set into `candidate_pairs.tsv` for every S1
- [ ] Build pair features using frozen vectorizers/IDF/feature manifest
- [ ] Score all candidates with selected pair model (+ any neural feature artifacts)
- [ ] Aggregate pair scores into entity features and apply singleton model
- [ ] Apply locked pair/entity/additional-link thresholds
- [ ] Aggregate zero/one/many IDs per S1
- [ ] Write `matching_results.tsv` in original S1 order
- [ ] Ensure no inference component accesses ground truth, OOF labels, or test-wide label statistics

### 13.2 Output Format Compliance

- [ ] `matching_results.tsv`: columns `source1_entity_id`, `matched_entity_ids` (tab-separated)
- [ ] Exactly one row for every test S1 entity (1,732,544 rows)
- [ ] Singletons: `S1-ID<TAB><newline>` — empty value, never `None`, `nan`, `[]`, or quoted whitespace
- [ ] `matched_entity_ids` contains only existing S2/S3 IDs from test set, comma-separated, no duplicates
- [ ] Deterministic ordering: score-descending then ID tie-break
- [ ] `candidate_pairs.tsv`: columns `source1_entity_id`, `candidate_entity_ids` (tab-separated)
- [ ] One row per test S1, including zero-candidate rows
- [ ] Every predicted match ID appears in its candidate list
- [ ] UTF-8, LF newlines, single tab separator, no DataFrame index, no accidental quoting
- [ ] Use atomic temporary-file replacement to prevent partial submissions
- [ ] Include France entities — every test entity must appear

**Acceptance gate:** Exact test S1 coverage; no incomplete partitions; resource budget met.

---

## Phase 14: Submission Validation

**Objective:** Run strict internal checks and official validator with ID check.

### 14.1 Internal Validator

- [ ] Implement internal Python validator checking:
  - Missing/extra S1 IDs, duplicate S1 rows, row-count mismatch
  - Duplicate candidate/match IDs within a row
  - IDs outside loaded test S2/S3 sets or wrong S1/S2/S3 prefixes
  - Any match not contained in its final candidate set
  - Wrong headers, non-tab separation, extra columns, invalid UTF-8, malformed lines
  - Non-empty singleton sentinels
  - Truncated files, unexpected ordering
  - Mismatch with canonical candidate artifact hash/counts
- [ ] Use memory-aware sorted joins, integer/hash indexes, or disk-backed structures for ID validation at scale (not naive Python sets)

### 14.2 Official Validator

- [ ] Run official validator:
  ```bash
  python3 utils/validate_submission.py \
      --matching chimera_submission/output/matching_results.tsv \
      --candidate chimera_submission/output/candidate_pairs.tsv \
      --test-dir dataset/test \
      --check-ids
  ```
- [ ] Confirm `PASS` with `--check-ids` flag (not just default which skips ID check)
- [ ] Pipeline must automatically execute both validators and stop packaging on any failure

**Acceptance gate:** Zero internal errors; official validator PASS; matches ⊆ candidates.

---

## Phase 15: Documentation & Reproducibility Packaging

**Objective:** Pin requirements, expand README, fill methodology, assemble/audit zip.

### 15.1 Dependencies & Environment

- [ ] Populate `chimera_submission/code/business_entity_resolution/requirements.txt` with pinned, license-reviewed dependencies
- [ ] Verify all dependencies have acceptable licenses
- [ ] Verify any pretrained model is ≤8B parameters with MIT/Apache 2.0 license

### 15.2 Code Organization

- [ ] Ensure `chimera_submission/code/business_entity_resolution/src/main.py` is a thin CLI orchestrator (commands: `profile`, `train`, `evaluate`, `infer`)
- [ ] Organize modules per plan §21 repository architecture:
  - `src/data/` (loading.py, contracts.py, schemas.py)
  - `src/preprocessing/` (normalize.py, tokenization.py)
  - `src/blocking/` (exact.py, tfidf.py, rare_tokens.py, union.py, evaluate.py)
  - `src/features/` (name.py, address.py, cross_field.py, build.py)
  - `src/models/` (pair_model.py, singleton_model.py, thresholds.py, neural.py)
  - `src/validation/` (metric.py, folds.py, outputs.py)
  - `src/inference/` (pipeline.py)
  - `src/output/` (writers.py)
  - `src/common/` (config.py, logging.py, artifacts.py)
- [ ] Ensure experiment notebooks call reusable modules from `src/` (no duplicated logic)

### 15.3 Documentation

- [ ] Expand `chimera_submission/code/business_entity_resolution/README.md` with exact reproduction commands
- [ ] Document data → blocking → matching → output end-to-end flow
- [ ] Document expected inference command:
  ```bash
  python src/main.py --data-dir /path/to/dataset --output-dir /path/to/output
  ```
- [ ] If training is separate, document that too
- [ ] Fill in `Documentation_template.md` with:
  - Methodology used
  - Candidate generation / blocking strategy
  - Model architecture and feature engineering
  - Any other relevant approach information

### 15.4 Configuration & Reproducibility

- [ ] Create validated YAML config (`configs/baseline.yaml`, `configs/selected.yaml`) covering all parameters from plan §22
- [ ] Save fully resolved config per run: Git state, package versions, environment/hardware summary, data hashes, feature manifest, fold mapping
- [ ] Set all seeds: Python, NumPy, fold assignment, LightGBM, sampling, neural frameworks
- [ ] Fix thread counts for reproducibility; document any nondeterministic GPU operations

### 15.5 Submission Archive Assembly

- [ ] Assemble `chimera_submission.zip` with structure:
  ```
  chimera_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md
  ```
- [ ] Verify clean-environment documented command regenerates identical outputs
- [ ] Audit archive for license compliance
- [ ] Verify no external data lookup code paths exist
- [ ] Verify offline pipeline (no network calls)

**Acceptance gate:** Clean-environment documented command regenerates identical outputs; archive structure/license audit pass.

---

## Experiment Tracking (Ongoing Throughout All Phases)

- [ ] Create `experiments/experiments.csv` (append-only) with all required columns from plan §17
- [ ] Create immutable YAML/JSON manifest per run in `experiments/manifests/`
- [ ] Follow strict ablation order from plan §17:
  1. Exact/core/char-name blocking
  2. Add address retrieval
  3. Add rare-token then optional word retrieval
  4. Baseline lexical features + LightGBM
  5. Add address/numeric features
  6. Add cross-field/retrieval features
  7. Hard-negative mining
  8. Singleton model and joint thresholds
  9. Embedding features
  10. Dense retrieval
  11. Cross-encoder
  12. Ensembles
- [ ] Change one conceptual component at a time
- [ ] Promote only for repeatable cross-fitted macro-F0.5 gain with acceptable memory/runtime
- [ ] Preserve negative results

---

## Country-Shift Testing (Cross-Cutting Concern)

- [ ] Report performance per observed country
- [ ] Run leave-US-out scorer test (if fold sizes permit)
- [ ] Run leave-India-out scorer test (if fold sizes permit)
- [ ] Train on one country, evaluate on other
- [ ] Mask country features and compare
- [ ] Compare country-filtered retrieval with global fallback
- [ ] Assess France through: invariance tests, candidate diagnostics, score distributions, conservative threshold sensitivity (no pseudo-labels)
- [ ] Track feature drift and thresholds by country

---

## Later / Optional Blocking Experiments

- [ ] Word-level TF-IDF KNN channel (add only if it improves oracle score)
- [ ] MinHash/LSH for token/shingle Jaccard (retain only if recall/runtime/memory beats sparse alternatives)
- [ ] Multilingual dense bi-encoder ANN retrieval (only after license review and lexical baseline)

---

## Performance & Memory Strategy (Cross-Cutting Concern)

- [ ] Fit vectorizers once per fold/config and reuse
- [ ] Query retrieval indexes in bounded S1 batches
- [ ] Persist final candidates in partitioned long-form columnar files
- [ ] Vectorize exact/token/numeric features (avoid Python nested loops)
- [ ] Use parallel partitions for unavoidable string distances
- [ ] Compute cheap features first; benchmark expensive features before full runs
- [ ] Control process/thread oversubscription; log batch peak RSS
- [ ] Estimate total disk before full runs
- [ ] Use model prediction in batches; write scores incrementally
- [ ] Validate partition completeness before aggregation
- [ ] Benchmark likely bottlenecks: char n-gram matrices, top-K retrieval, candidate explosion, edit-distance features, LightGBM training rows, Python-set ID validation, dense embedding/cross-encoder inference

---

## Risk Mitigations (Cross-Cutting Concern)

All risks from plan §24 should be actively monitored:

- [ ] Monitor blocking recall per fold/country for irrecoverable miss risk
- [ ] Monitor collision rates and raw-vs-clean ablation for over-normalization risk
- [ ] Review high-score false positives per entity for false-merge risk
- [ ] Track singleton confusion matrix for singleton overmatching risk
- [ ] Run country-shift tests for open-set country risk
- [ ] Track error slices for missing/numeric-conflict/short addresses
- [ ] Monitor fold variance and learning curves for overfitting
- [ ] Track candidate distribution for explosion risk
- [ ] Report threshold sensitivity and cross-fold dispersion
- [ ] Audit artifact lineage/fold for leakage
- [ ] Run internal + official validators for output format errors
- [ ] Maintain license manifest for compliance
- [ ] Audit code for external data paths
- [ ] Assert canonical candidate artifact feeds both model and writer
- [ ] Benchmark peak RSS for ID validation
- [ ] Check for unseen duplicate/shared targets in Phase 1 graph audit

---

## Contradictions Between README.md and plan.md

> **No substantive contradictions found.** `plan.md` is a detailed expansion of `README.md`'s requirements, not a divergence from them. The only notable differences are:
>
> 1. **Validator `--check-ids` flag**: `README.md` shows the validator command without `--check-ids`. `plan.md` (§20) explicitly adds `--check-ids` and notes the default ID check is off. These are compatible — `plan.md` adds a stricter validation step on top of the README's baseline.
>
> 2. **Singleton count**: `plan.md` states 123,247 singletons (5.5848%). `README.md` does not mention specific counts. No contradiction, but the plan's numbers should be re-verified in Phase 1 since they came from a preliminary profile.
