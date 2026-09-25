# Business Entity Resolution Pipeline

Team: `chimera`

## Expected interface

```bash
python src/main.py --data-dir /path/to/dataset --output-dir /path/to/output
```

## Phase 1: input validation

Validate the training TSVs and print record counts, the S1 singleton ratio, and
connected-component sizes:

```bash
python src/data_contract.py --data-dir /path/to/train
```

Run the synthetic tests from this directory after installing `requirements.txt`:

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

## Phase 2: normalization

`src.normalization.normalize_source` accepts the records returned by
`src.data_contract.load_source`. Each normalized record retains the original
`BusinessRecord` and exposes clean, folded, and core name views plus clean and
alias address views. `collision_rates` reports the fraction of nonempty values
that share a view with another record.

## Phase 3: lexical candidates

Pass normalized S1, S2, and S3 mappings to
`src.retrieval.generate_candidates`, optionally with a `RetrievalConfig`. The
generator yields bounded, deduplicated candidates with per-channel ranks and
scores. `write_candidate_artifact` writes the internal pair-level audit
artifact; the final competition `candidate_pairs.tsv` has a different,
per-S1 format and is produced by the later inference phase.

## Phase 4: training-only retrieval benchmark

`src.phase4_benchmark` validates and indexes the training TSVs, runs the five
Phase 3 retrieval channels, and writes `metrics.json`, `phase4_report.md`, a
per-ground-truth-link retrieval-status TSV, and an auditable sample of genuine
retrieval misses. It does not read test labels or estimate ranking failures.
The complete training run and retrieval decision are deferred to the user's CPU run; the
local tests use only small synthetic fixtures.

```bash
python -m src.phase4_benchmark \
  --train-dir /path/to/dataset/train \
  --output-dir /path/to/phase4-report \
  --work-dir /path/to/phase4-work
```

The work directory holds a disk-backed store and per-channel candidate arrays.
`--stage` can run `store`, one channel at a time, or `metrics` after all five
channels complete. Completed channel stages are reused only with the same
retrieval configuration. These training-wide results are exploratory, not
cross-fitted model-selection estimates. Ranking failure analysis must wait
for OOF LightGBM scores after Phases 7/8.

## Phase 6: pair features

`src.pair_features.PairFeatureExtractor` emits a fixed, numeric feature schema
for each Phase 3 `Candidate`. It includes normalized edit distances, token and
character TF-IDF similarity, numeric address overlap, country/missingness
interactions, and per-channel retrieval provenance. Pass a repeatable,
unlabeled, fold-appropriate record stream to `fit_from_records`; fit the TF-IDF
encoders separately within each training fold when building OOF features.
`transform` yields `(candidate, features)` pairs lazily, with no labels or
ground-truth fields. Full-data feature materialization and timing are deferred
to the user's CPU run after the retrieval decision.

## Phase 7: pair-model baseline

`src.pair_model.assign_component_folds` maps Phase 1 truth-graph components
to GroupKFold assignments. `train_pair_baseline` fits one deterministic
LightGBM fold from numeric Phase 6 features and returns validation scores,
learning history, and pair-level precision/recall/AUC diagnostics. It rejects
overlapping train/validation components and non-finite features. Local tests
fit only tiny synthetic matrices; full training, memory checks, and learning-
curve review are deferred to the user's CPU run. Pair diagnostics are not retrieval,
ranking-failure, or entity-level accuracy estimates.

`src.cpu_training.train_cpu_baseline` connects the disk-backed Phase 4 candidates
to a bounded, explicitly sized training-S1 sample. It fits fold-specific TF-IDF
encoders, component-disjoint LightGBM OOF models, reports retrieved-pair ranking,
searches A/B/C decision policies on OOF scores, and freezes a separate final
model and feature extractor. This is a CPU-feasible exploratory sample; its
metrics are not full-training estimates. No full project-data training is run
by the repository tests.

## Phase 8: OOF scores and calibration

`src.oof_predictions.generate_oof_predictions` scores each pair with a
LightGBM model trained outside its truth-graph component fold. OOF models use
fixed boosting rounds so held-out labels cannot select their iteration.
Isotonic calibration uses an inner OOF loop within the other folds; its
calibrated score for an outer fold never uses that fold's labels. The final
calibrator fitted on all raw OOF scores is for later inference only, not OOF
evaluation. At least three component folds are required. A label-free TSV can
be written with `write_oof_artifact`.

`src.oof_ranking.evaluate_oof_ranking` reports top-1 retrieved-pair ranking
failure, retrieved-pair Recall@5/10, multi-positive entity any-hit rates, and
false-positive competition from source-sorted raw OOF scores. Retrieval misses are counted
separately and excluded from ranking-rate denominators. No project-data OOF
training or ranking report has been run locally.

## Phase 9: entity score and comparison

`src.entity_decision.entity_f05` and `macro_f05` implement the competition's
per-S1 macro F₀.₅ exactly, including score 1 for a correctly empty singleton.
`summarize_entity_scores` captures max/second scores, gap, counts, spread, and
concentration. `compare_oof_score_paths` streams source-sorted candidate rows
and compares raw and calibrated OOF scores at the same supplied threshold,
including entities with no candidates. It does not select calibration or a
final threshold; those decisions require full OOF results from the user's CPU run.

## Phase 10: frozen match-set policy and OOF threshold search

`src.threshold_policy.predict_match_set` implements decision Policy C. It sorts
unique candidates by descending score then ascending ID, uses the top score as
entity confidence, rejects only when `top1 < entity_threshold`, picks top-1
when `gap >= gap_threshold`, otherwise retains every score `>= pair_threshold`
with top-1 fallback. A single candidate has decision gap `+inf`; a tie has
gap zero. Policies A (pair gate only) and B (entity + pair gates, with fallback)
are available through `predict_policy`. `FrozenDecisionConfig` serializes the
selected rule and reuses its exact prediction function at inference.

`src.threshold_search.assemble_oof_entities` accepts all training S1 truth,
entity fold/country maps, and Phase 8 OOF pair arrays (raw or calibrated).
Include zero-candidate S1 entities. Supply an explicit `ThresholdGrid` to
`search_thresholds`; no grid values are built in. Each heldout fold's threshold
is selected from other OOF folds, and all three decision policies are compared
by worst heldout fold, then overall macro F₀.₅, dispersion, and simplicity.
The chosen family's all-OOF threshold refit is separately labeled exploratory.
Run `threshold_sensitivity` with a chosen perturbation and
`leave_one_country_out` with a chosen minimum country size, then serialize via
`save_search_report` and `save_frozen_config`. The search artifact records the
grid, selected thresholds, fold and country diagnostics, singleton error
rates, and decision proportions. These functions are implemented and covered
by synthetic tests; project-data search and threshold freezing await the user's CPU run.

## Phase 11: nested hard-negative mining

`src.hard_negative_mining.generate_hard_negative_oof_predictions` fits inner
mining models inside each outer training split, so neither the mined pairs nor
their scores depend on that outer fold's labels. It selects high-scoring known
negatives at an explicitly supplied score cutoff, caps them per S1, and gives
them additional LightGBM training weight without copying feature rows. It
produces new raw OOF scores and per-fold mining diagnostics. No mining cutoff
is selected from test data or the heldout fold. `compare_raw_oof_hard_negatives`
re-runs the same Phase 9/10 threshold search on baseline and mined raw scores;
it reports, but does not automatically accept, any cross-fitted entity F₀.₅
gain. If the calibrated pathway is retained, its nested OOF calibrator must
also be regenerated before a final acceptance decision. Full project-data
training, downstream re-optimization, and the keep/reject decision await the
user's local CPU run.

## Phase 12: CPU entity meta-model

`src.entity_meta_model.generate_meta_oof_decisions` trains a compact logistic
ZERO/ONE/MANY classifier from score-only entity aggregates of pair OOF scores.
For each outer fold it fits the classifier on other folds and tunes
`multi_pair_threshold` from inner OOF class predictions using exact entity
macro F₀.₅. ZERO returns empty, ONE top-1, and MANY every candidate above the
threshold, falling back to top-1 if none pass; MANY never forces a second
match. The final all-OOF model and threshold are exploratory until compared.
`compare_meta_to_phase10` takes the selected Phase 10 family's per-fold
configs, applies both systems to the identical OOF entities, and requires
caller-supplied worst-fold and dispersion tolerances for a keep decision.
`save_meta_artifact`/`load_meta_artifact` freeze the model and threshold; load
only trusted local artifacts because joblib uses pickle. The full CPU run and
acceptance decision remain pending.

## Phase 13: optional dense retrieval on a 16 GB CPU laptop

This branch is **off by default**. The 384-dimensional multilingual MiniLM
encoder is listed as Apache-2.0 and about 0.1B parameters on its
[model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
Install `requirements-dense.txt` only if the Phase 4 miss taxonomy indicates a
semantic/linguistic retrieval bottleneck. The [Faiss IVF-PQ index](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
compresses target vectors; target embeddings are streamed to a float16 file on
disk instead of held in memory. For roughly 10.3 million targets, that file
alone needs about 7.4 GiB of disk space, and encoding may take substantial CPU
time. A pinned encoder commit revision is required for reproducibility.

`python -m src.dense_retrieval --help` lists independent `encode`, `index`,
`query`, and train-only `evaluate` stages. Use a separate dense work directory;
never point it at the existing Phase 4 work directory. The target and query
encoding stages checkpoint after each flushed batch and can resume. Dense
retrieval is accepted only after unique ground-truth recovery, candidate
volume, and the full Phase 6→10 OOF entity-level impact justify its cost.

## Disk-backed pipeline bridge

`src.pipeline_store.build_unlabeled_store` validates test Source-1/2/3 TSVs
into the same normalized SQLite schema as the training store, but creates no
ground-truth table and never reads test labels. `DiskCandidateStore` reads the
completed lexical channel arrays (and optional dense arrays) as memory maps,
unions and caps hits with deterministic provenance, and fetches only one S1's
target records at a time. New rare-token runs persist their actual score array;
older Phase 4 work without it uses a documented reciprocal-rank surrogate.
The training/prediction runner is the next implementation step.
