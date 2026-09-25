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
The complete training run and retrieval decision are deferred to Colab; the
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
to Colab after the retrieval decision.
