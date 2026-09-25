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
