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

Run the synthetic contract and normalization tests from this directory:

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

## Phase 2: normalization

`src.normalization.normalize_source` accepts the records returned by
`src.data_contract.load_source`. Each normalized record retains the original
`BusinessRecord` and exposes clean, folded, and core name views plus clean and
alias address views. `collision_rates` reports the fraction of nonempty values
that share a view with another record.
