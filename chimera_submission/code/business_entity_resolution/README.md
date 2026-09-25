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

Run the synthetic contract tests from this directory:

```bash
python -B -m unittest discover -s tests -p 'test_data_contract.py' -v
```
