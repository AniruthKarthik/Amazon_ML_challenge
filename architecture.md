# Architecture — Business Entity Resolution Pipeline

> **Team:** chimera  
> **Derived strictly from:** `README.md` and `plan.md`  
> **Purpose:** Implementation-ready architecture reference. Covers system topology, data flow, module contracts, internal schemas, pipeline sequencing, and cross-cutting concerns.

---

## 1. Problem Domain

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ENTITY RESOLUTION TASK                         │
│                                                                       │
│  Given:   Source 1 (S1) — deduplicated reference entities             │
│           Source 2 (S2) — noisy duplicate pool                        │
│           Source 3 (S3) — noisy duplicate pool                        │
│                                                                       │
│  Predict: For each S1 entity, zero/one/many matching S2 ∪ S3 records │
│                                                                       │
│  Metric:  Macro-averaged F₀.₅ (precision-heavy, per-entity)          │
│           Singletons score 1.0 if correctly predicted empty           │
│                                                                       │
│  Scale:   ~2.2M train S1 × ~10.3M train S2∪S3 = ~22.8T pairs        │
│           ~1.7M test S1  × ~10.0M test S2∪S3  = ~17.3T pairs        │
│                                                                       │
│  Countries: Train = {US, India}  |  Test = {US, India, France}       │
│             Country is OPEN-SET — never hard-code a vocabulary        │
└─────────────────────────────────────────────────────────────────────────┘
```

The task is a **retrieval + pairwise matching + entity-level set decision** problem. It is NOT:
- A one-to-one assignment problem (one S1 can match many S2/S3)
- A multiclass/winner-takes-all problem (multiple correct links per entity)
- A plain binary classification problem (does not handle candidate discovery or set-level decisions)

### Key Label Statistics (from plan, to verify in Phase 1)

| Statistic | Value |
|---|---|
| Train singletons | 123,247 (5.5848%) |
| Total links | 7,638,365 |
| Mean links per S1 | 3.461 |
| Max links per S1 | 11 |

---

## 2. High-Level System Architecture

```
                       TRAINING                                      INFERENCE
             +-------------------------+                  +-------------------------+
TSV files -->| loader + contract checks|<-----------------| test TSV files          |
             +------------+------------+                  +------------+------------+
                          |                                            |
                          v                                            v
             +-------------------------+                  +-------------------------+
             | multi-view normalizer   |                  | frozen multi-view       |
             | (raw values preserved)  |                  | normalizer              |
             +------------+------------+                  +------------+------------+
                          |                                            |
                          v                                            v
             +---------------------------------------------------------------+
             | union blocking: exact + char TF-IDF + rare token (+ optional) |
             | per-channel top-K -> union -> deduplicate -> final cap         |
             +----------------------+----------------------------------------+
                                    |
                   +----------------+----------------+
                   | final candidate-pair table C_i  |
                   +-----------+---------------------+
                               |                         +-----------------------+
                               +------------------------>| candidate_pairs.tsv   |
                               v                         +-----------------------+
                   +-----------------------+
                   | pair feature builder  |
                   +-----------+-----------+
                               |
                               v
                   +-----------------------+       optional neural signals
                   | LightGBM pair scorer  |<------(embeddings/cross-encoder)
                   +-----------+-----------+
                               |
                               v
                   +-----------------------+
                   | per-S1 score summary  |
                   | + singleton model     |
                   +-----------+-----------+
                               |
                               v
                   +-----------------------+
                   | OOF-selected pair /   |
                   | entity thresholds     |
                   +-----------+-----------+
                               |
                               v
                   +-----------------------+       +-----------------------+
                   | aggregate zero/one/   |------>| matching_results.tsv  |
                   | many IDs per S1       |       +-----------+-----------+
                   +-----------------------+                   |
                                                               v
                                                  internal + official validators
```

### Core Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Search strategy | Union of independent sparse/lexical retrievers | Different noise modes need different recovery paths; Cartesian is infeasible |
| Initial retrieval | Char n-gram TF-IDF + exact + rare-token channels | Tolerates typos, punctuation, transliteration without country rules |
| Matcher | LightGBM on explicit pair features | Handles nonlinear tabular interactions, missingness, mixed scales; MIT license |
| Final decision | Pair scorer → entity-level singleton model → OOF-tuned thresholds | Default 0.5 doesn't optimize macro F₀.₅; singletons need explicit control |
| Country | Generic relational features + dynamic retrieval with global fallback | Fixed vocabularies fail on France and future unseen labels |
| Data representation | Raw + multiple conservative normalized views | One aggressive normalization destroys discriminative evidence |
| Validation | Entity/connected-component folds with full pipeline OOF | Random pair splits leak entity variants |
| Neural models | Optional signals after baseline, gated by ablation + license check | Pure Transformer may miss numeric/address structure; expensive |

---

## 3. Repository Structure

```
chimera_submission/code/business_entity_resolution/
├── configs/
│   ├── baseline.yaml               # initial experiment configuration
│   └── selected.yaml               # locked final configuration
├── src/
│   ├── main.py                      # thin CLI orchestrator (profile|train|evaluate|infer)
│   ├── data/
│   │   ├── loading.py               # TSV I/O with explicit sep="\t"
│   │   ├── contracts.py             # schema/ID/coverage/uniqueness validation
│   │   └── schemas.py               # typed record definitions
│   ├── preprocessing/
│   │   ├── normalize.py             # multi-view name/address/country transforms
│   │   └── tokenization.py          # token extraction, n-gram views
│   ├── blocking/
│   │   ├── exact.py                 # exact clean/core name inverted indexes
│   │   ├── tfidf.py                 # char/word n-gram TF-IDF KNN retrieval
│   │   ├── rare_tokens.py           # high-IDF token inverted index
│   │   ├── union.py                 # multi-channel union, dedup, caps
│   │   └── evaluate.py             # blocking-specific metrics
│   ├── features/
│   │   ├── name.py                  # pairwise name similarity features
│   │   ├── address.py               # pairwise address similarity features
│   │   ├── cross_field.py           # cross-field, country, retrieval features
│   │   └── build.py                 # feature assembly, manifest, tiered compute
│   ├── models/
│   │   ├── pair_model.py            # LightGBM pair training/scoring
│   │   ├── singleton_model.py       # entity-level singleton classifier
│   │   ├── thresholds.py            # joint threshold search & optimization
│   │   └── neural.py               # optional embedding/cross-encoder adapters
│   ├── validation/
│   │   ├── metric.py                # exact entity_f05 and macro_f05
│   │   ├── folds.py                 # connected-component GroupKFold
│   │   └── outputs.py              # strict submission format validation
│   ├── inference/
│   │   └── pipeline.py             # frozen artifact orchestration
│   ├── output/
│   │   └── writers.py              # deterministic TSV serialization
│   └── common/
│       ├── config.py                # YAML config loading & validation
│       ├── logging.py               # structured logging
│       └── artifacts.py             # version/checksum/path utilities
├── tests/                           # unit/integration/golden tests
├── experiments/
│   ├── experiments.csv              # append-only experiment log
│   └── manifests/                   # immutable per-run YAML/JSON manifests
├── artifacts/                       # generated (not packaged unless required)
├── README.md                        # reproduction instructions
└── requirements.txt                 # pinned, license-reviewed dependencies

chimera_submission/output/           # only final competition TSVs
├── matching_results.tsv
└── candidate_pairs.tsv
```

### Module Responsibilities

| Module | Does | Does NOT |
|---|---|---|
| `data` | TSV I/O, source/ground-truth contracts, typed schemas, manifests | Normalization, modeling |
| `preprocessing` | Deterministic raw-to-multi-view transforms, tokenization | Fold splitting, label decisions |
| `blocking` | Retriever fit/query, union/dedup/caps, provenance, blocking metrics | Pair scoring, threshold decisions |
| `features` | Reusable vectorized pair transformations | Fold splitting, label decisions |
| `models` | Pair training/scoring, singleton stacking, calibration, thresholds, optional neural | Data loading, normalization |
| `validation` | Exact metric, leakage-safe folds, submission format checks | Model training, output writing |
| `inference` | Orchestration of frozen components only | Fitting, label access |
| `output` | Deterministic competition-format serialization | Scoring, threshold logic |
| `common` | Config validation, structured logs, artifact/version utilities | Business logic |

### CLI Interface

```bash
# Data profiling
python src/main.py profile --data-dir /path/to/dataset

# Training (all phases)
python src/main.py train --data-dir /path/to/dataset --config configs/baseline.yaml

# Evaluation (OOF)
python src/main.py evaluate --data-dir /path/to/dataset --config configs/selected.yaml

# Test inference (frozen pipeline)
python src/main.py infer --data-dir /path/to/dataset --output-dir /path/to/output
```

---

## 4. Data Schemas

### 4.1 Input TSV Schema (all source files)

| Column | Type | Constraints |
|---|---|---|
| `entity_id` | string | Non-null, unique within source, prefix = `S1-`/`S2-`/`S3-` matching file |
| `business_name` | string | Never missing in any source |
| `business_address` | string | May be missing in S2/S3 |
| `country` | string | Never missing; open-set label |

**Loading contract:** `pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False, encoding="utf-8")`

### 4.2 Ground Truth Schema (`train_ground_truth.tsv`)

| Column | Type | Constraints |
|---|---|---|
| `source1_entity_id` | string | Every train S1 ID exactly once |
| `matched_entity_ids` | string | Comma-separated S2/S3 IDs; empty for singletons; no duplicates within row; all IDs must exist in source files |

### 4.3 Normalized Entity Record (internal)

One row per source record, produced by the preprocessing module:

| Field | Type | Source |
|---|---|---|
| `entity_id` | string, non-null, unique in source | Direct from input |
| `source` | categorical (`S1`/`S2`/`S3`) | Derived from prefix, validated |
| `business_name_raw` | string | Exact supplied value |
| `business_name_clean` | string | NFKC + casefold + conservative punctuation/space |
| `business_name_folded` | string | Accent-folded clean name |
| `business_name_core` | string | Legal suffix normalized (e.g., Ltd→Limited) |
| `business_address_raw` | string | Exact supplied value |
| `business_address_clean` | string | NFKC + casefold + conservative punctuation/space |
| `business_address_folded` | string | Accent-folded clean address |
| `business_address_alias` | string | Safe alias/abbreviation representation |
| `country_raw` | string | Original label |
| `country_clean` | string | Generic normalized label |
| `name_missing` | bool | Explicit missingness flag |
| `address_missing` | bool | Explicit missingness flag |
| `country_missing` | bool | Explicit missingness flag |

**Invariants:**
- Raw columns are byte-for-byte recoverable from loaded values
- Empty fields canonicalized to `""` (never `"None"`, `"nan"`, `null`)
- Normalization is deterministic and idempotent
- Accent folding never replaces the primary clean form

### 4.4 Final Candidate Pair Record (long form)

One row per unique `(S1, target)` pair after union/dedup/cap:

| Field | Type | Notes |
|---|---|---|
| `source1_entity_id` | string | S1 record key |
| `candidate_entity_id` | string | S2 or S3 record key |
| `candidate_source` | categorical (`S2`/`S3`) | Derived from prefix |
| Per-channel hit flags | bool × N | Which retrieval channels found this pair |
| Per-channel ranks | int × N | Rank within each channel |
| Per-channel similarities | float × N | Score within each channel |
| `best_rank` | int | Best rank across all channels |
| `best_score` | float | Best similarity across all channels |
| `retrieval_channel_count` | int | Number of channels that found this pair |
| `candidate_order` | int | Deterministic ordering for batching/output |
| `is_match` | bool | **Training only** — set-membership join against ground truth |

**Invariants:**
- No raw label-derived field survives into inference features
- This is the canonical artifact: both pair scorer and `candidate_pairs.tsv` derive from it
- Post-union, post-cap, immutable once generated per fold/run

### 4.5 Pairwise Feature Record

| Field | Type | Notes |
|---|---|---|
| Pair keys | string × 2 | `source1_entity_id`, `candidate_entity_id` |
| Versioned feature columns | float/bool | Schema-stable, finite values only |
| Missingness indicators | bool | Accompany imputed numeric values |

**Invariants:**
- Pure functions of two records + training-fitted artifacts
- Pair IDs excluded from model input
- Feature-manifest hash fixes column order and definitions

### 4.6 Entity Decision Record

| Field | Type | Notes |
|---|---|---|
| `source1_entity_id` | string | S1 entity |
| `candidate_count` | int | Total candidates for this entity |
| Score summary features | float × N | max, 2nd-highest, mean, std, quantiles, entropy |
| `singleton_probability` | float | From singleton model |
| `selected_threshold_version` | string | Which threshold config applied |
| `predicted_ids` | list[string] | Ordered list of matched S2/S3 IDs |
| `reason_code` | string | `singleton_gate`, `above_pair_threshold`, `no_candidates` |

---

## 5. Normalization Architecture

```
                    ┌──────────────────┐
                    │   Raw Input      │
                    │  (byte-for-byte  │
                    │   preserved)     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Shared Pipeline   │
                    │ ┌──────────────┐  │
                    │ │ Missing →""  │  │
                    │ │ + flag       │  │
                    │ ├──────────────┤  │
                    │ │ Unicode NFKC │  │
                    │ ├──────────────┤  │
                    │ │ casefold()   │  │
                    │ ├──────────────┤  │
                    │ │ Punctuation  │  │
                    │ │ normalize    │  │
                    │ ├──────────────┤  │
                    │ │ Whitespace   │  │
                    │ │ collapse     │  │
                    │ └──────────────┘  │
                    └──┬─────────┬─────┘
                       │         │
            ┌──────────▼──┐  ┌──▼──────────┐
            │ NAME BRANCH │  │ ADDR BRANCH │
            ├─────────────┤  ├─────────────┤
            │ _clean      │  │ _clean      │
            │ _folded     │  │ _folded     │
            │ _core       │  │ _alias      │
            │ token sets  │  │ token sets  │
            │ n-gram views│  │ (α,num,αnum)│
            └─────────────┘  └─────────────┘
```

### Multi-View Strategy

The system produces **multiple parallel views** of each text field rather than a single aggressively normalized form. This preserves evidence for the model while enabling different retrieval and comparison strategies.

| View | Purpose | What it changes | What it preserves |
|---|---|---|---|
| `_raw` | Ground truth / debugging | Nothing | Everything |
| `_clean` | Primary comparison | NFKC, case, whitespace, safe punctuation | Accents, script, structure |
| `_folded` | Cross-accent matching | Additionally removes combining marks | Script, structure |
| `_core` (name) | Legal suffix equivalence | Canonical suffix normalization | Core name tokens |
| `_alias` (address) | Abbreviation equivalence | Reviewed safe aliases | Ambiguous tokens untouched |

### Critical Normalization Rules

- **Never** transliterate all non-Latin text to ASCII as the sole representation
- **Never** blindly rewrite ambiguous tokens (e.g., `st` → `street`; it could be saint, a name, etc.)
- **Never** parse addresses into mandatory US/India-specific components
- **Always** preserve raw alongside normalized
- **Always** apply `casefold()` not `lower()` (Unicode-aware)
- Corporate suffix removal is a **weak equivalence signal**, not ground truth

---

## 6. Candidate Generation (Blocking) Architecture

### 6.1 Channel Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL CHANNELS                           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Ch.1: Exact  │  │ Ch.2: Exact  │  │ Ch.3: Char n-gram   │  │
│  │ clean name   │  │ core name    │  │ TF-IDF name KNN     │  │
│  │ (inverted    │  │ (inverted    │  │ (primary fuzzy)      │  │
│  │  index)      │  │  index,      │  │                      │  │
│  │              │  │  capped)     │  │                      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│  ┌──────┴───────┐  ┌──────┴───────────────────┐  │              │
│  │ Ch.4: Char   │  │ Ch.5: Rare-token         │  │              │
│  │ n-gram TF-IDF│  │ inverted index           │  │              │
│  │ address KNN  │  │ (high-IDF name/addr      │  │              │
│  │ (skip blanks)│  │  tokens; DF ceiling;     │  │              │
│  │              │  │  posting cap)            │  │              │
│  └──────┬───────┘  └──────┬───────────────────┘  │              │
│         │                 │                      │              │
└─────────┼─────────────────┼──────────────────────┼──────────────┘
          │                 │                      │
          └────────┬────────┴──────────────────────┘
                   │
          ┌────────▼────────┐
          │   UNION         │  per-channel top-K results
          │   + DEDUP       │  by (S1_id, candidate_id)
          │   + FINAL CAP   │  preserve exact hits first,
          │   + PROVENANCE  │  then bounded fuzzy quotas
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │ Canonical Final  │  ← immutable artifact
          │ Candidate Set    │  ← feeds scorer AND
          │ (long-form)      │     candidate_pairs.tsv
          └─────────────────┘
```

### 6.2 Per-Channel Configuration

| Channel | Index Type | Input View | Key Parameters |
|---|---|---|---|
| Exact clean name | Inverted index | `business_name_clean` | — |
| Exact core name | Inverted index | `business_name_core` | Posting cap for common cores |
| Char n-gram name | TF-IDF + top-K sparse | `business_name_clean`/`folded` | analyzer, n-gram range, min/max_df, K |
| Char n-gram address | TF-IDF + top-K sparse | `business_address_clean`/`folded` | Same + skip-blank logic |
| Rare-token index | Inverted index | High-IDF name/address tokens | DF ceiling, posting cap |

### 6.3 Optional Later Channels

| Channel | Trigger |
|---|---|
| Word-level TF-IDF KNN | Add only if it improves oracle score |
| MinHash/LSH | Only if recall/runtime/memory beats sparse alternatives |
| Dense bi-encoder ANN | Only after license review + lexical baseline established |

### 6.4 Candidate Controls

```
Per channel:
  ├── top-K per field, source, representation (experiment: K ∈ {5, 10, 20, 40})
  ├── similarity floor (set only after recall analysis)
  └── posting cap (for explosion prevention on exact blocks)

Post-union:
  ├── deduplicate by (source1_entity_id, candidate_entity_id)
  ├── preserve high-confidence exact hits first
  ├── bounded quotas / diversity across fuzzy channels
  └── deterministic per-S1 final cap
```

### 6.5 Blocking Evaluation Metrics

| Metric | Formula / Description |
|---|---|
| **True-pair blocking recall (micro)** | Σ\|Cᵢ ∩ Tᵢ\| / Σ\|Tᵢ\| |
| **Entity complete-recall rate** | Fraction of non-singletons where Tᵢ ⊆ Cᵢ |
| **Entity any-recall rate** | Fraction of non-singletons with ≥1 retrieved truth |
| **Reduction ratio** | 1 − Σ\|Cᵢ\| / (\|Q\| × \|D\|) |
| **Candidate load** | Mean, median, P95, P99, max per S1; total pairs; zero-candidate rate |
| **Oracle macro F₀.₅ ceiling** | Pᵢ = Cᵢ ∩ Tᵢ with oracle precision 1; macro-average entity scores |

All metrics broken down by source and country.

---

## 7. Feature Engineering Architecture

### 7.1 Feature Taxonomy

```
┌─────────────────────────────────────────────────────────┐
│                   PAIR FEATURE VECTOR                   │
│                                                         │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │  NAME FEATURES  │  │ ADDRESS FEATURES│              │
│  │  ───────────────│  │  ──────────────-│              │
│  │  Exact matches  │  │  Edit distances │              │
│  │  Edit distances │  │  Token overlap  │              │
│  │  Token overlap  │  │  TF-IDF cosine  │              │
│  │  TF-IDF cosine  │  │  Numeric tokens │              │
│  │  IDF-weighted   │  │  Missing flags  │              │
│  │  Suffix flags   │  │  Length diffs   │              │
│  │  Length/count   │  │                 │              │
│  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                        │
│  ┌────────▼────────────────────▼────────┐              │
│  │      CROSS-FIELD FEATURES            │              │
│  │  ────────────────────────────────────│              │
│  │  Name × Address products/min/max     │              │
│  │  Country relation flags & interact.  │              │
│  │  Source (S2/S3) & interactions       │              │
│  │  Missing-field interactions          │              │
│  │  Agreement / conflict summaries      │              │
│  └────────┬─────────────────────────────┘              │
│           │                                             │
│  ┌────────▼─────────────────────────────┐              │
│  │      RETRIEVAL FEATURES              │              │
│  │  ────────────────────────────────────│              │
│  │  Channel hit flags                   │              │
│  │  Per-channel ranks, scores           │              │
│  │  Reciprocal ranks, best rank         │              │
│  │  Channel agreement count             │              │
│  └──────────────────────────────────────┘              │
│                                                         │
│  ┌──────────────────────────────────────┐   OPTIONAL   │
│  │  Embedding cosine similarities       │◄── Phase 10+ │
│  │  Cross-encoder scores                │              │
│  └──────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Name Features (detailed)

| Feature | Type | Purpose |
|---|---|---|
| Exact match (clean/folded/core) | bool × 3 | High-precision anchors |
| Normalized Levenshtein | float | Local edit / typo similarity |
| Jaro-Winkler | float | Prefix agreement |
| Token-set ratio | float | Token reordering / subsets |
| Token-sort ratio | float | Sorted token comparison |
| Token Jaccard | float | Shared-token fraction |
| Directional containment | float × 2 | Partial name inclusion |
| Monge-Elkan | float | Best token correspondences (expensive — benchmark first) |
| Char TF-IDF cosine | float | Corpus-aware fuzzy overlap |
| Word TF-IDF cosine | float | Word-level corpus similarity |
| Char length (each + diff) | int/float | Similarity reliability context |
| Token count (each + diff) | int/float | Similarity reliability context |
| Empty flags | bool | Missing-field indicator |
| IDF-weighted token overlap | float | Distinctive shared words |
| Rare-token overlap / count | float/int | High-value shared tokens |
| Suffix agreement/removal flags | bool | Whether core equality came from normalization |

### 7.3 Address Features (detailed)

| Feature | Type | Purpose |
|---|---|---|
| Normalized Levenshtein, Jaro-Winkler | float × 2 | Edit similarity |
| Token-set/sort ratios | float × 2 | Token reordering |
| Token Jaccard, containment | float × 3 | Overlap measures |
| Monge-Elkan | float | Optional (benchmark first) |
| Char/word TF-IDF cosine | float × 2 | Corpus-aware similarity |
| Numeric-token intersection / union | int × 2 | Shared numbers |
| Numeric exact set match | bool | All numbers match |
| Numeric overlap ratio | float | Proportion of shared numbers |
| Numeric subset (each direction) | bool × 2 | Containment |
| Numeric conflict indicator | bool | Conflicting numbers present |
| Alphanumeric-token overlap | float | Unit/building strings |
| Generic numeric sequence agreement | float | No PIN/ZIP assumptions |
| Address missing (left/right/both) | bool × 3 | Explicit missingness |
| Token-count / length differences | float × 2 | Reliability context |
| Similarity undefined flag | bool | Due to missingness |

### 7.4 Country Features

| Feature | Type |
|---|---|
| `country_exact_match` | bool |
| `country_both_missing` | bool |
| `country_left_missing` | bool |
| `country_right_missing` | bool |
| `country_mismatch` | bool |

All relational — no vocabulary-specific encoding.

### 7.5 Computation Tiers

```
TIER 1 (cheap, vectorized):       TIER 2 (expensive, conditional):
 ├── Exact matches                  ├── Edit distances (Levenshtein, JW)
 ├── Token counts / lengths         ├── Monge-Elkan
 ├── Missing flags                  ├── Cross-encoder scores
 ├── Numeric token sets             └── Dense embedding cosines
 ├── Token Jaccard / containment
 ├── TF-IDF cosine (pre-computed)
 └── Retrieval provenance
```

Tier 2 features computed only on final candidate set and only if ablations justify them.

---

## 8. Modeling Architecture

### 8.1 Two-Stage Model Stack

```
                ┌────────────────────────┐
                │   STAGE 1: Pair Model  │
                │   (LightGBM, binary)   │
                │                        │
                │   Input: pair features │
                │   Output: pair_score   │
                └───────────┬────────────┘
                            │
                   per-S1 aggregation
                            │
                ┌───────────▼────────────┐
                │  STAGE 2: Singleton    │
                │  Model (binary)        │
                │                        │
                │  Input: entity-level   │
                │  score summaries       │
                │  Output: singleton_p   │
                └───────────┬────────────┘
                            │
                ┌───────────▼────────────┐
                │  Threshold Decision    │
                │                        │
                │  t_e: entity gate      │
                │  t_1: first match      │
                │  t_a: additional match │
                │                        │
                │  Output: predicted     │
                │  ID set (0/1/many)     │
                └────────────────────────┘
```

### 8.2 Pair Model (LightGBM)

| Aspect | Design |
|---|---|
| Objective | Binary (log-loss) — training diagnostic only |
| Selection metric | Exact end-to-end OOF macro F₀.₅ |
| Positives | Ground-truth matched pairs found by blocking |
| Negatives | Mixture of easy (low-sim) + hard (top-ranked false) per S1 |
| `scale_pos_weight` | NOT blindly applied — compare no weights, modest, controlled sampling |
| Calibration | Only if it improves threshold stability (isotonic/Platt) |
| Seeds | Deterministic: Python, NumPy, LightGBM |
| Early stopping | Entity-aware validation metric |
| Diagnostics | Feature importances, SHAP, learning curves |
| Alternatives | CatBoost, XGBoost as ablation options only |

### 8.3 Singleton Model

| Aspect | Design |
|---|---|
| Target | yᵢ = 1[|Tᵢ| > 0] (has any true match) |
| Features | Entity-level score summaries from OOF pair scores |
| Training data | OOF meta-features only (prevents leakage) |
| Evaluation | Nested/cross-fitted predictions |
| Final fit | All OOF meta-features → apply to pair model refit on all data |
| Key rule | Top1-top2 margin is a feature, never a hard rule (multiple matches are valid) |

### 8.4 Threshold Optimization

| Parameter | Description |
|---|---|
| tₑ (entity threshold) | Whether any match is allowed |
| t₁ (first-match threshold) | Score for first accepted match |
| tₐ (additional-match threshold) | Score for subsequent matches (optional) |
| Source-specific thresholds | Only with repeated OOF evidence of stable benefit |

**Search:** Coarse grid → local refinement over OOF score quantiles  
**Selection:** Cross-fitted (select on other OOF folds, evaluate on held-out)  
**Tie-break:** Simpler scheme > higher precision > fewer links > stability  
**Lock:** One final configuration from all OOF data, before test inference

### 8.5 Neural Models (Optional, after baseline)

| Component | Timing | Condition |
|---|---|---|
| Embedding features (bi-encoder) | Phase 10 | License ✓, ≤8B params, OOF gain |
| Dense ANN retrieval | Phase 10 | Incremental oracle F₀.₅ gain |
| Bi-encoder fine-tuning | Phase 10 | Group-safe folds, no external data |
| Cross-encoder reranking | Phase 11 | Material net OOF improvement |

All neural components are **features into LightGBM**, not standalone matchers.

---

## 9. Cross-Validation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FOLD CONSTRUCTION                          │
│                                                                 │
│  1. Build bipartite graph: S1 ↔ S2/S3 via ground-truth links  │
│  2. Compute connected components → fold groups                  │
│  3. Deterministic shuffled GroupKFold balanced on:              │
│     • entity count                                              │
│     • singleton status                                          │
│     • country                                                   │
│     • link cardinality                                          │
│  4. NEVER randomly split pair rows                              │
└─────────────────────────────────────────────────────────────────┘

Per fold execution:

  TRAINING FOLD                        VALIDATION FOLD
  ┌─────────────┐                      ┌─────────────┐
  │ Fit:        │                      │ Transform:  │
  │ • normalizer│    frozen artifacts  │ • normalize │
  │   vocabs    │ ──────────────────►  │ • retrieve  │
  │ • IDF       │                      │ • cap       │
  │ • vectorizer│                      │ • features  │
  │ • model     │                      │ • score     │
  └─────────────┘                      │ • threshold │
                                       │ • aggregate │
                                       │ • metric    │
                                       └─────────────┘
```

### OOF Artifact Inventory (per fold)

| Artifact | Purpose |
|---|---|
| Fold manifest + group IDs | Reproducibility |
| Fitted preprocessing/retrieval config | Frozen transforms |
| Final candidate pairs + provenance | Candidate evaluation |
| Blocked-out truth report | Blocking miss analysis |
| Pair features / version hash | Feature audit |
| Pair labels | Training verification |
| OOF raw + calibrated scores | Threshold optimization input |
| Per-S1 meta-features | Singleton model input |
| Cross-fitted singleton score | Entity decision input |
| Final prediction sets | Metric computation |
| Exact entity scores | Performance reporting |
| Blocking/candidate statistics | Blocking optimization |
| Runtime/memory logs | Resource planning |

### Leakage Prevention Checklist

- [ ] No entity crosses fold boundaries
- [ ] Fitted artifacts use training fold only
- [ ] Hard negatives mined from models that didn't train on that entity's group
- [ ] Singleton meta-features built from OOF pair scores only
- [ ] No truth label, target count, or fold assignment in features
- [ ] Validation candidates never negative-sampled

---

## 10. Inference Pipeline (Exact Sequence)

```
STEP  MODULE              ACTION
────  ──────              ──────
 1    common/config       Load frozen experiment config; verify artifact/data versions
 2    data/loading        Load test TSVs with sep="\t"; run contracts
 3    preprocessing       Generate normalized views with FROZEN normalizer
 4    blocking            Load/finalize S2+S3 retrieval indexes
 5    blocking            Retrieve candidates for S1 in batches
 6    blocking/union      Union → cap → deterministic dedup
 7    output/writers      Persist canonical candidate set
 8    output/writers      Aggregate → candidate_pairs.tsv (all S1)
 9    features/build      Build pair features with FROZEN vectorizers/IDF
10    models/pair_model   Score all candidates with pair model (+ neural if selected)
11    models/singleton    Aggregate pair scores → entity features → singleton model
12    models/thresholds   Apply LOCKED pair/entity/additional-link thresholds
13    output/writers      Aggregate zero/one/many IDs per S1
14    output/writers      Write matching_results.tsv (original S1 order)
15    validation/outputs  Run strict internal validator
16    (external)          Run official validate_submission.py --check-ids
17    common/artifacts    Write run manifest (hashes, counts, timestamps)
```

**Inference invariants:**
- No component accesses ground truth, OOF labels, or test-wide label statistics
- Unsupervised test-corpus transforms allowed only if explicitly designed & validated
- Conservative default: frozen training-fitted artifacts only

---

## 11. Output Format Specification

### 11.1 `matching_results.tsv` (scored on leaderboard)

```
source1_entity_id\tmatched_entity_ids
S1-00001\tS2-00047,S2-00193,S3-00812
S1-00002\tS3-00004
S1-00003\t
```

| Rule | Constraint |
|---|---|
| Columns | Exactly: `source1_entity_id`, `matched_entity_ids` |
| Rows | Exactly one per test S1 entity |
| Separator | Single tab |
| Singletons | `S1-ID<TAB><newline>` — empty value, never `None`/`nan`/`[]` |
| ID list | Comma-separated, unique existing S2/S3 IDs only |
| Ordering | Score-descending then ID tie-break (deterministic) |
| Encoding | UTF-8, LF newlines |
| Safety | No DataFrame index, no accidental quoting, atomic temp-file write |

### 11.2 `candidate_pairs.tsv` (not scored, used for audit)

```
source1_entity_id\tcandidate_entity_ids
S1-00001\tS2-00047,S2-00193,S3-00812,S3-00999
S1-00002\tS3-00004
S1-00003\t
```

Same format rules as `matching_results.tsv`, with column name `candidate_entity_ids`.

**Critical constraint:** Every ID in `matching_results.tsv` MUST appear in the same entity's `candidate_pairs.tsv` row (match ⊆ candidates).

### 11.3 Validation Stack

```
┌───────────────────────────────┐
│ Internal Python Validator     │  Fails on: missing/extra S1,
│ (strict, runs first)         │  duplicate IDs, wrong prefixes,
│                               │  match ∉ candidates, format
│                               │  errors, truncation, sentinel
│                               │  values, artifact hash mismatch
└───────────────┬───────────────┘
                │ must pass
                v
┌───────────────────────────────┐
│ Official validate_submission  │  python3 utils/validate_submission.py
│ (with --check-ids)           │  --matching ... --candidate ...
│                               │  --test-dir ... --check-ids
└───────────────────────────────┘
```

Pipeline stops packaging on any failure from either validator.

---

## 12. Configuration Architecture

### 12.1 YAML Config Schema

```yaml
# configs/baseline.yaml (illustrative structure from plan §22)

paths:
  data_root: /path/to/dataset
  output_dir: /path/to/output
  artifact_dir: /path/to/artifacts

normalization:
  version: "v1"
  # Controls for name/address/country normalization

retrieval:
  channels:
    exact_clean_name:
      enabled: true
    exact_core_name:
      enabled: true
      posting_cap: ...
    char_tfidf_name:
      enabled: true
      analyzer: "char_wb"
      ngram_range: [3, 5]
      min_df: ...
      max_df: ...
      sublinear_tf: true
      dtype: float32
      top_k_per_source: 20
      score_floor: null     # set after recall analysis
    char_tfidf_address:
      enabled: true
      # similar params
    rare_token:
      enabled: true
      df_ceiling: ...
      posting_cap: ...
  final_entity_cap: ...
  per_source_quotas: ...

features:
  version: "v1"
  enabled_groups: [name, address, cross_field, retrieval]
  expensive_features:
    monge_elkan: false      # enable after benchmark
    cross_encoder: false    # enable after Phase 11

folds:
  n_folds: 5
  assignment_seed: 42

negative_sampling:
  policy: "mixed"
  hard_negative_ratio: ...
  # hard-negative mining params

model:
  pair:
    type: "lightgbm"
    params:
      max_depth: ...
      num_leaves: ...
      learning_rate: ...
    early_stopping_rounds: ...
    calibration: null       # isotonic | platt | null
  singleton:
    enabled: true
    # params

thresholds:
  entity_threshold: ...     # from OOF optimization
  first_match_threshold: ...
  additional_match_threshold: ...

seeds:
  python: 42
  numpy: 42
  fold: 42
  lgbm: 42
  sampling: 42

resources:
  batch_size: ...
  worker_threads: ...
  memory_budget_gb: ...

neural:                      # optional
  model_id: null
  revision: null
  max_length: ...
  batch_size: ...
  device: "cpu"

output:
  ordering: "score_desc_then_id"
  validation_mode: "strict"
```

### 12.2 Per-Run Reproducibility Manifest

Every run saves:

| Artifact | Purpose |
|---|---|
| Fully resolved config YAML | Exact parameters |
| Git commit + worktree-diff hash | Code version |
| Package lock / version report | Dependency versions |
| Environment / hardware summary | Compute context |
| Data manifest hashes | Input integrity |
| Feature manifest hash | Feature contract |
| Fold mapping | CV assignment |
| All library seeds | Reproducibility |
| Thread counts | Determinism |
| Artifact checksums | Output integrity |

---

## 13. Experiment Tracking Architecture

```
experiments/
├── experiments.csv              # append-only ledger
└── manifests/
    ├── exp_001_baseline.yaml
    ├── exp_002_add_address.yaml
    └── ...
```

### Required CSV Columns

| Category | Columns |
|---|---|
| Identity | `experiment_id`, timestamp, git commit/diff hash, data-manifest hash, seed |
| Normalization | Version |
| Blocking | Config, retrieval/vectorizer versions, total candidates, mean/median/P95/P99/max per S1 |
| Blocking metrics | Micro recall, complete/any entity recall, reduction ratio, oracle macro F₀.₅ |
| Features | Feature-set version |
| Model | Type, hyperparameters, training rows, negative-sampling policy |
| Pair metrics | Precision/recall/F₀.₅ at chosen threshold (diagnostics) |
| Entity metrics | Macro F₀.₅ overall / by fold / by country |
| Singleton | Accuracy, error counts, macro-F₀.₅ delta from gating |
| Thresholds | Values and selection folds |
| Resources | Wall time, peak memory, hardware/thread settings |
| Provenance | Artifact paths/checksums, license status, notes, parent experiment |

### Strict Ablation Order

```
 1. exact/core/char-name blocking
 2. + address retrieval
 3. + rare-token (then optional word retrieval)
 4. baseline lexical features + LightGBM
 5. + address/numeric features
 6. + cross-field/retrieval features
 7. hard-negative mining
 8. singleton model + joint thresholds
 9. embedding features
10. dense retrieval
11. cross-encoder
12. ensembles
```

**Rule:** Change one conceptual component at a time. Promote only for repeatable cross-fitted macro-F₀.₅ gain.

---

## 14. Performance & Memory Architecture

### 14.1 Bottleneck Priority (likely order)

| Priority | Bottleneck | Mitigation |
|---|---|---|
| 1 | Char n-gram vocabulary/matrices + top-K over ~10M targets | CSR float32; batch queries; top-N sparse multiplication |
| 2 | Candidate explosion + disk I/O | Per-channel K, posting caps, final cap, incremental writes |
| 3 | Edit-distance feature computation | RapidFuzz batch; parallel partitions; tier 2 only if justified |
| 4 | LightGBM training rows | Controlled negative sampling; bounded training set |
| 5 | Python-set ID validation at scale | Sorted joins; integer/hash indexes; SQLite/DuckDB |
| 6 | Dense embedding / cross-encoder inference | Batch; GPU if available; restrict to prefiltered subset |

### 14.2 Memory Strategy

```
DO:                                     DON'T:
 ├── CSR sparse float32 matrices         ├── Densify corpus matrices
 ├── Fit vectorizers once, reuse         ├── Materialize query × corpus
 ├── Batch S1 queries                    ├── Hold all candidates in memory
 ├── Partition candidate artifacts       ├── Python nested loops over pairs
 ├── Incremental score writing           ├── Load all features at once
 ├── Stream comma-list TSV at output     ├── Naive Python set for all IDs
 └── Log peak RSS per batch              └── Oversubscribe threads
```

---

## 15. Acceptance Gates (Release-Blocking)

| Gate | Criteria |
|---|---|
| **Data** | Every file: expected UTF-8 TSV schema; source prefixes, ID uniqueness, ground-truth coverage/existence, graph properties verified |
| **Normalization** | Deterministic, idempotent, Unicode/missing/abbreviation tests pass; raw fields preserved |
| **Blocking** | Every validation S1 has candidate record; recall/reduction/oracle F₀.₅ reported; resource limits met |
| **Leakage** | Fold/component audit passes; no entity crosses folds; fitted artifacts + hard negatives have traceable lineage |
| **Metric** | Exact macro-F₀.₅ unit tests pass (singletons, multi-links) |
| **OOF** | Every train S1 has exactly one cross-fitted decision; no validation candidates negative-sampled away |
| **Model** | Promoted features/components show repeatable OOF improvement or documented resource win |
| **Inference** | All matches ⊆ canonical candidates; all S1 entities have exactly one output row; no invalid/duplicate IDs |
| **Validation** | Strict internal validator + official validator with `--check-ids` pass |
| **Reproduction** | Clean documented command regenerates identical validated outputs |
| **Compliance** | No external lookup; all deps/models pinned + licensed; pretrained ≤8B parameters |

**Rule:** Do not proceed to a costlier phase when its dependency gate is failing.

---

## 16. Submission Package Architecture

```
chimera_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (uploaded to leaderboard)
│   └── candidate_pairs.tsv         # blocking candidate set (for audit)
├── code/
│   └── business_entity_resolution/
│       ├── configs/
│       │   └── selected.yaml       # locked configuration
│       ├── src/                     # all source code
│       │   ├── main.py
│       │   ├── data/
│       │   ├── preprocessing/
│       │   ├── blocking/
│       │   ├── features/
│       │   ├── models/
│       │   ├── validation/
│       │   ├── inference/
│       │   ├── output/
│       │   └── common/
│       ├── tests/
│       ├── README.md               # exact reproduction instructions
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # filled methodology write-up
```

---

## 17. Risk Architecture

### Risk Heat Map (from plan §24)

| Risk | Impact | Probability |
|---|---|---|
| Blocking misses true links | Irrecoverable recall loss | High |
| False-positive merges | Severe F₀.₅ precision loss | High |
| Singleton overmatching | 1→0 score per entity | High |
| Open-set country shift (France) | Degraded/missing outputs | High |
| Address ambiguity/partialness | Incorrect decisions | High |
| Candidate explosion | OOM / excessive runtime | High |
| Memory-heavy ID validation | Skipped integrity checks | High |
| Over-aggressive normalization | False merges / lost distinctions | Medium |
| Pair-model overfitting | Inflated CV | Medium |
| Threshold overfitting | Unstable private score | Medium |
| Leakage | Invalid optimistic validation | Medium |
| Output-format errors | Submission rejection | Medium |
| Candidate/match artifact drift | Matches absent from candidates | Medium |
| Neural model adds no value | Wasted time/compute | Medium |
| License/parameter violation | Disqualification | Low-Medium |
| Accidental external data | Disqualification | Low |
| Unseen duplicate/shared targets | Fold leakage | Unknown |

### Mitigation Architecture

```
Detection Layer:
  ├── Phase 1 data audit catches schema/ID/graph issues
  ├── Blocking metrics catch recall gaps per fold/country
  ├── OOF scoring catches model overfitting/leakage
  ├── Singleton confusion matrix catches gating errors
  ├── Country-shift tests catch France degradation
  └── Dual validators catch output format errors

Prevention Layer:
  ├── Multi-view normalization (never over-normalize)
  ├── Connected-component folds (no entity leakage)
  ├── Hard negative mining (better false-positive discrimination)
  ├── Cross-fitted threshold selection (no threshold overfitting)
  ├── Canonical candidate artifact (no drift between model and output)
  ├── License manifest (compliance before download)
  └── Offline pipeline (no external data paths)
```

---

## 18. Data Flow Diagram (Complete)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE DATA FLOW                             │
│                                                                          │
│  RAW TSVs ──► LOADER ──► CONTRACT CHECKS ──► DATA MANIFEST             │
│                               │                                          │
│                               ▼                                          │
│                          NORMALIZER                                      │
│                    ┌──────────┼──────────┐                               │
│                    ▼          ▼          ▼                                │
│                 raw       clean      folded     core/alias               │
│                views      views      views      views                    │
│                    │          │          │          │                     │
│                    └──────────┼──────────┘          │                    │
│                               │                     │                    │
│                               ▼                     ▼                    │
│              ┌─────────────────────────────────────────┐                 │
│              │         BLOCKING CHANNELS               │                 │
│              │  exact│tfidf│rare_token│(optional)       │                 │
│              └──────────────────┬──────────────────────┘                 │
│                                 │                                        │
│                                 ▼                                        │
│                    UNION → DEDUP → CAP                                  │
│                                 │                                        │
│                    ┌────────────┼────────────┐                           │
│                    ▼                         ▼                            │
│           candidate_pairs.tsv     FEATURE BUILDER                       │
│                                        │                                 │
│                                        ▼                                 │
│                              PAIR MODEL (LightGBM)                      │
│                                   + optional neural                     │
│                                        │                                 │
│                                        ▼                                 │
│                              PER-S1 AGGREGATION                         │
│                                        │                                 │
│                                        ▼                                 │
│                              SINGLETON MODEL                            │
│                                        │                                 │
│                                        ▼                                 │
│                              THRESHOLD DECISION                         │
│                                        │                                 │
│                                        ▼                                 │
│                           matching_results.tsv                          │
│                                        │                                 │
│                                        ▼                                 │
│                         INTERNAL + OFFICIAL VALIDATORS                  │
│                                        │                                 │
│                                        ▼                                 │
│                              RUN MANIFEST                               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 19. Exact Competition Metric Implementation

```python
def entity_f05(true_ids: set[str], predicted_ids: set[str]) -> float:
    """Per-entity F₀.₅ score."""
    if not true_ids:
        return 1.0 if not predicted_ids else 0.0
    if not predicted_ids:
        return 0.0
    tp = len(true_ids & predicted_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(truth_by_s1, predictions_by_s1) -> float:
    """Macro-averaged F₀.₅ over all S1 entities."""
    assert set(truth_by_s1) == set(predictions_by_s1)
    scores = [
        entity_f05(set(truth_by_s1[s1]), set(predictions_by_s1[s1]))
        for s1 in truth_by_s1
    ]
    return sum(scores) / len(scores)
```

This exact implementation is used for **every** blocking, feature, model, threshold, singleton, and ensemble decision throughout the pipeline.

---

## 20. Constraints Summary

| Constraint | Source | Enforcement |
|---|---|---|
| TSV format (tab-separated) | README | Loader contract |
| ≤8B parameters, MIT/Apache 2.0 license | README, plan | License manifest + pre-download check |
| No external data/APIs/geocoding | README | Offline pipeline; code audit |
| Open-set country (no hard-coding) | README, plan | No fixed vocabulary; relational features only |
| Every test S1 in output | README | Internal + official validator |
| Matches ⊆ candidates | README, plan | Hash/subset assertion |
| No duplicate IDs | README | Validator checks |
| Only existing S2/S3 IDs | README | ID existence validation |
| F₀.₅ metric (precision-heavy) | README | Exact implementation + unit tests |
| `candidate_pairs.tsv` = final scored set | README, plan | Single canonical artifact |
