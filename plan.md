# Amazon ML Challenge 2026 — Business Entity Resolution Technical Plan

**Team:** `chimera`  
**Document purpose:** implementation-ready design; no pipeline or model code is implemented by this document.  
**Primary objective:** maximize the exact entity-level macro F0.5 score while producing a reproducible, auditable submission that uses only the supplied data.

## Scope and non-negotiable constraints

- Source 1 (S1) is the deduplicated reference. For every S1 record, predict zero, one, or multiple matching records from the union of Source 2 (S2) and Source 3 (S3).
- Read and write all competition files as UTF-8 TSV with an explicit `sep="\t"`.
- Preserve and submit one row for every test S1 entity, including France and predicted singletons.
- Optimize the competition's exact entity-level macro F0.5, not pair accuracy, ROC-AUC, or a proxy metric.
- Treat country as an open-set string. No India/US-only branches, fixed country vocabulary, or mandatory same-country filtering are allowed.
- Do not use external entity databases, business lookup, geocoding, registries, APIs, web search, or internet-derived record augmentation.
- Any pretrained model considered later must be no larger than 8B parameters and its model weights, code, and tokenizer licenses must be verified as MIT or Apache 2.0 before use.
- `candidate_pairs.tsv` must describe the final deduplicated candidate set actually scored by the model, not an earlier retrieval pool.
- Baseline components are marked **Baseline**. Components marked **Optional experiment** are added only after an out-of-fold (OOF) ablation demonstrates value relative to runtime and memory cost.

## 1. Problem understanding

Let $Q = \{q_i\}_{i=1}^{N}$ be S1, let $D = S2 \cup S3$, and let $T_i \subseteq D$ be the ground-truth match set for S1 entity $q_i$. The system must emit a predicted set $P_i \subseteq D$. The cardinality of both $T_i$ and $P_i$ may be zero, one, or greater than one; the task is not one-to-one assignment and must not enforce a global matching constraint.

Practically, each S1 record is a query against millions of S2/S3 records. A successful system must:

1. retrieve a small, high-recall set $C_i \subset D$;
2. estimate match evidence for every pair $(q_i, c), c \in C_i$;
3. decide whether the entity is a singleton at all; and
4. select any number of valid links from the scored set.

This is therefore a **retrieval + pairwise matching + entity-level set decision** problem. A plain binary classifier does not solve candidate discovery, cannot recover a true link omitted by blocking, and does not by itself decide how many links an entity should receive. A multiclass or winner-takes-all model is also invalid because one S1 can have multiple correct S2/S3 links.

For a non-singleton entity:

\[
p_i = \frac{|P_i \cap T_i|}{|P_i|}, \qquad
r_i = \frac{|P_i \cap T_i|}{|T_i|}, \qquad
F_{0.5,i} = \frac{1.25 p_i r_i}{0.25 p_i + r_i}.
\]

The final score is $N^{-1}\sum_i F_{0.5,i}$. For a true singleton, predicting an empty set scores 1 and any non-empty set scores 0. For a non-singleton, predicting no links scores 0. Because beta is 0.5, precision has greater influence than recall: speculative links, especially on singletons, are expensive. At the same time, candidate generation must retain true links because downstream models cannot recover a blocked-out match.

Confirmed training labels contain 123,247 singletons (5.5848%), 7,638,365 total links, a mean of 3.461 links per S1, and a maximum of 11 links. The system must therefore model both singleton status and multi-link output explicitly.

## 2. Repository assessment

### 2.1 Current structure and confirmed data inventory

| Path | Current state | Assessment / future action |
|---|---|---|
| `README.md` | Full supplied problem statement | Keep as challenge reference; do not mix implementation documentation into it. |
| `dataset/train/*.tsv`, `dataset/test/*.tsv` | Complete local dataset; ignored by Git | Input only. Phase 1 must produce reproducible profile artifacts without committing raw data. |
| `utils/validate_submission.py` | Supplied, working stdlib validator | Retain unchanged. It checks formatting and coverage; ID existence is optional and candidate-subset failure is only a warning, so add a stricter internal validator later. |
| `chimera_submission/code/business_entity_resolution/src/main.py` | Empty | Replace later with a thin CLI orchestrator, not business logic. |
| `chimera_submission/code/business_entity_resolution/src/notebook.ipynb` | Empty zero-byte file, not valid notebook JSON | Remove or replace only during implementation; experiments should live outside production `src/`. There is currently no notebook logic to preserve. |
| `chimera_submission/code/business_entity_resolution/requirements.txt` | One blank byte | Populate later with pinned, license-reviewed dependencies. |
| `chimera_submission/code/business_entity_resolution/README.md` | Minimal expected CLI only | Expand after implementation with exact train/inference/reproduction commands. |
| `chimera_submission/output/*.tsv` | Headers only | Safe placeholders, not predictions. |
| `chimera_submission/Documentation_template.md` | Supplied empty methodology template | Fill only after experiments select the final pipeline. |
| `plan.md` | This document | Source of truth for implementation order and acceptance gates. |

The Git worktree already contains user changes, including moved scaffold files and an untracked notebook/plan. Future work must preserve those changes and avoid broad cleanup. `.gitignore` excludes `dataset/`, which is appropriate. `.gitattributes` still points at the old `student_resource/dataset/**` LFS path; this stale path is low priority because the current dataset is ignored, but should be cleaned during packaging if it causes confusion.

There is currently no implemented loading, preprocessing, blocking, feature, modeling, CV, metric, inference, or internal validation component. There is also no duplicated implementation logic yet. The technical-debt risk is prospective: one-off EDA/notebook logic must not become a second implementation of production transformations. Any experiment notebook created later must call reusable modules from `src/`; successful logic moves into tested modules rather than being copied.

### 2.2 Dataset facts confirmed by direct TSV inspection

| Split/source | Rows | Countries | Missing names | Missing addresses | Missing country |
|---|---:|---|---:|---:|---:|
| train S1 | 2,206,821 | US 1,323,633; India 883,188 | 0 | 0 | 0 |
| train S2 | 5,034,616 | US 3,016,817; India 2,017,799 | 0 | 168,967 | 0 |
| train S3 | 5,285,603 | US 3,170,056; India 2,115,547 | 0 | 175,916 | 0 |
| test S1 | 1,732,544 | US 663,106; India 809,986; France 259,452 | 0 | 0 | 0 |
| test S2 | 4,887,273 | US 1,871,330; India 2,312,565; France 703,378 | 0 | 129,408 | 0 |
| test S3 | 5,082,316 | US 1,945,701; India 2,405,000; France 731,615 | 0 | 136,098 | 0 |

All inspected source rows have exactly four TSV fields and the expected S1/S2/S3 prefix. Ground-truth row count equals train S1 row count; exact ID-set equality and global ID uniqueness still require explicit Phase 1 integrity checks. The raw comparison spaces are approximately 22.8 trillion train pairs and 17.3 trillion test pairs, making Cartesian feature computation infeasible.

## 3. End-to-end system architecture

```text
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

### 3.1 Core architecture decisions

| Decision | Choice and rationale | Rejected alternative |
|---|---|---|
| Search strategy | Union of independent sparse/lexical retrievers. Different noise modes require different recovery paths. | Cartesian comparison is infeasible; a single exact key has an unacceptably low and brittle recall ceiling. |
| Initial retrieval | Character n-gram TF-IDF plus exact and rare-token channels. Character n-grams tolerate typos, punctuation, transliteration fragments, and word-boundary variation without country-specific rules. | Dense-only retrieval is costly, harder to audit, and not justified before a lexical baseline. Word-only retrieval is fragile to spelling noise. |
| Matcher | LightGBM on explicit pair features. It handles nonlinear interactions, missingness, mixed feature scales, large tabular data, and has an MIT license. | Pure rules are hard to calibrate; deep end-to-end matching adds cost before establishing a strong baseline; top-1 similarity ignores multi-match semantics. |
| Final decision | Pair scorer followed by entity-level singleton model and OOF-tuned thresholds. | Default 0.5 and pair-independent decisions do not optimize macro F0.5 or singleton behavior. |
| Country | Generic agreement/missing/mismatch features and optional dynamically keyed retrieval channel, always with a global fallback. | Fixed country one-hot vectors or `if India/US` rules fail on France and future labels. Hard country filtering can delete true pairs when labels are noisy. |
| Data representation | Raw plus several conservative normalized views. | One aggressively normalized string irreversibly destroys discriminative evidence. |
| Validation | Entity/connected-component folds with the whole retrieval-to-output pipeline run OOF. | Random pair splits leak entity variants and overstate generalization. |
| Neural models | Optional signals after baseline, gated by ablation, license, parameter, memory, and runtime checks. | A pure Transformer stack may miss numeric/address structure and makes full-corpus retrieval expensive. |

## 4. Data loading and data contracts

### 4.1 Loading rules

- Use `pandas.read_csv(path, sep="\t", dtype="string", keep_default_na=False, encoding="utf-8")` or an equivalent streaming loader with exactly the same semantics.
- Never infer the separator, numeric types, or null markers. Entity IDs and postal/address numbers are strings.
- Validate the header before expensive loading. Reject extra/missing columns, malformed row widths, invalid UTF-8, blank IDs, unexpected prefixes, and duplicate `entity_id` values within or across target sources.
- Verify source-file membership agrees with the ID prefix.
- Verify train ground truth contains exactly one row for every train S1 ID, no unknown S1 IDs, no repeated link IDs per row, and only existing S2/S3 target IDs.
- Record row counts, null/blank counts, distinct ID counts, country counts, and file hashes in a data manifest.
- Canonicalize empty fields internally to one representation (nullable string or `""`) while retaining explicit missing flags. Do not turn the literal text `"None"` into an output value.
- Preserve input order for output assembly, but use `entity_id` as the stable join key.

### 4.2 Stable internal schemas

**Normalized entity record** (one row per source record):

| Field | Type | Meaning |
|---|---|---|
| `entity_id` | string, non-null, unique in source | Original identifier. |
| `source` | categorical/string | Derived and validated as `S1`, `S2`, or `S3`. |
| `business_name_raw` | string | Exact supplied value. |
| `business_name_clean` | string | Conservative Unicode/punctuation/whitespace view. |
| `business_name_folded` | string | Accent-folded companion view. |
| `business_name_core` | string | Conservative suffix-normalized view. |
| `business_address_raw` | string | Exact supplied value. |
| `business_address_clean` | string | Conservative normalized address. |
| `business_address_folded` | string | Accent-folded companion view. |
| `business_address_alias` | string | Safe alias/abbreviation representation. |
| `country_raw`, `country_clean` | string | Original and generic normalized label. |
| `name_missing`, `address_missing`, `country_missing` | bool | Explicit missingness. |

**Final candidate pair record** (long form, one row per unique S1-target pair):

- keys: `source1_entity_id`, `candidate_entity_id`, `candidate_source`;
- retrieval provenance: per-channel hit flags, ranks, similarities, best rank/score, number of channels;
- deterministic `candidate_order` for batching/output;
- `is_match` only in labeled training artifacts, derived from a set-membership join after candidate generation;
- no raw label-derived field may survive into inference features.

**Pairwise feature record:** pair keys followed by versioned numeric/boolean feature columns; finite values only; missingness indicators accompany imputed numeric values. A feature-manifest hash fixes column order and definitions.

**Pair model output:** pair keys, `pair_score_raw`, optional OOF-fitted calibrated score, model/fold/version IDs. Scores are not final links.

**Entity decision record:** S1 ID, candidate count, score summary features, singleton probability/score, selected threshold version, ordered predicted ID list, and a reason code such as `singleton_gate`, `above_pair_threshold`, or `no_candidates`.

## 5. Text normalization strategy

Normalization must be deterministic, tested, Unicode-safe, and multi-view. It must improve comparison while preserving evidence for the model.

### 5.1 Shared conservative operations

1. Convert missing input to an empty internal string and set a missing flag.
2. Apply Unicode NFKC normalization to standardize compatibility characters without transliterating scripts.
3. Apply Unicode `casefold()` rather than ASCII-only lowercasing.
4. Normalize typographic apostrophes/dashes and punctuation to spaces or stable tokens where safe; normalize `&` through an additional alias view rather than overwriting raw evidence.
5. Collapse repeated whitespace and trim.
6. Produce both accent-preserving and accent-folded forms. Accent folding uses Unicode decomposition and removes combining marks only in the folded view; it never replaces the primary clean form.
7. Never transliterate all non-Latin text to ASCII as the sole representation.

### 5.2 Name views

- `business_name_raw`: untouched input.
- `business_name_clean`: NFKC + case-fold + conservative punctuation/space normalization.
- `business_name_folded`: accent-folded clean name.
- `business_name_core`: remove or canonicalize only a reviewed, configurable set of legal/corporate suffix tokens (for example, equivalent forms of limited/corporation/private) from terminal positions. Retain flags and counts describing removed suffixes.
- token multiset and character n-gram views derived from clean/folded forms.

Corporate suffix rewriting is a weak equivalence signal, not truth: suffixes can be part of a distinctive name, and unseen languages may have unknown forms. Exact-clean, core, and raw-derived similarity features remain side by side.

### 5.3 Address views

- `business_address_raw`: untouched input.
- `business_address_clean`: NFKC + case-fold + conservative punctuation/space normalization.
- `business_address_folded`: accent-folded companion.
- `business_address_alias`: an additional representation using only unambiguous, reviewed address aliases and stable number formatting.
- extracted token sets: alphabetic, numeric, and alphanumeric tokens; extraction is generic and makes no assumptions about postal-code length or country format.

Ambiguous tokens such as `st` must not be blindly rewritten to `street`: it can mean street, saint, a name fragment, or something language-specific. Where useful, preserve both the original token and an expanded hypothesis in a separate representation. Do not parse addresses into mandatory US/India-specific components.

### 5.4 Tests

Golden unit tests must cover idempotence, blanks, composed/decomposed accents, French diacritics, Devanagari, punctuation, ampersands, apostrophes, digit-letter tokens, repeated whitespace, and ambiguous abbreviations. The raw columns must be byte-for-byte recoverable from the loaded values.

## 6. Open-set country handling

Country is normalized generically with NFKC, case-folding, and whitespace cleanup. Pair features are relational rather than vocabulary-specific:

- `country_exact_match`;
- `country_both_missing`;
- `country_left_missing` / `country_right_missing`;
- `country_mismatch`.

If country-partitioned retrieval is used for efficiency, partitions are created dynamically from values present at runtime and are a retrieval channel, not the only search path. A global lexical fallback must remain so a mislabeled or unseen-country pair is not impossible.

Validation must report performance per observed country and run explicit shift tests: train the scorer mostly/entirely on one observed training country and evaluate on the other; train on one and validate on the other where fold sizes permit; mask country features; and compare country-filtered retrieval with global fallback. France has no train labels, so test-France behavior is assessed through invariance tests, candidate diagnostics, score distributions, and conservative threshold sensitivity—not invented pseudo-label performance.

## 7. Candidate generation / blocking architecture

Candidate generation defines the attainable recall and must be evaluated independently from matching.

### 7.1 Baseline union channels

Run each channel against S2 and S3, retain provenance, then union and deduplicate by `(source1_entity_id, candidate_entity_id)`:

1. **Exact normalized name**: inverted index on `business_name_clean`; cheap and high precision.
2. **Exact core name**: inverted index on `business_name_core`; recovers legal-suffix variants but is separately capped for very common cores.
3. **Character n-gram TF-IDF name KNN**: primary fuzzy retriever over clean and/or folded names. Tune `char`/`char_wb` and n-gram ranges on blocking metrics.
4. **Character n-gram TF-IDF address KNN**: independent recovery path for renamed/noisy businesses; skip blank addresses safely.
5. **Rare-token inverted index**: retrieve candidates sharing high-IDF name/address tokens; ignore tokens above a configurable document-frequency ceiling and cap postings.

Character n-grams are the strongest likely initial fuzzy baseline because they tolerate local edits, punctuation, spacing, affixes, and partial overlap while remaining sparse and auditable. This is a hypothesis to validate, not a guaranteed result.

### 7.2 Later experiments

- **Word-level TF-IDF KNN:** useful for reordered intact tokens but likely weaker alone under typos; add only as a union channel if it improves oracle score.
- **MinHash/LSH:** scalability experiment for token/shingle Jaccard; retain only if recall/runtime/memory beats sparse alternatives.
- **Multilingual dense bi-encoder ANN:** potential semantic/transliteration recovery; only after license review and lexical baseline, with an auditable ANN index and measured incremental recall.

### 7.3 Candidate controls and persistence

- Configure top-K per field, source, and representation; initial experiments should include K values such as 5, 10, 20, and 40 rather than assuming one setting.
- Apply similarity floors per channel only after recall analysis. Never use a single threshold copied between name and address spaces.
- Prevent exact-block explosion: if an exact key has a very large posting list, rank within it using another field and retain a configured maximum, while measuring lost truths.
- Deduplicate before feature generation and keep `retrieval_channel_count`, per-channel rank/score, and a provenance bitset.
- Apply a deterministic per-S1 final cap after union. Preserve high-confidence exact hits first, then allocate bounded quotas/diversity across fuzzy channels. Measure the oracle loss caused by this cap.
- Process queries in batches and write partitioned long-form candidate artifacts incrementally. Do not materialize every candidate pair as Python objects.
- Fit vectorizers only on the permitted training partition for strict CV; serialize them for final inference. Transform corpus/query blocks into CSR sparse matrices and use a top-N sparse similarity implementation or benchmarked ANN method rather than a dense all-pairs matrix.
- The post-union, post-cap table is the canonical final candidate set. Both the pair scorer and `candidate_pairs.tsv` must be generated from this same immutable artifact.

## 8. Blocking evaluation

For validation entities with candidate sets $C_i$:

- **True-pair blocking recall (micro):** 
  \(\sum_i |C_i \cap T_i| / \sum_i |T_i|\).
- **Entity complete-recall rate:** fraction of non-singleton entities for which $T_i \subseteq C_i$.
- **Entity any-recall rate:** fraction of non-singletons with at least one retrieved truth.
- **Reduction ratio:** 
  \(1 - \sum_i |C_i| / (|Q|\,|D|)\).
- **Candidate load:** mean, median, P95, P99, maximum candidates per S1, total pairs, and zero-candidate rate; break all metrics down by source and country.
- **Oracle macro F0.5 ceiling:** set $P_i=C_i\cap T_i$. True singletons receive 1; missed non-singletons receive 0; partially retrieved entities receive their exact F0.5 with oracle precision 1. Macro-average those entity scores.

Oracle macro F0.5 is more decision-relevant than an arbitrary “99.9% recall” target: it accounts for which entities lose links, multi-link cardinality, and the macro scoring structure. Report both because micro recall can conceal complete misses on many entities.

Run a blocking matrix over name/address K combinations, channel unions, final caps, and per-source quotas. Record incremental recall and candidate cost for each added channel. Select the smallest configuration whose OOF oracle F0.5 is near the best observed ceiling and whose memory/runtime fits final inference.

## 9. Training pair construction

1. Parse each ground-truth cell into a set and explode it into positive `(S1, target)` pairs.
2. Run the exact production blocking configuration in each CV fold.
3. Label candidate pairs by set membership. A target that is another true match of the same S1 is always positive; never sample it as a negative.
4. Record blocked-out positives separately. Do not silently inject them into validation candidates, since that would inflate blocking evaluation. They may be injected into the **training** pair set for scorer learning only under a clearly named `positive_rescue` policy, with retrieval features set honestly; compare with no rescue.
5. Split candidate negatives into:
   - easy negatives: low lexical similarity/random candidates from the retrieved set;
   - hard negatives: top-ranked false candidates, same/near-exact name but conflicting address, strong address but conflicting name, and highest model-scored false pairs.
6. Sample negatives per S1 with a reproducible mixture that retains score/rank diversity. Keep an unsampled candidate validation set for honest threshold evaluation.

Hard negatives matter because false merges dominate the F0.5 penalty and random negatives make the classification task unrealistically easy. After the first model, score all training candidates OOF, mine high-scoring false positives, retrain, and accept the iteration only if entity-level OOF F0.5 improves. Prevent feedback leakage by mining from predictions made by a model that did not train on that entity's group.

## 10. Feature engineering architecture

All features are pure functions of two records plus training-fitted artifacts. Pair IDs themselves are excluded from the model.

### 10.1 Name features

- exact matches on clean, folded, and core views: high-precision anchors;
- normalized Levenshtein and Jaro-Winkler: local edits, typos, and prefix agreement;
- token-set ratio and token-sort ratio: token reordering and subsets;
- token Jaccard and directional containment: shared-token fraction and partial names;
- Monge-Elkan token similarity: best approximate token correspondences, computed only after profiling its runtime;
- char and word TF-IDF cosine: corpus-aware fuzzy overlap;
- character length, token count, absolute/relative differences, and empty flags: similarity reliability context;
- IDF-weighted token overlap and rare-token overlap/count: distinctive shared words carry more evidence than common suffixes;
- corporate-suffix agreement/removal flags: preserve whether a core equality was created by normalization.

### 10.2 Address features

- normalized Levenshtein, Jaro-Winkler, token-set/sort, Jaccard, containment, and optional Monge-Elkan;
- char and word TF-IDF cosine;
- numeric-token intersection/union, exact set match, overlap ratio, subset in each direction, and conflict indicators;
- alphanumeric-token overlap for unit/building strings;
- generic numeric sequence agreement without assuming PIN/ZIP length;
- address missing on left/right/both, token-count/length differences, and whether similarity is undefined due to missingness.

Numbers often disambiguate otherwise similar street/business text, but numeric mismatch is not an automatic non-match because addresses may be partial or relocated.

### 10.3 Cross-field and retrieval features

- name × address similarity products and min/max/mean of primary scores;
- exact name × address score and exact address × name score;
- country relation flags × name/address evidence;
- candidate source (`S2` vs `S3`) and source × similarity interactions;
- missing-field interactions, especially exact name with missing address;
- retrieval channel flags, ranks, scores, reciprocal ranks, best rank, and number of agreeing channels;
- agreement/conflict summaries, such as strong name + numeric-address conflict.

Cross-field interactions let the model demand corroboration where ambiguity is high and relax one field when another is missing. Retrieval features capture how consistently independent channels found a pair but cannot replace content features.

Feature computation is tiered: cheap vectorized exact/token/length features first; expensive edit/Monge-Elkan features only on the final candidate set and only if ablations justify them. Every feature group receives a version, tests, runtime benchmark, and null/finite-value audit.

## 11. Baseline modeling strategy

The preferred serious baseline is **LightGBM** with binary objective over candidate pairs.

Reasons:

- strong performance on nonlinear tabular interactions and missingness;
- efficient histogram training and prediction at candidate scale;
- native categorical support if needed, though relational booleans are preferred;
- interpretable feature importance/SHAP diagnostics;
- MIT-licensed implementation.

CatBoost and XGBoost are valid ablation alternatives, not initial requirements. Start with bounded tree depth/leaves, early stopping on entity-aware validation, deterministic seeds, and logged thread counts. Use pair log-loss or average precision only as training diagnostics. Select the model and early-stopping round by exact end-to-end OOF macro F0.5, with pair precision/recall at selected thresholds as supporting metrics. Pair accuracy is dominated by negatives; ROC-AUC can look excellent while the few high-score false positives ruin singleton/entity precision.

Do not blindly apply a large `scale_pos_weight`. It changes score distributions and can damage the threshold/singleton calibration required by F0.5. Compare controlled negative sampling, modest weights, no weights, and optional OOF calibration (isotonic or Platt where sample support permits). Treat LightGBM output as a ranking score unless calibration is empirically verified; thresholds are learned on the exact downstream metric.

## 12. Neural modeling experiments

Neural work begins only after the classical OOF baseline, bottleneck profile, and error taxonomy exist.

1. **Embedding features:** encode clean name, address, and structured full-record text using a multilingual bi-encoder; add cosine similarities to LightGBM.
2. **Dense ANN retrieval:** query S2/S3 embedding indexes and union retrieved pairs with lexical candidates; measure incremental oracle F0.5 and candidate growth.
3. **Bi-encoder fine-tuning:** train on positives plus OOF hard negatives, using group-safe folds and no external entity data.
4. **Cross-encoder reranking:** concatenate tagged fields from S1 and candidate records, score only the final/safely prefiltered candidate set, and add the OOF cross-encoder probability/score as a LightGBM feature.

The hybrid lexical + structural + neural design is preferred over a pure Transformer matcher: exact tokens, numeric address evidence, missingness, and source/country relations are valuable and cheap; neural signals may recover semantic or transliteration variants but add latency, hardware needs, calibration risk, and less transparent failure modes.

Before downloading or using any model, record the exact model/revision, parameter count, weight license, code license, tokenizer license, source URL in a license manifest, and a cached artifact checksum. Proceed only when all challenge requirements are unambiguously satisfied. Pretrained weights are feature models, not permission to query external business information.

## 13. Entity-level singleton model

Train a second-stage binary model for (y_i = 1[|T_i|>0]), using only features available at inference:

- maximum and second-highest pair scores;
- top1-top2 difference as a weak feature, never a hard rule;
- mean, standard deviation, quantiles, and entropy/concentration of top-k scores;
- counts above several fixed score levels;
- maximum/second-highest name, address, and combined lexical similarities;
- number of exact-name/core-name hits and number of retrieval channels agreeing;
- candidate count, source mix, and zero-candidate flag;
- missing-field and country-relation summaries.

Singleton optimization is valuable because a single false positive changes a true singleton from 1 to 0, and a pairwise model may emit one plausible-looking candidate even when all are wrong. Conversely, overly aggressive singleton gating erases all credit for a true non-singleton.

To prevent leakage, all meta-features for training the singleton model must be built from OOF pair-model scores. For honest evaluation of the stack, use nested/cross-fitted meta-model predictions: train the entity model on OOF meta-features from other folds and predict the held-out fold. At final training, fit the singleton model on all OOF meta-features, then apply it to features from the pair model refit on all training data.

The top1-top2 margin is never used alone: two top candidates may both legitimately be correct, as the label cardinality confirms.

## 14. Threshold optimization

Never default to 0.5. Using OOF/cross-fitted scores, search jointly over:

- entity/singleton threshold (t_e): whether any match is allowed;
- first-match pair threshold (t_1);
- optional additional-match threshold (t_a), applied after the entity is judged non-singleton;
- optionally source-specific thresholds only if repeated OOF evidence shows stable benefit and no open-set brittleness.

For each tuple, reconstruct a prediction set for every S1 entity and compute the exact macro F0.5. Use a coarse grid followed by local refinement over observed score quantiles rather than assuming calibrated probabilities. Tie-breaking favors the simpler threshold scheme, higher precision, fewer links, and stability across folds/countries. Report fold mean, dispersion, worst fold, per-country score, singleton contribution, and sensitivity near the optimum.

Avoid threshold overfitting by reporting a cross-fitted estimate: for each evaluation fold, select thresholds using the other OOF folds and evaluate on the held-out fold. Lock one final configuration using all OOF data before test inference. Bootstrap entities to estimate uncertainty if time permits.

The decision function must return all candidate IDs passing the applicable thresholds. It must not be top-1-only. A lower or higher additional-match threshold is an empirical option because once an entity is confidently non-singleton, the risk structure changes; it is not assumed. A top1-top2 margin may inform the entity model but cannot suppress the second candidate automatically.

## 15. Exact competition metric

```python
def entity_f05(true_ids: set[str], predicted_ids: set[str]) -> float:
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
    # Both mappings must cover exactly the same complete S1 universe.
    assert set(truth_by_s1) == set(predictions_by_s1)
    scores = [
        entity_f05(set(truth_by_s1[s1]), set(predictions_by_s1[s1]))
        for s1 in truth_by_s1
    ]
    return sum(scores) / len(scores)
```

Unit tests must cover correct/incorrect singletons, empty predictions on non-singletons, zero-overlap predictions, duplicate-input rejection before set conversion, perfect match, false positives, false negatives, and multi-match examples. Every blocking, feature, model, threshold, singleton, and ensemble decision ultimately uses this exact implementation.

## 16. Cross-validation design

- Build a bipartite graph from S1-to-S2/S3 truth links. Use connected components as groups so any records connected to the same true entity remain in one fold. If integrity checks prove every target belongs to at most one S1, S1 ID grouping is equivalent, but retain the component implementation as the safe default.
- Use deterministic shuffled GroupKFold-style assignment balanced on entity count, singleton status, country, and link cardinality. Never randomly split pair rows.
- For each fold, fit learned normalization vocabularies/IDF/vectorizers/model artifacts on training-fold data only. Transform held-out records with frozen artifacts. Rule-based NFKC/case-fold operations require no fitting.
- Run the full held-out path: normalize -> retrieve against the defined S2/S3 corpus -> final candidate cap -> features -> pair score -> entity meta-features -> threshold -> aggregate -> exact metric.
- Keep the validation candidate population unsampled. Negative sampling is for model fitting only.
- Prevent the matcher from using entity IDs as features. Validate that no truth label, target count, or fold assignment leaks into transformations.

**OOF artifacts generated for every fold:** fold manifest and group IDs; fitted preprocessing/retrieval configuration; final candidate pairs with provenance; blocked-out truth report; full pair features or a version/hash reference; pair labels; OOF raw/calibrated scores; per-S1 meta-features; cross-fitted singleton score; final prediction sets; exact entity scores; blocking/candidate statistics; runtime/memory logs.

**Country-shift tests:** standard stratified folds; leave-US-out/leave-India-out scorer tests where feasible; train-mostly-one-country/evaluate-other; country-feature ablation; global-vs-country-channel blocking. Track feature drift and thresholds by country. France remains a genuinely unseen-label test domain, so the final design favors language/script-safe features and globally validated thresholds.

## 17. Experiment tracking

Maintain append-only `experiments/experiments.csv` plus one immutable YAML/JSON manifest per run.

Required columns:

- `experiment_id`, timestamp, Git commit/worktree-diff hash, data-manifest hash, seed;
- normalization version;
- blocking configuration and retrieval/vectorizer versions;
- total candidates and mean/median/P95/P99/max per S1;
- micro blocking recall, complete/any entity recall, reduction ratio, oracle macro F0.5;
- feature-set version;
- model, hyperparameters, training rows, negative-sampling policy;
- pair precision/recall/F0.5 at chosen threshold as diagnostics;
- entity macro F0.5 overall/by fold/by country;
- singleton accuracy, singleton-specific error counts, and macro-F0.5 delta from singleton gating;
- thresholds and threshold-selection folds;
- wall time, peak memory, hardware/thread settings;
- artifact paths/checksums, license status, notes, and parent experiment.

Strict ablation order:

1. exact/core/char-name blocking;
2. add address retrieval;
3. add rare-token then optional word retrieval;
4. baseline lexical features + LightGBM;
5. add address/numeric features;
6. add cross-field/retrieval features;
7. hard-negative mining;
8. singleton model and joint thresholds;
9. embedding features;
10. dense retrieval;
11. cross-encoder;
12. ensembles.

Change one conceptual component at a time. Promote it only for repeatable cross-fitted macro-F0.5 gain with acceptable memory/runtime; preserve negative results.

## 18. Inference pipeline

The exact test-time sequence is:

1. load selected frozen experiment config and verify artifact/data schema versions;
2. load test TSVs using explicit tab parsing and run contracts;
3. generate raw-preserving normalized views with the frozen normalizer;
4. load/finalize S2 and S3 retrieval indexes and retrieve candidates for S1 in batches;
5. union, cap, deterministically deduplicate, and persist the canonical final long-form candidate set;
6. aggregate that exact set into `candidate_pairs.tsv` for every S1;
7. build pair features using frozen vectorizers/IDF/feature manifest;
8. score all candidates with the selected pair model and any selected neural feature artifacts;
9. aggregate pair scores into entity features and apply the singleton model;
10. apply locked pair/entity/additional-link thresholds and aggregate zero/one/many IDs;
11. write `matching_results.tsv` in original S1 order;
12. run strict internal validation and then the provided official validator; write a run manifest with hashes and counts.

No inference component may access ground truth, OOF labels, test-wide label statistics, or any statistic unavailable when reproducing the submission. Unsupervised test-corpus transforms are allowed only if explicitly designed, validated without labels, saved in the inference manifest, and consistent with competition rules; the conservative default is frozen training-fitted artifacts.

## 19. Output semantics

### `matching_results.tsv`

- Exact columns: `source1_entity_id`, `matched_entity_ids`.
- Exactly one row for every test S1 in input order.
- `matched_entity_ids` is a deterministic comma-separated list of unique existing S2/S3 IDs; use stable score-descending then ID tie-break order.
- A singleton is `S1-ID<TAB><newline>`—an empty value, never `None`, `nan`, `[]`, or quoted whitespace.

### `candidate_pairs.tsv`

- Exact columns: `source1_entity_id`, `candidate_entity_ids`.
- Exactly one row for every test S1, including zero-candidate rows.
- IDs are the unique final post-union/post-filter/post-cap candidates actually passed to the matcher, not a preliminary blocking pool.
- Every predicted match must be present in the same entity's candidate list.

Write UTF-8, LF newlines, a single tab separator, no DataFrame index, no accidental quoting, and no duplicate IDs. Use atomic temporary-file replacement so interrupted jobs cannot leave apparently valid partial submissions.

## 20. Output validation

Create an internal Python validator that fails on:

- missing/extra S1 IDs, duplicate S1 rows, or row-count mismatch;
- duplicate candidate/match IDs within a row;
- IDs outside the loaded test S2/S3 sets or wrong S1/S2/S3 prefixes;
- any match not contained in its final candidate set;
- wrong headers, non-tab separation, extra columns, invalid UTF-8, malformed lines, or non-empty singleton sentinels;
- truncated files, unexpected ordering when deterministic mode is required, and mismatch with the canonical candidate artifact hash/counts.

At this scale, validate target existence with memory-aware sorted joins, integer/hash indexes, SQLite/DuckDB, or chunked membership structures rather than assuming Python sets fit. Then run:

```bash
python3 utils/validate_submission.py \
  --matching chimera_submission/output/matching_results.tsv \
  --candidate chimera_submission/output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

The provided validator's candidate-subset check is only a warning and its ID check is off by default, so a PASS without `--check-ids` is necessary but not sufficient. The pipeline must automatically execute both validators and stop packaging on any internal failure.

## 21. Proposed repository architecture

```text
chimera_submission/code/business_entity_resolution/
├── configs/
│   ├── baseline.yaml
│   └── selected.yaml
├── src/
│   ├── main.py
│   ├── data/
│   │   ├── loading.py
│   │   ├── contracts.py
│   │   └── schemas.py
│   ├── preprocessing/
│   │   ├── normalize.py
│   │   └── tokenization.py
│   ├── blocking/
│   │   ├── exact.py
│   │   ├── tfidf.py
│   │   ├── rare_tokens.py
│   │   ├── union.py
│   │   └── evaluate.py
│   ├── features/
│   │   ├── name.py
│   │   ├── address.py
│   │   ├── cross_field.py
│   │   └── build.py
│   ├── models/
│   │   ├── pair_model.py
│   │   ├── singleton_model.py
│   │   ├── thresholds.py
│   │   └── neural.py
│   ├── validation/
│   │   ├── metric.py
│   │   ├── folds.py
│   │   └── outputs.py
│   ├── inference/
│   │   └── pipeline.py
│   ├── output/
│   │   └── writers.py
│   └── common/
│       ├── config.py
│       ├── logging.py
│       └── artifacts.py
├── tests/
├── experiments/
│   ├── experiments.csv
│   └── manifests/
├── artifacts/                 # generated, not packaged unless selected/required
├── README.md
└── requirements.txt

chimera_submission/output/     # only final competition TSVs
```

Responsibilities:

- `data`: TSV I/O, source/ground-truth contracts, typed schemas, manifests; no normalization or modeling.
- `preprocessing`: deterministic raw-to-multi-view transformations and tokenization.
- `blocking`: retriever fitting/querying, union/dedup/caps, provenance, and blocking-only metrics.
- `features`: reusable vectorized pair transformations; no fold splitting or label decisions.
- `models`: pair training/scoring, singleton stacking, calibration, threshold selection, optional neural adapters.
- `validation`: exact metric, leakage-safe folds, strict submission checks.
- `inference`: orchestration of frozen components only.
- `output`: deterministic competition-format serialization.
- `common`: configuration validation, structured logs, artifact/version utilities.
- `tests`: unit/integration/golden tests.
- `experiments`: small metadata and manifests, not ad hoc production implementations.
- `artifacts`: generated vectorizers, indexes, models, features, and OOF files with checksums and cleanup policy.

`main.py` should parse CLI commands such as `profile`, `train`, `evaluate`, and `infer`, delegate to modules, and remain thin. Training logic and inference orchestration stay separate while calling the same normalization, blocking, and feature code.

## 22. Configuration and reproducibility

Use validated YAML configuration with explicit schema and no hidden notebook defaults. It controls:

- input/output/artifact paths relative to a declared project/data root;
- normalization and feature versions;
- char/word TF-IDF analyzer, n-gram range, `min_df`, `max_df`, feature cap, sublinear TF, dtype;
- enabled retrieval channels, per-source/per-channel K, score floors, posting caps, and final entity cap;
- enabled feature groups and expensive-feature switches;
- fold count/assignment seed and all library seeds;
- negative sampling and hard-negative parameters;
- LightGBM parameters, early stopping, calibration choice;
- singleton-model parameters and threshold grids;
- batch sizes, worker/thread limits, memory budget;
- optional neural model revision, max length, batch size, and device;
- output ordering and validation mode.

For every run, save the fully resolved config, Git state, package lock/version report, environment/hardware summary, data hashes, feature manifest, fold mapping, logs, and artifact checksums. Serialize normalizers, TF-IDF vocabularies/IDF arrays, retrieval indexes or rebuild manifests, feature-column order, pair model, singleton model, calibrator, and locked thresholds as one versioned selected-experiment bundle.

Set seeds for Python, NumPy, fold assignment, LightGBM, sampling, and neural frameworks. Fix thread counts where reproducibility matters and document any nondeterministic GPU operations. The final README must provide one documented inference command matching the expected interface:

```bash
python src/main.py --data-dir /path/to/dataset --output-dir /path/to/output
```

If training is a separate command, document it too; final inference must be reproducible from the packaged selected artifacts or from an explicitly documented end-to-end train-and-infer sequence.

## 23. Performance and memory strategy

- Store TF-IDF data as CSR sparse matrices, preferably `float32`; never densify corpus matrices.
- Fit once per fold/config and reuse vectorizers for retrieval cosine and feature cosine where definitions match.
- Normalize/cache entity tables once per normalization version in a columnar format; partition by split/source and checksum against raw input.
- Query retrieval indexes in bounded S1 batches and use top-N sparse multiplication/ANN methods that do not materialize query-by-corpus matrices.
- Persist final candidates in partitioned long-form columnar files. Stream the required wide comma-list TSV only at output time.
- Vectorize exact/token/numeric features; avoid Python nested loops over all pairs. Use compiled RapidFuzz-style batch functions and parallel partitions for unavoidable string distances.
- Compute cheap features first and benchmark whether expensive Monge-Elkan/cross-encoder features can be restricted to a smaller, deterministic rerank subset without changing `candidate_pairs.tsv` semantics: if a candidate is not scored by the final matching model, it must not be represented as scored.
- Control process/thread oversubscription; log batch peak RSS and estimate total disk before full runs.
- Use model prediction in batches and write scores incrementally; validate partition completeness before aggregation.

Likely bottlenecks, in order, are char n-gram vocabulary/matrices and top-K retrieval over roughly 10M targets; candidate explosion and disk I/O; edit-distance feature computation; LightGBM training rows; Python-set-based full ID validation; and, if enabled, dense embedding/cross-encoder inference. Each module requires a representative-scale benchmark before a full test run. Candidate K/caps are selected from oracle score versus resource curves, not convenience.

## 24. Risk register

| Risk | Impact | Probability | Detection | Mitigation |
|---|---|---:|---|---|
| Blocking misses true links | Irrecoverable recall and oracle-score loss | High | Micro/complete recall and oracle macro F0.5 by fold/country | Union independent channels; error analysis; tune K/caps; add channels only for demonstrated gaps. |
| Over-aggressive normalization | False merges and lost distinctions | Medium | Golden examples; collision-rate report; raw-vs-clean ablation | Multi-view representations; preserve raw; conservative reversible steps. |
| False-positive merges | Severe F0.5 precision loss | High | High-score false-positive review; per-entity precision | Hard negatives, corroborating features, conservative OOF thresholds. |
| Singleton overmatching | True singleton score changes from 1 to 0 | High | Singleton confusion matrix and score contribution | OOF entity model, explicit singleton gate, precision-biased thresholding. |
| Open-set country shift | France degradation and missing outputs | High | Leave-country-out tests; country ablation; test drift reports | Unicode/multilingual views, generic flags, dynamic labels, global fallback. |
| Address ambiguity/partialness | Incorrect rejection or merge | High | Error slices for missing/numeric-conflict/short addresses | Multiple views, missing flags, soft evidence, never hard-code formats. |
| Pair-model overfitting | Inflated CV and poor leaderboard score | Medium | Fold variance, learning curves, cross-fitted estimates | Group/component folds, regularization, feature ablation, no IDs. |
| Candidate explosion | OOM, excessive disk/runtime | High | Candidate distribution and projected bytes/time | Posting caps, per-channel K, dedup, final cap, batches, sparse storage. |
| Threshold overfitting | Unstable private score | Medium | Threshold sensitivity, fold/country dispersion | Cross-fitted selection, simple scheme, lock before test. |
| Leakage | Invalid optimistic validation | Medium | Artifact lineage/fold audit; adversarial leakage tests | Group folds, fold-fit transforms, OOF stacking/mining only. |
| Output-format errors | Submission rejection | Medium | Internal and official validators | Single writer, atomic files, strict schemas, automated post-run validation. |
| License/parameter violation | Disqualification | Low/Medium | License manifest review before artifact use | Approved allow-list; pin revision/checksum; <=8B assertion. |
| Accidental external-data use | Disqualification | Low | Dependency/network/code audit and methodology review | Offline pipeline; supplied data only; prohibit lookup/geocoding components. |
| Candidate/match artifact drift | Matches absent from submitted candidates | Medium | Hash/count/subset assertion | One canonical final candidate artifact feeds model and writer. |
| Memory-heavy ID validation | Validator OOM or skipped integrity check | High | Peak-RSS benchmark; test with full scale | Sorted/chunked join or disk-backed index; internal strict validation. |
| Unseen duplicate/shared targets | Fold leakage or conflicting assignment | Unknown | Phase 1 graph/uniqueness audit | Connected-component grouping; document actual graph properties. |
| Neural model adds no value | Lost time/compute and slower inference | Medium | Strict incremental OOF ablation | Keep optional; stop if gain/cost gate fails. |

## 25. Implementation phases

| Phase | Objective and key tasks | Outputs | Acceptance criteria | Depends on |
|---|---|---|---|---|
| 1. Repository audit + data profiling | Re-run full schema/ID/graph integrity checks; distributions, missingness, lengths/scripts, collisions, label cardinality; benchmark representative reads. | Data manifest, profile report, issue list, initial tests. | All TSVs load; exact schemas/coverage/uniqueness known; no unreported integrity error. | This plan |
| 2. Preprocessing | Implement multi-view name/address/country normalization with tests and cache format. | Versioned normalized tables, unit tests, collision report. | Deterministic/idempotent tests pass; raw preserved; Unicode cases pass. | 1 |
| 3. Baseline candidate generation | Exact clean/core, char-name/address TF-IDF, rare tokens; batching, provenance, union/caps. | Fold-ready retrievers and candidate artifacts. | Completes representative/full fold within resource budget; deterministic unique candidates. | 2 |
| 4. Blocking evaluation | Implement metrics; K/channel/cap experiments; analyze misses. | Blocking leaderboard, selected baseline config. | Recall, candidate distribution, reduction ratio, and oracle F0.5 reported OOF. | 3 |
| 5. Lexical/structural features | Implement versioned name/address/cross/retrieval features and benchmarks. | Feature matrices/manifests/tests. | Finite schema-stable features; no IDs/labels; runtime acceptable. | 4 |
| 6. Baseline LightGBM model | Build group-safe training pairs, negatives, train and diagnose baseline. | Fold models, pair diagnostics, importances. | Every validation candidate scored OOF; reproducible training; no leakage. | 5 |
| 7. Exact CV + macro F0.5 thresholding | Implement metric, complete pipeline folds, cross-fitted threshold search. | OOF predictions and score report. | Exact metric unit tests pass; every train S1 has one OOF decision; threshold stability recorded. | 6 |
| 8. Hard-negative mining | Mine OOF high-score false pairs and retrain with controlled sampling. | Mined-negative artifact and ablation. | Measurable cross-fitted macro-F0.5 improvement or feature rejected. | 7 |
| 9. Singleton optimization | Build OOF meta-features, cross-fitted entity model, joint thresholds. | Singleton model and contribution report. | No stack leakage; improves/stabilizes macro score and singleton errors. | 7/8 |
| 10. Dense embeddings | License-reviewed multilingual embedding features, then optional ANN retrieval. | License manifest, embeddings/index, ablations. | Legal/resource gates pass and incremental OOF gain justifies cost. | 9 |
| 11. Optional cross-encoder | Rerank feasible candidate subset; produce OOF feature. | Model/revision, OOF scores, latency report. | License/size compliant; no leakage; material net improvement. | 9, optionally 10 |
| 12. Final ablation/ensemble selection | Compare only validated components; analyze folds/countries/resources. | Locked `selected.yaml`, artifact bundle. | Best robust configuration chosen before test labels/leaderboard feedback; reproducible. | 9–11 |
| 13. Test inference | Run frozen pipeline in batches and persist canonical candidates/scores/decisions. | Both final TSVs, run manifest/logs. | Exact test S1 coverage; no incomplete partitions; resource budget met. | 12 |
| 14. Submission validation | Run strict internal checks and official validator with ID check. | Validation reports and file hashes. | Zero internal errors; official validator PASS; matches subset candidates. | 13 |
| 15. Documentation/reproducibility packaging | Pin requirements, expand README, fill methodology, assemble/audit zip. | Reproducible submission archive. | Clean-environment documented command regenerates identical outputs; archive structure/license audit pass. | 14 |

Phase 1 includes deeper, scripted integrity/EDA work even though this plan used a preliminary read-only profile. No production pipeline work begins until this document is accepted.

## 26. Acceptance criteria and stage gates

The following are release-blocking:

- **Data gate:** every file has expected UTF-8 TSV schema; source prefixes, ID uniqueness, ground-truth coverage/existence, and graph properties are verified and recorded.
- **Normalization gate:** deterministic, idempotent, Unicode/missing/ambiguous-abbreviation tests pass; raw fields are preserved.
- **Blocking gate:** every validation S1 has a candidate-set record; blocking recall, reduction ratio, candidate quantiles, and oracle macro F0.5 are reported; selected config meets measured resource limits.
- **Leakage gate:** fold/component assignment audit passes; no entity crosses folds; fitted artifacts and hard negatives have traceable fold lineage.
- **Metric gate:** exact macro-F0.5 unit tests, including singletons and multi-links, pass.
- **OOF gate:** every train S1 has exactly one cross-fitted pair/entity decision and no validation candidates were negative-sampled away.
- **Model gate:** promoted features/components show repeatable OOF improvement or a documented resource win without meaningful score loss.
- **Inference gate:** all final matches are subsets of the canonical candidates; all S1 entities have exactly one output row; no invalid/duplicate target IDs exist.
- **Validation gate:** strict internal validator and supplied validator with `--check-ids` pass.
- **Reproduction gate:** a clean documented command using the locked config/artifacts regenerates byte-identical or semantically identical validated outputs.
- **Compliance gate:** no external lookup path exists; all dependencies/models have pinned versions and acceptable licenses; any pretrained model is <=8B parameters.

Do not proceed to a costlier phase when its dependency gate is failing. A failed optional experiment is recorded and removed from the selected pipeline rather than patched into production.

## 27. Recommended development order

### Minimum viable competitive baseline

1. Data contracts, exact metric, entity/component CV, and conservative multi-view normalization.
2. Exact clean/core name + char n-gram name/address TF-IDF + rare-token union blocking.
3. Core lexical, address numeric, country-relation, missingness, and retrieval features.
4. LightGBM pair scorer with candidate hard negatives.
5. OOF entity-level threshold search, deterministic outputs, and both validators.

This baseline is incomplete without end-to-end OOF blocking evaluation and singleton-aware decisions; a good pair classifier alone is not sufficient.

### High-value upgrades

- iterative OOF hard-negative mining;
- entity-level singleton model using OOF pair scores;
- IDF-weighted/rare-token and numeric-conflict feature refinement;
- careful K/channel/cap optimization against oracle macro F0.5;
- calibration only if it improves threshold stability.

### Experimental upgrades

- word TF-IDF union channel;
- MinHash/LSH;
- multilingual embedding similarities and dense ANN retrieval;
- fine-tuned bi-encoder;
- cross-encoder signal;
- model/threshold ensembles.

Add experimental complexity only when a strict OOF ablation shows robust macro-F0.5 gain and the full test-scale resource projection is acceptable. Prefer improving blocking/error slices and thresholding before adding neural models.

## 28. Final architecture decision summary

The recommended final direction is:

> **High-recall, resource-bounded multi-retriever blocking -> engineered lexical/structural/retrieval features -> LightGBM pair scoring -> optional license-compliant multilingual embedding or cross-encoder signals -> OOF-trained entity-level singleton model -> exact macro-F0.5 joint threshold optimization -> deterministic validated zero/one/many-link output.**

The decisive principles are: candidate recall sets the ceiling; false merges and singleton errors deserve explicit control; countries and scripts are open-set; multi-match entities prohibit top-1 logic; OOF artifacts must reproduce the entire pipeline; and optional complexity earns a place only through measured improvement.
