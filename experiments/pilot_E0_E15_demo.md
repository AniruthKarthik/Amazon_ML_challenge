# Pilot experiments — measured (2026-09-25, 12-core / 15GB, /tmp/er_venv)

## Environment
- venv: numpy 2.5.3, scipy 1.18.1, sklearn 1.9.1, lightgbm 4.7.0, sparse-dot-topn 1.2.0
- Full unit suite: 161 tests OK (1 skipped)

## E0 pilot stratification (measured)
- Population: S1=2,206,821 (US 1,323,633 / India 883,188); cardinality 0:123247 1:119157 2:375212 3:530841 4:484115 5:321957 6:164868 7:63968 8:18680 9:4205 10:534 11:37
- 5K pilot sha 735488a0... strata {india|0:112 india|1:108 india|2-3:820 india|4+:962 us|0:167 us|1:162 us|2-3:1233 us|4+:1436} -> /tmp/er_exp/pilot_5k.tsv
- 100K pilot sha 8e4cec11... (strata proportional, see log)

## File store build (Phase D, measured)
- Streaming normalize 2.2M S1 + 10.3M targets + truth: 356.7s, 2.6GB (source1 421M, targets 2.1G, truth 154M)
- No SQLite files created.

## Exact truncation audit (Phase B, measured on 10.3M targets)
- name_clean K=50: truncated_keys=3073 truncated_hits=137567 rate=0.01333
- name_clean K=20: rate=0.03685; K=10: rate=0.05931
- name_core K=50: truncated_keys=5778 truncated_hits=307952 rate=0.02984
- Decision: exact-topK truncation is a real recall leak (1.3-3.0% of exact hits at K=50). Fix (spillover/budget) required before final freeze.

## Retrieval demo (400 pilot S1, 16.4K targets = truth + 15K distractors, K=20/cap=100)
- E0 recall: 1401/1408 = 0.995, 46.6 cand/S1
- Numeric blocker opt-in: recall unchanged on tiny slice, +extra candidates (earlier 200-S1 slice: +398 cands, recall 1.0 both) -> needs full-scale Unique-GT test, not accepted yet.

## Pair model demo (18,625 pairs, pos_rate 0.075, balanced_w 12.3)
- Base (50 rounds, leaves 15): precision@0.5 0.9888 recall@0.5 0.9698 AUC 0.9994
- Weighted (scale_pos_weight=12.3): AUC 0.9989, delta -0.0005 -> proxy_gate(margin 0.005) = False -> correctly REJECTED for full re-opt. Logged via gate_record.
- NOTE: optimistic slice (7.5% positives vs ~1-3% at full cap 250). Full-scale re-measurement required.

## Policy demo (same slice, thresholds 0.5/0.5/0.2, D max_matches=8)
- Policy C macroF05 0.9756 vs Policy D 0.9770 -> D direction supported, confirm at scale.
- Per-cardinality D: 0:{8 ents, 1.0} 1:{8, 0.875} 2-3:{79, 0.978} 4+:{105, 0.982}
- Reliability: max calibration gap 0.1893 -> isotonic evaluation warranted (Phase F E11).

## Features
- v1 dim 40, v2 dim 53. v2 sample verified (containment/acronym/number-conflict/is_s2). Country strip 40->37 keys verified.

## Freeze / parallel
- workers=12, feature_workers=6, 1.73M test S1 sharded 12x144K. No frozen model dir exists yet (check correctly reports missing). No test inference run (requires full OOF + frozen thresholds).

## Accepted / rejected so far
- ACCEPT (measured): pilot harness, file-store build, normalization expansion (unit-green), feature v2 schema, Policy D candidate, reliability/mining-gate/freeze utilities.
- REJECT/DEFER: numeric blocker (no Unique-GT win yet), weighted retrain on this slice (proxy fail), dense/cross-encoder/CatBoost (gates not fired).
- MANDATORY before freeze: full-scale cap/K sweep, exact-truncation fix, 5-fold nested OOF on >=100K S1, fine-grid threshold search, France-proxy (US-train->India-eval) ablation, frozen test inference + official --check-ids.
