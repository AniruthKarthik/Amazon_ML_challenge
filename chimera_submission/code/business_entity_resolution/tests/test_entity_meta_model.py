"""Synthetic Phase 12 set-rule, fold-isolation and CPU model checks."""

import json
import tempfile
import unittest
from pathlib import Path

from src.entity_meta_model import (
    MetaModelConfig, compare_meta_to_phase10, entity_meta_features,
    generate_meta_oof_decisions, load_meta_artifact,
    meta_feature_vector, predict_frozen_meta, predict_meta_match_set, save_meta_artifact,
    save_meta_report,
)
from src.threshold_policy import FrozenDecisionConfig
from src.threshold_search import OOFEntity


def fixture():
    rows = []
    for fold in range(3):
        rows.extend((
            OOFEntity(f"zero-{fold}", ("a", "b"), (0.15, 0.1),
                      frozenset(), fold, "US"),
            OOFEntity(f"one-{fold}", ("a", "b"), (0.9, 0.2),
                      frozenset({"a"}), fold, "IN"),
            OOFEntity(f"many-{fold}", ("a", "b"), (0.85, 0.82),
                      frozenset({"a", "b"}), fold, "GB"),
        ))
    return tuple(rows)


class MetaDecisionTests(unittest.TestCase):
    def test_all_rule_branches_and_no_forced_second_match(self):
        ids = ["b", "a", "c"]
        scores = [0.85, 0.9, 0.2]
        self.assertEqual(predict_meta_match_set("ZERO", ids, scores, 0.8), [])
        self.assertEqual(predict_meta_match_set("ONE", ids, scores, 0.8), ["a"])
        self.assertEqual(predict_meta_match_set("MANY", ids, scores, 0.85),
                         ["a", "b"])
        self.assertEqual(predict_meta_match_set("MANY", ids, scores, 0.87),
                         ["a"])
        self.assertEqual(predict_meta_match_set("MANY", ids, scores, 0.95),
                         ["a"])
        self.assertEqual(predict_meta_match_set("MANY", [], [], 0.8), [])
        self.assertEqual(predict_meta_match_set("ONE", [], [], 0.8), [])

    def test_dedup_ties_and_invalid_inputs(self):
        self.assertEqual(predict_meta_match_set("MANY", ["b", "a", "b"],
                                                [0.9, 0.9, 0.1], 0.9),
                         ["a", "b"])
        with self.assertRaises(ValueError):
            predict_meta_match_set("TWO", ["a"], [0.5], 0.5)
        with self.assertRaises(ValueError):
            predict_meta_match_set("ONE", ["a"], [0.5], -0.1)

    def test_feature_vector_is_score_only_and_finite(self):
        values = entity_meta_features(fixture()[0])
        self.assertEqual(len(values), 7)
        self.assertEqual(values[0], 2)
        empty = OOFEntity("empty", (), (), frozenset({"a"}), 0, "US")
        self.assertEqual(entity_meta_features(empty).tolist(), [0] * 7)
        self.assertEqual(meta_feature_vector("x", ["a", "a"], [0.1, 0.9])[0], 1)


class MetaOOFTests(unittest.TestCase):
    def test_oof_training_thresholds_metrics_and_artifact_roundtrip(self):
        entities = fixture()
        result = generate_meta_oof_decisions(
            entities, (0.5, 0.8, 0.9), MetaModelConfig(max_iter=100),
        )
        self.assertEqual(set(result.fold_thresholds), {0, 1, 2})
        self.assertEqual(set(result.predictions), {entity.source_id for entity in entities})
        self.assertEqual(result.crossfit_metrics.entities, len(entities))
        self.assertIn(result.frozen_multi_pair_threshold, (0.5, 0.8, 0.9))
        for entity in entities:
            self.assertTrue(set(result.predictions[entity.source_id]) <= set(entity.candidate_ids))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.joblib"
            save_meta_artifact(path, result)
            loaded_model, threshold, loaded_config = load_meta_artifact(path)
            self.assertEqual(threshold, result.frozen_multi_pair_threshold)
            self.assertEqual(loaded_config, result.model_config)
            self.assertEqual(predict_frozen_meta(
                loaded_model, entities[1].candidate_ids, entities[1].scores, threshold),
                predict_frozen_meta(result.final_model, entities[1].candidate_ids,
                                    entities[1].scores, threshold))
            with self.assertRaises(FileExistsError):
                save_meta_artifact(path, result)

    def test_heldout_labels_do_not_change_fold_class_or_threshold(self):
        entities = fixture()
        first = generate_meta_oof_decisions(entities, (0.5, 0.8))
        changed = list(entities)
        changed[0] = OOFEntity("zero-0", ("a", "b"), (0.15, 0.1),
                               frozenset({"a", "b"}), 0, "US")
        second = generate_meta_oof_decisions(changed, (0.5, 0.8))
        self.assertEqual(first.fold_thresholds[0], second.fold_thresholds[0])
        for entity in entities[:3]:
            self.assertEqual(first.predicted_classes[entity.source_id],
                             second.predicted_classes[entity.source_id])
            self.assertEqual(first.predictions[entity.source_id],
                             second.predictions[entity.source_id])

    def test_same_fold_phase10_comparison_and_explicit_robustness_limits(self):
        entities = fixture()
        result = generate_meta_oof_decisions(entities, (0.5, 0.8))
        configs = {fold: FrozenDecisionConfig("A", 0.8, None, None, "raw")
                   for fold in range(3)}
        comparison = compare_meta_to_phase10(entities, result, configs, 0, 0)
        self.assertEqual(comparison.meta, result.crossfit_metrics)
        self.assertEqual(comparison.deterministic.entities, len(entities))
        self.assertEqual(comparison.keep_meta,
                         comparison.macro_delta > 0 and comparison.worst_fold_delta >= 0
                         and comparison.fold_std_delta <= 0)
        with self.assertRaises(ValueError):
            compare_meta_to_phase10(entities, result, {0: configs[0]}, 0, 0)
        with self.assertRaises(ValueError):
            compare_meta_to_phase10(entities, result, configs, -0.1, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta-report.json"
            save_meta_report(path, result, comparison)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["score_source"], "training OOF pair scores only")
            self.assertEqual(payload["meta_crossfit_metrics"]["entities"], len(entities))
            with self.assertRaises(FileExistsError):
                save_meta_report(path, result, comparison)

    def test_invalid_grid_and_fold_count(self):
        with self.assertRaises(ValueError):
            generate_meta_oof_decisions(fixture(), ())
        with self.assertRaises(ValueError):
            generate_meta_oof_decisions(fixture()[:3], (0.5,))


if __name__ == "__main__":
    unittest.main()
