"""Tiny OOF-only fixtures for Phase 10 search and robustness reporting."""

import json
import tempfile
import unittest
from pathlib import Path

from src.threshold_policy import FrozenDecisionConfig
from src.threshold_search import (
    OOFEntity, ThresholdGrid, assemble_oof_entities, evaluate_predictions,
    leave_one_country_out, save_search_report, search_thresholds,
    threshold_sensitivity,
)


def fixture():
    return (
        OOFEntity("s1", ("a", "b"), (0.9, 0.85), frozenset({"a", "b"}), 0, "US"),
        OOFEntity("s2", ("a",), (0.2,), frozenset(), 0, "US"),
        OOFEntity("s3", ("a", "b"), (0.8, 0.4), frozenset({"a"}), 1, "IN"),
        OOFEntity("s4", (), (), frozenset({"missing"}), 1, "IN"),
    )


class ThresholdSearchTests(unittest.TestCase):
    def test_assembly_includes_zero_candidate_and_checks_fold(self):
        truth = {"s1": frozenset({"a"}), "s2": frozenset()}
        args = (["s1"], ["a"], [0.7], [0], truth,
                {"s1": 0, "s2": 1}, {"s1": "US", "s2": "IN"})
        entities = assemble_oof_entities(*args)
        self.assertEqual(entities[1].candidate_ids, ())
        with self.assertRaises(ValueError):
            assemble_oof_entities(args[0], args[1], args[2], [1], *args[4:])

    def test_exact_metrics_and_undefined_denominators(self):
        entities = fixture()
        predictions = {"s1": ["a", "b"], "s2": [], "s3": ["a"], "s4": []}
        result = evaluate_predictions(entities, predictions)
        self.assertEqual(result.overall_macro_f05, 0.75)
        self.assertEqual(result.fold_f05, {0: 1.0, 1: 0.5})
        self.assertEqual(result.worst_fold_f05, 0.5)
        self.assertEqual(result.singleton_precision, 0.5)
        self.assertEqual(result.false_positive_singleton_rate, 0)
        self.assertEqual((result.empty_percent, result.top1_percent, result.multi_percent),
                         (50, 25, 25))
        self.assertEqual(result.average_predicted_matches, 0.75)
        no_singletons = evaluate_predictions(entities[:1], {"s1": ["a"]})
        self.assertIsNone(no_singletons.singleton_precision)
        self.assertIsNone(no_singletons.false_positive_singleton_rate)
        with self.assertRaises(ValueError):
            evaluate_predictions(entities, dict(predictions, s1=["other"]))

    def test_search_selects_from_crossfit_and_freezes_full_oof_fit(self):
        entities = fixture()
        grid = ThresholdGrid((0.5, 0.8), (0.3, 0.7), (0.0, 0.1))
        result = search_thresholds(entities, grid, "raw")
        self.assertEqual(set(result.policies), {"A", "B", "C"})
        self.assertIn(result.selected_policy, result.policies)
        for policy in result.policies.values():
            self.assertEqual(set(policy.fold_configs), {0, 1})
            self.assertEqual(policy.crossfit_metrics.entities, 4)
            self.assertEqual(policy.frozen_config.policy, policy.policy)
        self.assertEqual(result.frozen_config, result.policies[result.selected_policy].frozen_config)

    def test_heldout_label_does_not_affect_its_fold_threshold(self):
        entities = fixture()
        grid = ThresholdGrid((0.3, 0.9), (0.2, 0.8), (0.0, 0.4))
        baseline = search_thresholds(entities, grid, "raw")
        changed = list(entities)
        changed[0] = OOFEntity("s1", ("a", "b"), (0.9, 0.85),
                               frozenset(), 0, "US")
        altered = search_thresholds(changed, grid, "raw")
        for policy in ("A", "B", "C"):
            self.assertEqual(baseline.policies[policy].fold_configs[0],
                             altered.policies[policy].fold_configs[0])

    def test_full_gap_policy_is_not_automatically_selected(self):
        entities = (
            OOFEntity("a", ("x",), (0.2,), frozenset(), 0, "US"),
            OOFEntity("b", ("x",), (0.2,), frozenset(), 1, "IN"),
        )
        result = search_thresholds(entities, ThresholdGrid((0.8,), (0.1,), (0.0,)), "raw")
        self.assertEqual(result.selected_policy, "A")
        self.assertGreater(result.policies["A"].crossfit_metrics.overall_macro_f05,
                           result.policies["C"].crossfit_metrics.overall_macro_f05)

    def test_robustness_and_report_serialization(self):
        entities = fixture()
        grid = ThresholdGrid((0.5, 0.8), (0.3, 0.7), (0.0, 0.1))
        result = search_thresholds(entities, grid, "raw")
        sensitivity = threshold_sensitivity(entities, result.frozen_config, 0.05)
        self.assertIn("baseline", sensitivity)
        self.assertIn("pair_threshold_plus", sensitivity)
        country = leave_one_country_out(entities, result.selected_policy, grid, "raw", 2)
        self.assertEqual(set(country), {"IN", "US"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            save_search_report(path, result, sensitivity, country)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["search"]["score_source"],
                             "leakage-safe training OOF pair predictions")
            self.assertEqual(payload["search"]["selected_policy"], result.selected_policy)
            self.assertEqual(payload["search"]["grid"]["pair"], [0.5, 0.8])
            self.assertIn("pair_threshold", payload["threshold_dispersion_across_folds"])
            self.assertEqual(payload["country_shift_macro_summary"]["eligible_countries"], 2)
            with self.assertRaises(FileExistsError):
                save_search_report(path, result, sensitivity, country)

    def test_invalid_grid_and_single_fold(self):
        with self.assertRaises(ValueError):
            ThresholdGrid((), (0.5,), (0.1,))
        with self.assertRaises(ValueError):
            search_thresholds(fixture()[:2], ThresholdGrid((0.5,), (0.5,), (0.1,)), "raw")
        with self.assertRaises(ValueError):
            threshold_sensitivity(fixture(), FrozenDecisionConfig("A", 0.5, None, None, "raw"), 0)


if __name__ == "__main__":
    unittest.main()
