"""Staged CPU pipeline CLI. Full project-data computation runs on the user's laptop."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cpu_inference import prepare_test_retrieval, predict_test, validate_outputs
from src.cpu_resources import available_cpu_count, parse_thread_count
from src.cpu_training import CPUTrainingConfig, train_cpu_baseline
from src.cpu_workflow import (
    CPUWorkflowConfig, open_training_candidate_store, run_cpu_workflow,
)
from src.pair_model import BaselineConfig
from src.phase4_store import open_store
from src.workflow_progress import WorkflowProgress


def _add_training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-entities", type=int, required=True,
                        help="explicit S1 sample size; not a full-training estimate")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=250)
    parser.add_argument("--max-train-pairs", type=int, default=1_000_000)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--grid-points", type=int, default=3)
    parser.add_argument("--feature-max-features", type=int, default=30_000)
    parser.add_argument("--country-min-entities", type=int, default=100)
    parser.add_argument("--boost-rounds", type=int, default=100)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--threads", type=parse_thread_count,
                        default=available_cpu_count(),
                        help="parallel retrieval/model workers; auto uses all available CPU cores (default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-meta", action="store_true")
    parser.add_argument("--max-worst-fold-drop", type=float)
    parser.add_argument("--max-fold-std-increase", type=float)


def _training_config(args: argparse.Namespace) -> CPUTrainingConfig:
    model = BaselineConfig(num_boost_round=args.boost_rounds,
                           early_stopping_rounds=0,
                           min_data_in_leaf=args.min_data_in_leaf,
                           num_threads=args.threads, seed=args.seed)
    return CPUTrainingConfig(
        train_entities=args.train_entities, max_train_pairs=args.max_train_pairs,
        folds=args.folds, grid_points=args.grid_points,
        feature_max_features=args.feature_max_features,
        country_min_entities=args.country_min_entities,
        seed=args.seed, model=model,
        evaluate_meta=args.evaluate_meta,
        max_worst_fold_drop=args.max_worst_fold_drop,
        max_fold_std_increase=args.max_fold_std_increase,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="sampled, leakage-safe CPU OOF baseline")
    train.add_argument("--phase4-work", type=Path, required=True)
    train.add_argument("--model-dir", type=Path, required=True)
    _add_training_options(train)
    train.add_argument("--dense-work", type=Path)
    train.add_argument("--dense-top-k", type=int, default=20)
    workflow = commands.add_parser("run", help="resumable provisional lexical CPU workflow")
    workflow.add_argument("--train-dir", type=Path, required=True)
    workflow.add_argument("--test-dir", type=Path, required=True)
    workflow.add_argument("--work-root", type=Path, required=True)
    workflow.add_argument("--output-dir", type=Path, required=True)
    workflow.add_argument("--phase4-work", type=Path)
    workflow.add_argument("--phase4-report", type=Path)
    workflow.add_argument("--shard-size", type=int)
    workflow.add_argument("--test-shard-size", type=int)
    workflow.add_argument("--batch-entities", type=int, default=128)
    workflow.add_argument("--allow-provisional", "--allow-exploratory",
                          dest="allow_exploratory", action="store_true")
    workflow.add_argument("--official-check-ids", action="store_true")
    _add_training_options(workflow)
    prepare = commands.add_parser("prepare-test", help="unlabeled test store/retrieval")
    prepare.add_argument("--test-dir", type=Path, required=True)
    prepare.add_argument("--model-dir", type=Path, required=True)
    prepare.add_argument("--test-work", type=Path, required=True)
    prepare.add_argument("--shard-size", type=int, default=100_000)
    prepare.add_argument("--threads", type=parse_thread_count,
                         default=available_cpu_count(),
                         help="parallel retrieval workers; auto uses all available CPU cores (default)")
    predict = commands.add_parser("predict", help="frozen model/policy TSV output")
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--test-work", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--batch-entities", type=int, default=128)
    predict.add_argument("--dense-test-work", type=Path)
    predict.add_argument("--allow-exploratory", action="store_true",
                         help="acknowledge sampled model is not Phase 15 locked")
    validate = commands.add_parser("validate", help="streaming internal output validation")
    validate.add_argument("--test-work", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--test-dir", type=Path)
    validate.add_argument("--official-check-ids", action="store_true",
                          help="also run official memory-heavy --check-ids validator")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "train":
        progress = WorkflowProgress(args.folds + 2 + int(args.evaluate_meta))
        config = _training_config(args)
        connection, store = open_training_candidate_store(
            args.phase4_work, args.top_k, args.max_candidates,
            args.dense_work, args.dense_top_k)
        try:
            manifest = args.phase4_work / "training_inputs.json"
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            result = train_cpu_baseline(
                store, args.model_dir, config,
                training_files_sha256=json.loads(manifest.read_text(encoding="utf-8")),
                progress=progress,
            )
        finally:
            connection.close()
        progress.check_complete()
    elif args.command == "run":
        result = run_cpu_workflow(CPUWorkflowConfig(
            train_dir=args.train_dir, test_dir=args.test_dir,
            work_root=args.work_root, output_dir=args.output_dir,
            training=_training_config(args), top_k=args.top_k,
            max_candidates=args.max_candidates,
            shard_size=args.shard_size,
            test_shard_size=args.test_shard_size, threads=args.threads,
            batch_entities=args.batch_entities,
            allow_exploratory=args.allow_exploratory,
            official_check_ids=args.official_check_ids,
            phase4_work=args.phase4_work, phase4_report=args.phase4_report,
        ))
    elif args.command == "prepare-test":
        progress = WorkflowProgress(6)
        report = json.loads((args.model_dir / "training_report.json").read_text())
        result = prepare_test_retrieval(
            args.test_dir, args.test_work,
            int(report["retrieval_top_k"]),
            int(report["retrieval_max_candidates"]),
            shard_size=args.shard_size, threads=args.threads, progress=progress,
        )
        progress.check_complete()
    elif args.command == "predict":
        progress = WorkflowProgress(1)
        progress.start("Test scoring, submission output, and validation")
        result = predict_test(
            args.model_dir, args.test_work, args.output_dir,
            batch_entities=args.batch_entities,
            allow_exploratory=args.allow_exploratory,
            dense_test_work_dir=args.dense_test_work,
            progress=progress,
        )
        progress.finish()
        progress.check_complete()
    else:
        progress = WorkflowProgress(1)
        progress.start("Validate existing submission output")
        if not (args.test_work / "store.sqlite").is_file():
            raise FileNotFoundError(args.test_work / "store.sqlite")
        connection = open_store(args.test_work / "store.sqlite")
        try:
            result = validate_outputs(
                connection, args.output_dir / "matching_results.tsv",
                args.output_dir / "candidate_pairs.tsv",
            )
        finally:
            connection.close()
        if args.official_check_ids:
            if args.test_dir is None:
                raise ValueError("--test-dir is required with --official-check-ids")
            validator = Path(__file__).resolve().parents[4] / "utils" / "validate_submission.py"
            subprocess.run([
                sys.executable, str(validator),
                "--matching", str(args.output_dir / "matching_results.tsv"),
                "--candidate", str(args.output_dir / "candidate_pairs.tsv"),
                "--test-dir", str(args.test_dir), "--check-ids",
            ], check=True)
        progress.finish()
        progress.check_complete()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
