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
from src.cpu_training import CPUTrainingConfig, train_cpu_baseline
from src.pair_model import BaselineConfig
from src.phase4_store import open_store
from src.pipeline_store import DiskCandidateStore
from src.retrieval import CHANNELS


def _training_store(work_dir: Path, top_k: int, max_candidates: int,
                    dense_work: Path | None, dense_top_k: int):
    if not (work_dir / "store.sqlite").is_file():
        raise FileNotFoundError(work_dir / "store.sqlite")
    for channel in CHANNELS:
        marker = work_dir / f"{channel}.complete"
        if not marker.exists():
            raise ValueError(f"Phase 4 channel is incomplete: {channel}")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        retrieval = payload.get("retrieval_config", {})
        if payload.get("channel") != channel or retrieval.get("top_k") != top_k or \
           retrieval.get("max_candidates") != max_candidates:
            raise ValueError(f"Phase 4 channel configuration mismatch: {channel}")
    connection = open_store(work_dir / "store.sqlite")
    try:
        return connection, DiskCandidateStore(
            connection, work_dir, top_k, max_candidates,
            dense_work_dir=dense_work, dense_top_k=dense_top_k)
    except BaseException:
        connection.close()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="sampled, leakage-safe CPU OOF baseline")
    train.add_argument("--phase4-work", type=Path, required=True)
    train.add_argument("--model-dir", type=Path, required=True)
    train.add_argument("--train-entities", type=int, required=True,
                       help="explicit S1 sample size; report is not a full-training estimate")
    train.add_argument("--top-k", type=int, default=50)
    train.add_argument("--max-candidates", type=int, default=250)
    train.add_argument("--dense-work", type=Path)
    train.add_argument("--dense-top-k", type=int, default=20)
    train.add_argument("--max-train-pairs", type=int, default=1_000_000)
    train.add_argument("--folds", type=int, default=3)
    train.add_argument("--grid-points", type=int, default=3)
    train.add_argument("--feature-max-features", type=int, default=30_000)
    train.add_argument("--boost-rounds", type=int, default=100)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--evaluate-meta", action="store_true")
    train.add_argument("--max-worst-fold-drop", type=float)
    train.add_argument("--max-fold-std-increase", type=float)
    prepare = commands.add_parser("prepare-test", help="unlabeled test store/retrieval")
    prepare.add_argument("--test-dir", type=Path, required=True)
    prepare.add_argument("--model-dir", type=Path, required=True)
    prepare.add_argument("--test-work", type=Path, required=True)
    prepare.add_argument("--shard-size", type=int, default=100_000)
    prepare.add_argument("--threads", type=int, default=4)
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
        model = BaselineConfig(num_boost_round=args.boost_rounds,
                               early_stopping_rounds=0,
                               num_threads=args.threads, seed=args.seed)
        config = CPUTrainingConfig(
            train_entities=args.train_entities, max_train_pairs=args.max_train_pairs,
            folds=args.folds, grid_points=args.grid_points,
            feature_max_features=args.feature_max_features,
            seed=args.seed, model=model,
            evaluate_meta=args.evaluate_meta,
            max_worst_fold_drop=args.max_worst_fold_drop,
            max_fold_std_increase=args.max_fold_std_increase,
        )
        connection, store = _training_store(
            args.phase4_work, args.top_k, args.max_candidates,
            args.dense_work, args.dense_top_k)
        try:
            result = train_cpu_baseline(store, args.model_dir, config)
        finally:
            connection.close()
    elif args.command == "prepare-test":
        report = json.loads((args.model_dir / "training_report.json").read_text())
        result = prepare_test_retrieval(
            args.test_dir, args.test_work,
            int(report["retrieval_top_k"]),
            int(report["retrieval_max_candidates"]),
            shard_size=args.shard_size, threads=args.threads,
        )
    elif args.command == "predict":
        result = predict_test(
            args.model_dir, args.test_work, args.output_dir,
            batch_entities=args.batch_entities,
            allow_exploratory=args.allow_exploratory,
            dense_test_work_dir=args.dense_test_work,
        )
    else:
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
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
