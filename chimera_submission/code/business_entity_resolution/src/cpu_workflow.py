"""Resumable, provisional CPU/GPU workflow over the existing pipeline stages.

Completed artifacts are reused only after input/configuration checks. This
orchestrator never reads test labels and never calls sampled OOF results final.
"""

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .cpu_inference import prepare_test_retrieval, predict_test, verify_inference_output
from .cpu_training import CPUTrainingConfig, train_cpu_baseline
from .phase4_benchmark import _sha256, _stage_config, run as run_phase4
from .phase4_store import open_store
from .pipeline_store import DiskCandidateStore
from .pipeline_provenance import model_code_sha256
from .retrieval import CHANNELS, RetrievalConfig
from .workflow_progress import WorkflowProgress


TRAIN_FILES = (
    "train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
    "train_ground_truth.tsv",
)


def open_training_candidate_store(
    work_dir: Path, top_k: int, max_candidates: int,
    dense_work: Path | None = None, dense_top_k: int = 20,
):
    """Check finished lexical markers and open a read-only candidate view."""
    if not (work_dir / "store.sqlite").is_file():
        raise FileNotFoundError(work_dir / "store.sqlite")
    expected_retrieval = json.loads(json.dumps(asdict(
        RetrievalConfig(top_k=top_k, max_candidates=max_candidates))))
    for channel in CHANNELS:
        marker = work_dir / f"{channel}.complete"
        if not marker.is_file():
            raise ValueError(f"Phase 4 channel is incomplete: {channel}")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        retrieval = payload.get("retrieval_config", {})
        if payload.get("channel") != channel or retrieval != expected_retrieval:
            raise ValueError(f"Phase 4 channel configuration mismatch: {channel}")
    connection = open_store(work_dir / "store.sqlite")
    try:
        return connection, DiskCandidateStore(
            connection, work_dir, top_k, max_candidates,
            dense_work_dir=dense_work, dense_top_k=dense_top_k)
    except BaseException:
        connection.close()
        raise


@dataclass(frozen=True)
class CPUWorkflowConfig:
    train_dir: Path
    test_dir: Path
    work_root: Path
    output_dir: Path
    training: CPUTrainingConfig
    top_k: int = 50
    max_candidates: int = 250
    shard_size: int | None = None
    test_shard_size: int | None = None
    threads: int = 4
    batch_entities: int = 128
    allow_exploratory: bool = False
    official_check_ids: bool = False
    phase4_work: Path | None = None
    phase4_report: Path | None = None

    def __post_init__(self) -> None:
        RetrievalConfig(top_k=self.top_k, max_candidates=self.max_candidates)
        if min(self.threads, self.batch_entities) < 1 or \
           (self.shard_size is not None and self.shard_size < 1) or \
           (self.test_shard_size is not None and self.test_shard_size < 1):
            raise ValueError("workflow resource limits must be positive")
        if not self.allow_exploratory:
            raise ValueError("provisional output requires explicit allow_exploratory=True")
        paths = (
            self.phase4_work or self.work_root / "phase4-work",
            self.phase4_report or self.work_root / "phase4-report",
            self.work_root / "model", self.work_root / "test-work",
            self.output_dir,
        )
        if len({path.resolve() for path in paths}) != len(paths) or \
           self.output_dir.resolve() == self.work_root.resolve():
            raise ValueError("workflow stage/output directories must be distinct")


def _train_hashes(train_dir: Path) -> dict[str, str]:
    return {name: _sha256(train_dir / name) for name in TRAIN_FILES}


def _shard_size(work_dir: Path, requested: int | None) -> int:
    previous = set()
    for channel in ("char_name", "char_address"):
        marker = work_dir / f"{channel}.complete"
        if marker.exists():
            previous.add(json.loads(marker.read_text(encoding="utf-8"))["shard_size"])
    if len(previous) > 1:
        raise ValueError("completed character channels have different shard sizes")
    existing = next(iter(previous), None)
    if existing is not None and (not isinstance(existing, int) or existing < 1):
        raise ValueError("completed character channel has invalid shard size")
    if existing is not None and requested is not None and requested != existing:
        raise ValueError("requested shard size differs from completed Phase 4 channels")
    return existing if existing is not None else (requested or 100_000)


def _check_phase4_report(
    train_dir: Path, report_dir: Path, work_dir: Path,
    retrieval: RetrievalConfig, shard_size: int,
    current_hashes: dict[str, str] | None = None,
) -> dict[str, str]:
    metrics_path = report_dir / "metrics.json"
    markdown = report_dir / "phase4_report.md"
    if not metrics_path.is_file() or not markdown.is_file():
        raise ValueError("Phase 4 report is incomplete; preserve it for review")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = metrics.get("manifest", {})
    hashes = current_hashes if current_hashes is not None else _train_hashes(train_dir)
    expected_retrieval = json.loads(json.dumps(asdict(retrieval)))
    if manifest.get("training_files_sha256") != hashes or \
       manifest.get("retrieval_config") != expected_retrieval or \
       manifest.get("shard_size") != shard_size:
        raise ValueError("Phase 4 report does not match training inputs/configuration")
    work_manifest = work_dir / "training_inputs.json"
    if not work_manifest.is_file() or \
       json.loads(work_manifest.read_text(encoding="utf-8")) != hashes:
        raise ValueError("Phase 4 work store does not match training inputs")
    for channel in CHANNELS:
        marker = work_dir / f"{channel}.complete"
        if not marker.is_file() or \
           json.loads(marker.read_text(encoding="utf-8")) != _stage_config(
               channel, retrieval, shard_size):
            raise ValueError(f"Phase 4 channel marker mismatch: {channel}")
    connection, _ = open_training_candidate_store(
        work_dir, retrieval.top_k, retrieval.max_candidates)
    connection.close()
    return hashes


def _check_model(
    model_dir: Path, config: CPUWorkflowConfig,
    train_hashes: dict[str, str],
) -> dict[str, object]:
    report_path = model_dir / "training_report.json"
    if not report_path.is_file():
        raise ValueError("existing model directory has no completed training report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("training_files_sha256") != train_hashes or \
       report.get("model_code_sha256") != model_code_sha256() or \
       report.get("config") != asdict(config.training) or \
       report.get("retrieval_top_k") != config.top_k or \
       report.get("retrieval_max_candidates") != config.max_candidates or \
       report.get("retrieval_dense_top_k") is not None or \
       tuple(report.get("retrieval_channels", ())) != CHANNELS:
        raise ValueError("existing model does not match requested training run")
    required = ("decision_config.json", "pair_model.txt",
                "feature_extractor.joblib", "threshold_search.json",
                "sampled_sources.tsv", "oof_pairs.tsv.gz")
    if any(not (model_dir / name).is_file() for name in required):
        raise ValueError("existing model artifact is incomplete")
    if report.get("selected_entity_decision") == "meta" and \
       not (model_dir / "meta_model.joblib").is_file():
        raise ValueError("selected meta model artifact is missing")
    if report.get("selected_entity_decision") not in ("deterministic", "meta"):
        raise ValueError("existing model has no supported entity decision")
    return report


@contextmanager
def _workflow_lock(work_root: Path):
    """Prevent two one-command runs from mutating the same stage artifacts."""
    with (work_root / ".pipeline.lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another CPU pipeline run is using this work root") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_locked(config: CPUWorkflowConfig) -> dict[str, object]:
    total_steps = 16 + config.training.folds + int(config.training.evaluate_meta)
    progress = WorkflowProgress(total_steps)
    progress.detail("Starting provisional pipeline; reused stages count as completed")
    phase4_work = config.phase4_work or config.work_root / "phase4-work"
    phase4_report = config.phase4_report or config.work_root / "phase4-report"
    model_dir = config.work_root / "model"
    test_work = config.work_root / "test-work"
    retrieval = RetrievalConfig(top_k=config.top_k,
                                max_candidates=config.max_candidates)
    shard_size = _shard_size(phase4_work, config.shard_size)
    phase4_ready = (phase4_report / "metrics.json").exists()
    if phase4_ready:
        progress.detail("Checking completed training retrieval artifacts")
        train_hashes = _check_phase4_report(
            config.train_dir, phase4_report, phase4_work, retrieval, shard_size)
        progress.reuse("Training store and input validation")
        for channel in CHANNELS:
            progress.reuse(f"Training retrieval: {channel}")
        progress.reuse("Training retrieval benchmark and report")
    else:
        if phase4_report.exists() and any(phase4_report.iterdir()):
            raise ValueError("partial Phase 4 report exists; preserve it for review")
        run_phase4(config.train_dir, phase4_report, phase4_work,
                   retrieval, "all", shard_size, config.threads, progress=progress)
        fresh_hashes = json.loads((phase4_work / "training_inputs.json").read_text(
            encoding="utf-8"))
        train_hashes = _check_phase4_report(
            config.train_dir, phase4_report, phase4_work, retrieval, shard_size,
            current_hashes=fresh_hashes)
    if model_dir.exists():
        progress.detail("Checking completed model artifacts")
        model_report = _check_model(model_dir, config, train_hashes)
        model_reused = True
        for fold in range(config.training.folds):
            progress.reuse(f"OOF LightGBM fold {fold + 1}/{config.training.folds}")
        progress.reuse("OOF audit and entity threshold search")
        if config.training.evaluate_meta:
            progress.reuse("Optional ZERO/ONE/MANY meta-model comparison")
        progress.reuse("Final sampled LightGBM model and training report")
    else:
        connection, store = open_training_candidate_store(
            phase4_work, config.top_k, config.max_candidates)
        try:
            model_report = train_cpu_baseline(
                store, model_dir, config.training,
                training_files_sha256=train_hashes, progress=progress)
        finally:
            connection.close()
        model_reused = False
    test_counts = prepare_test_retrieval(
        config.test_dir, test_work, config.top_k, config.max_candidates,
        shard_size=_shard_size(test_work, config.test_shard_size),
        threads=config.threads, progress=progress)
    progress.start("Test scoring, submission output, and validation")
    if config.output_dir.exists():
        validated = verify_inference_output(model_dir, test_work, config.output_dir)
        output_reused = True
    else:
        validated = predict_test(
            model_dir, test_work, config.output_dir,
            batch_entities=config.batch_entities, allow_exploratory=True,
            progress=progress)
        output_reused = False
    if config.official_check_ids:
        validator = Path(__file__).resolve().parents[4] / "utils" / "validate_submission.py"
        subprocess.run([
            sys.executable, str(validator),
            "--matching", str(config.output_dir / "matching_results.tsv"),
            "--candidate", str(config.output_dir / "candidate_pairs.tsv"),
            "--test-dir", str(config.test_dir), "--check-ids",
        ], check=True)
    progress.finish(reused=output_reused)
    progress.check_complete()
    return {
        "scope": "provisional sampled-training lexical baseline; not Phase 15 locked",
        "phase4_reused": phase4_ready,
        "model_reused": model_reused,
        "output_reused": output_reused,
        "training_backend": model_report["training_backend"],
        "selected_entity_decision": model_report["selected_entity_decision"],
        "training_source_entities": model_report["sampled_source_entities"],
        "test_source_counts": test_counts,
        "validated_output": validated,
        "output_dir": str(config.output_dir),
    }


def run_cpu_workflow(config: CPUWorkflowConfig) -> dict[str, object]:
    """Run or safely reuse Phase 4, OOF model, test retrieval and output stages."""
    config.work_root.mkdir(parents=True, exist_ok=True)
    with _workflow_lock(config.work_root):
        return _run_locked(config)
