"""Stable fingerprint of code that determines retrieval, scores and decisions."""

from __future__ import annotations

import hashlib
from pathlib import Path


MODEL_CODE_FILES = (
    "data_contract.py", "normalization.py", "retrieval.py",
    "phase4_store.py", "phase4_char.py", "pipeline_store.py",
    "pair_features.py", "pair_model.py", "threshold_policy.py",
    "threshold_search.py", "entity_meta_model.py", "cpu_training.py",
    "cpu_inference.py", "dense_retrieval.py",
)


def model_code_sha256() -> str:
    """Hash relevant source bytes, independent of repository branch names."""
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in MODEL_CODE_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with (directory / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
