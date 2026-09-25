"""Phase I — frozen-artifact checklist (stdlib only).

Validates that a model directory contains the complete frozen set before
test inference, without loading heavy artifacts.
"""

from __future__ import annotations

from pathlib import Path

REQUIRED_DETERMINISTIC = (
    "training_report.json",
    "decision_config.json",
    "pair_model.txt",
    "feature_extractor.joblib",
    "threshold_search.json",
    "sampled_sources.tsv",
    "oof_pairs.tsv.gz",
)
OPTIONAL_META = ("meta_model.joblib", "meta_comparison.json",)


def check_frozen_dir(model_dir: str | Path, allow_meta: bool = True) -> dict:
    """Return {missing, unexpected_meta, ready} for a model directory."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ValueError(f"model directory not found: {model_dir}")
    missing = [n for n in REQUIRED_DETERMINISTIC if not (model_dir / n).is_file()]
    meta_present = [(model_dir / n).is_file() for n in OPTIONAL_META]
    unexpected_meta = allow_meta is False and any(meta_present)
    return {
        "missing": missing,
        "meta_complete": all(meta_present),
        "meta_partial": any(meta_present) and not all(meta_present),
        "unexpected_meta": unexpected_meta,
        "ready": (not missing) and not unexpected_meta,
    }
