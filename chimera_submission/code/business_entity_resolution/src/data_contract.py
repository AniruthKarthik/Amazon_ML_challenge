"""Strict, lossless input validation for the entity-resolution pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


class DataContractError(ValueError):
    """An input TSV violates the competition data contract."""


@dataclass(frozen=True)
class BusinessRecord:
    entity_id: str
    business_name: str
    business_address: str
    country: str


def _read_tsv(path: Path, columns: tuple[str, ...]) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        try:
            try:
                header = next(reader)
            except StopIteration as exc:
                raise DataContractError(f"{path}: missing header") from exc
            if len(header) != len(columns) or set(header) != set(columns):
                raise DataContractError(
                    f"{path}: expected columns {columns}, received {tuple(header)}"
                )
            for row in reader:
                if len(row) != len(columns):
                    raise DataContractError(
                        f"{path}:{reader.line_num}: expected {len(columns)} fields, "
                        f"received {len(row)}"
                    )
                yield reader.line_num, dict(zip(header, row))
        except csv.Error as exc:
            raise DataContractError(f"{path}:{reader.line_num}: invalid TSV: {exc}") from exc


def _valid_id(identifier: str, source: int) -> bool:
    prefix = f"S{source}-"
    return (
        identifier.startswith(prefix)
        and len(identifier) > len(prefix)
        and not any(char.isspace() or char == "," for char in identifier)
    )


def load_source(path: str | Path, source: int) -> dict[str, BusinessRecord]:
    """Load one source TSV, preserving every row and every raw field value."""
    if source not in (1, 2, 3):
        raise ValueError("source must be 1, 2, or 3")
    path = Path(path)
    records: dict[str, BusinessRecord] = {}
    for line, row in _read_tsv(path, SOURCE_COLUMNS):
        identifier = row["entity_id"]
        if not _valid_id(identifier, source):
            raise DataContractError(f"{path}:{line}: invalid S{source} entity_id {identifier!r}")
        if not row["business_name"].strip():
            raise DataContractError(f"{path}:{line}: business_name is missing")
        if identifier in records:
            raise DataContractError(f"{path}:{line}: duplicate entity_id {identifier!r}")
        records[identifier] = BusinessRecord(**row)
    return records


def load_ground_truth(
    path: str | Path,
    source1: dict[str, BusinessRecord],
    source2: dict[str, BusinessRecord],
    source3: dict[str, BusinessRecord],
) -> dict[str, frozenset[str]]:
    """Validate complete S1 coverage and unique, existing S2/S3 targets."""
    path = Path(path)
    truth: dict[str, frozenset[str]] = {}
    target_owner: dict[str, str] = {}
    valid_targets = source2.keys() | source3.keys()
    for line, row in _read_tsv(path, TRUTH_COLUMNS):
        source_id = row["source1_entity_id"]
        if source_id not in source1:
            raise DataContractError(f"{path}:{line}: unknown S1 entity_id {source_id!r}")
        if source_id in truth:
            raise DataContractError(f"{path}:{line}: duplicate S1 row {source_id!r}")
        raw_targets = row["matched_entity_ids"]
        target_ids = [] if raw_targets == "" else raw_targets.split(",")
        for target_id in target_ids:
            if target_id not in valid_targets:
                raise DataContractError(f"{path}:{line}: unknown S2/S3 target {target_id!r}")
            if target_id in target_owner:
                raise DataContractError(
                    f"{path}:{line}: target {target_id!r} already linked to "
                    f"{target_owner[target_id]!r}"
                )
            target_owner[target_id] = source_id
        truth[source_id] = frozenset(target_ids)
    missing = source1.keys() - truth.keys()
    if missing:
        raise DataContractError(
            f"{path}: missing ground truth for {len(missing)} S1 entities; "
            f"first: {min(missing)!r}"
        )
    return truth


def connected_components(
    source1: dict[str, BusinessRecord],
    source2: dict[str, BusinessRecord],
    source3: dict[str, BusinessRecord],
    truth: dict[str, frozenset[str]],
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic components, including all isolated records."""
    identifiers = source1.keys() | source2.keys() | source3.keys()
    parent = {identifier: identifier for identifier in identifiers}

    def find(identifier: str) -> str:
        root = identifier
        while parent[root] != root:
            root = parent[root]
        while parent[identifier] != identifier:
            identifier, parent[identifier] = parent[identifier], root
        return root

    for source_id, targets in truth.items():
        for target_id in targets:
            parent[find(target_id)] = find(source_id)

    groups: dict[str, list[str]] = {}
    for identifier in sorted(identifiers):
        groups.setdefault(find(identifier), []).append(identifier)
    return tuple(sorted((tuple(group) for group in groups.values()), key=lambda g: g[0]))


def summarize_training(
    source1: dict[str, BusinessRecord],
    source2: dict[str, BusinessRecord],
    source3: dict[str, BusinessRecord],
    truth: dict[str, frozenset[str]],
) -> dict[str, object]:
    """Report Phase 1 record counts, S1 singleton ratio, and graph sizes."""
    components = connected_components(source1, source2, source3, truth)
    return {
        "source_counts": {"S1": len(source1), "S2": len(source2), "S3": len(source3)},
        "ground_truth_links": sum(map(len, truth.values())),
        "s1_singleton_ratio": (
            sum(not targets for targets in truth.values()) / len(source1)
            if source1 else 0.0
        ),
        "component_count": len(components),
        "component_sizes": sorted((len(component) for component in components), reverse=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate training data and report graph metrics")
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir
    source1 = load_source(data_dir / "train_source1.tsv", 1)
    source2 = load_source(data_dir / "train_source2.tsv", 2)
    source3 = load_source(data_dir / "train_source3.tsv", 3)
    truth = load_ground_truth(
        data_dir / "train_ground_truth.tsv", source1, source2, source3
    )
    print(json.dumps(summarize_training(source1, source2, source3, truth), sort_keys=True))


if __name__ == "__main__":
    main()
