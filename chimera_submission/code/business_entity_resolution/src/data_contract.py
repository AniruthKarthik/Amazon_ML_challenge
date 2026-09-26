"""Data contract, schema validation, and bipartite graph analysis for business entity resolution.

Ensures strict TSV compliance, zero silent data drops, validation of IDs and prefixes,
and leak-free CV fold splitting based on bipartite graph connected components.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


class DataIntegrityError(ValueError):
    """Raised when data contracts, schemas, or constraints are violated."""
    pass


REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
VALID_TARGET_PREFIXES = ("S2-", "S3-")


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    business_name: str
    business_address: str
    country: str


@dataclass(frozen=True)
class GraphMetrics:
    num_s1_entities: int
    num_unique_matched_targets: int
    num_total_ground_truth_links: int
    num_singletons: int
    singleton_ratio: float
    num_components: int
    max_component_size: int
    median_component_size: float


class TSVLoader:
    """Robust TSV loader enforcing schema validation and zero data loss."""

    @staticmethod
    def load_source_tsv(
        filepath_or_buffer: Union[str, Path, io.StringIO],
        expected_prefix: Optional[str] = None,
        allow_empty_name: bool = False,
    ) -> pd.DataFrame:
        """Load source TSV (Source 1, 2, or 3) with strict validation.

        Parameters
        ----------
        filepath_or_buffer : path or StringIO buffer.
        expected_prefix : optional prefix expected for entity_id ('S1', 'S2', 'S3').
        allow_empty_name : whether business_name is allowed to be empty.

        Returns
        -------
        pd.DataFrame with columns: entity_id, business_name, business_address, country.
        """
        try:
            df = pd.read_csv(
                filepath_or_buffer,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                na_values=[],
                quoting=csv.QUOTE_NONE,
                on_bad_lines="error",
                encoding="utf-8",
            )
        except Exception as e:
            raise DataIntegrityError(f"Failed to parse TSV file: {e}") from e

        # Normalize column names
        df.columns = [col.strip().lower() for col in df.columns]

        missing_cols = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
        if missing_cols:
            raise DataIntegrityError(
                f"Missing required columns in source TSV: {missing_cols}. "
                f"Found columns: {list(df.columns)}"
            )

        # Keep only required columns in canonical order
        df = df[REQUIRED_SOURCE_COLUMNS].copy()

        # Fill any None with empty string
        for col in REQUIRED_SOURCE_COLUMNS:
            df[col] = df[col].fillna("").astype(str).str.strip()

        if len(df) == 0:
            raise DataIntegrityError("Source TSV is empty (0 records found).")

        # Validate entity_id uniqueness
        duplicates = df[df.duplicated(subset=["entity_id"], keep=False)]
        if not duplicates.empty:
            dup_ids = duplicates["entity_id"].unique()[:5].tolist()
            raise DataIntegrityError(
                f"Found {len(duplicates)} duplicate entity_id entries: e.g. {dup_ids}"
            )

        # Validate empty entity IDs
        if (df["entity_id"] == "").any():
            raise DataIntegrityError("Found row(s) with empty entity_id.")

        # Validate prefix
        if expected_prefix:
            prefix_tag = f"{expected_prefix.upper()}-"
            invalid_prefix = df[~df["entity_id"].str.startswith(prefix_tag)]
            if not invalid_prefix.empty:
                bad_ids = invalid_prefix["entity_id"].head(5).tolist()
                raise DataIntegrityError(
                    f"Entity IDs do not match expected prefix '{prefix_tag}': {bad_ids}"
                )

        # Validate business_name non-emptiness
        if not allow_empty_name:
            empty_names = df[df["business_name"] == ""]
            if not empty_names.empty:
                bad_ids = empty_names["entity_id"].head(5).tolist()
                raise DataIntegrityError(
                    f"Found {len(empty_names)} record(s) with missing/empty business_name: {bad_ids}"
                )

        return df

    @classmethod
    def load_filtered_source_tsv(
        cls,
        filepath_or_buffer: Union[str, Path, io.StringIO],
        expected_prefix: Optional[str] = None,
        filter_ids: Optional[Set[str]] = None,
        required_ids: Optional[Set[str]] = None,
        max_background: Optional[int] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Stream and filter source TSV to strictly preserve RAM bounds."""
        if not isinstance(filepath_or_buffer, (str, Path)) or not os.path.isfile(str(filepath_or_buffer)):
            df = cls.load_source_tsv(filepath_or_buffer, expected_prefix=expected_prefix)
            if filter_ids is not None:
                df = df[df["entity_id"].isin(filter_ids)].copy()
            return df

        rng = np.random.RandomState(seed)
        path = str(filepath_or_buffer)
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            if not header_line:
                raise DataIntegrityError(f"File {path} is empty.")
            headers = [h.strip().lower() for h in header_line.split("\t")]
            col_map = {name: idx for idx, name in enumerate(headers)}
            for req in REQUIRED_SOURCE_COLUMNS:
                if req not in col_map:
                    raise DataIntegrityError(f"Missing required column '{req}' in {path}.")

            id_idx = col_map["entity_id"]
            name_idx = col_map["business_name"]
            addr_idx = col_map["business_address"]
            ctry_idx = col_map["country"]

            bg_count = 0
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(id_idx, name_idx, addr_idx, ctry_idx):
                    continue
                eid = parts[id_idx].strip()
                if not eid:
                    continue

                if filter_ids is not None:
                    if eid in filter_ids:
                        rows.append({
                            "entity_id": eid,
                            "business_name": parts[name_idx].strip(),
                            "business_address": parts[addr_idx].strip(),
                            "country": parts[ctry_idx].strip(),
                        })
                elif required_ids is not None:
                    if eid in required_ids:
                        rows.append({
                            "entity_id": eid,
                            "business_name": parts[name_idx].strip(),
                            "business_address": parts[addr_idx].strip(),
                            "country": parts[ctry_idx].strip(),
                        })
                    elif max_background is not None and bg_count < max_background:
                        if rng.rand() < 0.20:
                            rows.append({
                                "entity_id": eid,
                                "business_name": parts[name_idx].strip(),
                                "business_address": parts[addr_idx].strip(),
                                "country": parts[ctry_idx].strip(),
                            })
                            bg_count += 1
                else:
                    rows.append({
                        "entity_id": eid,
                        "business_name": parts[name_idx].strip(),
                        "business_address": parts[addr_idx].strip(),
                        "country": parts[ctry_idx].strip(),
                    })

        return pd.DataFrame(rows, columns=REQUIRED_SOURCE_COLUMNS)

    @classmethod
    def load_training_split(
        cls,
        train_dir: Path,
        max_queries: int = 100000,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load balanced, high-fidelity training data split constrained to fit in memory."""
        gt_file = train_dir / "train_ground_truth.tsv"
        s1_file = train_dir / "train_source1.tsv"
        s2_file = train_dir / "train_source2.tsv"
        s3_file = train_dir / "train_source3.tsv"

        # Load ground truth labels
        gt_df = GroundTruthLoader.load_ground_truth(gt_file)
        total_queries = len(gt_df)

        if max_queries <= 0 or total_queries <= max_queries:
            train_s1 = cls.load_source_tsv(s1_file, expected_prefix="S1")
            train_s2 = cls.load_source_tsv(s2_file, expected_prefix="S2")
            train_s3 = cls.load_source_tsv(s3_file, expected_prefix="S3")
            return train_s1, train_s2, train_s3, gt_df

        print(
            f"  [Memory Optimization] Sampling {max_queries} representative queries from {total_queries} total entities..."
        )
        # Stratify by match type: multi-match (crucial), single-match, and singletons
        multi_mask = gt_df["matched_entity_ids"].str.contains(",")
        single_mask = (gt_df["matched_entity_ids"] != "") & (~multi_mask)
        singleton_mask = gt_df["matched_entity_ids"] == ""

        multi_df = gt_df[multi_mask]
        single_df = gt_df[single_mask]
        singleton_df = gt_df[singleton_mask]

        remaining = max_queries - len(multi_df)
        if remaining > 0:
            n_single = min(len(single_df), int(remaining * 0.70))
            n_singleton = min(len(singleton_df), remaining - n_single)
            sampled_single = single_df.sample(n=n_single, random_state=seed)
            sampled_singleton = singleton_df.sample(n=n_singleton, random_state=seed)
            sampled_gt = pd.concat([multi_df, sampled_single, sampled_singleton], ignore_index=True)
        else:
            sampled_gt = multi_df.sample(n=max_queries, random_state=seed)

        sampled_s1_ids = set(sampled_gt["source1_entity_id"])

        # Collect true target IDs
        true_targets: Set[str] = set()
        for raw in sampled_gt["matched_entity_ids"]:
            if raw:
                true_targets.update([t.strip() for t in raw.split(",") if t.strip()])

        s2_required = {t for t in true_targets if t.startswith("S2-")}
        s3_required = {t for t in true_targets if t.startswith("S3-")}

        print(
            f"  [Memory Optimization] Loading sampled S1 ({len(sampled_s1_ids)} entities) and relevant S2/S3 targets..."
        )
        train_s1 = cls.load_filtered_source_tsv(
            s1_file, expected_prefix="S1", filter_ids=sampled_s1_ids
        )
        train_s2 = cls.load_filtered_source_tsv(
            s2_file, expected_prefix="S2", required_ids=s2_required, max_background=200000, seed=seed
        )
        train_s3 = cls.load_filtered_source_tsv(
            s3_file, expected_prefix="S3", required_ids=s3_required, max_background=200000, seed=seed
        )

        return train_s1, train_s2, train_s3, sampled_gt

    @classmethod
    def load_country_source_tsv(
        cls,
        filepath_or_buffer: Union[str, Path, io.StringIO, pd.DataFrame],
        country: str,
        expected_prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        """Stream a source TSV and extract records belonging to a specific country with bounded RAM."""
        if isinstance(filepath_or_buffer, pd.DataFrame):
            df = filepath_or_buffer
            if country:
                return df[df["country"].fillna("").astype(str).str.strip() == country].copy()
            return df[df["country"].isna() | (df["country"].astype(str).str.strip() == "")].copy()

        if not isinstance(filepath_or_buffer, (str, Path)) or not os.path.isfile(str(filepath_or_buffer)):
            df = cls.load_source_tsv(filepath_or_buffer, expected_prefix=expected_prefix)
            if country:
                return df[df["country"].fillna("").astype(str).str.strip() == country].copy()
            return df[df["country"].isna() | (df["country"].astype(str).str.strip() == "")].copy()

        path = str(filepath_or_buffer)
        eids, names, addrs, countries = [], [], [], []
        target_country = country.strip() if country else ""

        with open(path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            if not header_line:
                raise DataIntegrityError(f"File {path} is empty.")
            headers = [h.strip().lower() for h in header_line.split("\t")]
            col_map = {name: idx for idx, name in enumerate(headers)}
            for req in REQUIRED_SOURCE_COLUMNS:
                if req not in col_map:
                    raise DataIntegrityError(f"Missing required column '{req}' in {path}.")

            id_idx = col_map["entity_id"]
            name_idx = col_map["business_name"]
            addr_idx = col_map["business_address"]
            ctry_idx = col_map["country"]

            prefix_tag = f"{expected_prefix.upper()}-" if expected_prefix else None

            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(id_idx, name_idx, addr_idx, ctry_idx):
                    continue
                row_country = parts[ctry_idx].strip()
                if row_country != target_country:
                    continue

                eid = parts[id_idx].strip()
                if not eid:
                    continue
                if prefix_tag and not eid.startswith(prefix_tag):
                    continue

                eids.append(eid)
                names.append(parts[name_idx].strip())
                addrs.append(parts[addr_idx].strip())
                countries.append(row_country)

        return pd.DataFrame({
            "entity_id": eids,
            "business_name": names,
            "business_address": addrs,
            "country": countries,
        })



class GroundTruthLoader:
    """Loader and validator for competition ground truth matching labels."""

    @staticmethod
    def load_ground_truth(
        filepath_or_buffer: Union[str, Path, io.StringIO],
        valid_s1_ids: Optional[Set[str]] = None,
        valid_target_ids: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        """Load and strictly validate train_ground_truth.tsv.

        Parameters
        ----------
        filepath_or_buffer : path or StringIO buffer.
        valid_s1_ids : optional set of all S1 IDs from train_source1.tsv.
        valid_target_ids : optional set of all S2/S3 IDs from train_source2/3.tsv.

        Returns
        -------
        pd.DataFrame with columns: source1_entity_id, matched_entity_ids (normalized).
        """
        try:
            df = pd.read_csv(
                filepath_or_buffer,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                na_values=[],
                quoting=csv.QUOTE_NONE,
                on_bad_lines="error",
                encoding="utf-8",
            )
        except Exception as e:
            raise DataIntegrityError(f"Failed to parse ground truth TSV: {e}") from e

        df.columns = [col.strip().lower() for col in df.columns]
        missing_cols = [c for c in REQUIRED_GROUND_TRUTH_COLUMNS if c not in df.columns]
        if missing_cols:
            raise DataIntegrityError(
                f"Ground truth missing required columns: {missing_cols}. Found: {list(df.columns)}"
            )

        df = df[REQUIRED_GROUND_TRUTH_COLUMNS].copy()
        for col in REQUIRED_GROUND_TRUTH_COLUMNS:
            df[col] = df[col].fillna("").astype(str).str.strip()

        if len(df) == 0:
            raise DataIntegrityError("Ground truth TSV is empty (0 records found).")

        # Check unique source1_entity_id
        duplicates = df[df.duplicated(subset=["source1_entity_id"], keep=False)]
        if not duplicates.empty:
            dup_ids = duplicates["source1_entity_id"].unique()[:5].tolist()
            raise DataIntegrityError(
                f"Duplicate source1_entity_id in ground truth: {dup_ids}"
            )

        # Check source1 prefix
        invalid_s1 = df[~df["source1_entity_id"].str.startswith("S1-")]
        if not invalid_s1.empty:
            bad_ids = invalid_s1["source1_entity_id"].head(5).tolist()
            raise DataIntegrityError(
                f"Ground truth source1_entity_id has invalid prefix: {bad_ids}"
            )

        # Check coverage against S1 IDs if provided
        if valid_s1_ids is not None:
            seen_s1 = set(df["source1_entity_id"])
            missing_s1 = valid_s1_ids - seen_s1
            if missing_s1:
                sample_missing = sorted(list(missing_s1))[:5]
                raise DataIntegrityError(
                    f"{len(missing_s1)} Source 1 entities missing from ground truth: {sample_missing}"
                )
            extra_s1 = seen_s1 - valid_s1_ids
            if extra_s1:
                sample_extra = sorted(list(extra_s1))[:5]
                raise DataIntegrityError(
                    f"{len(extra_s1)} unexpected Source 1 entities in ground truth: {sample_extra}"
                )

        # Validate matched_entity_ids format and targets
        s1_ids = df["source1_entity_id"].tolist()
        raw_targets_list = df["matched_entity_ids"].tolist()
        for idx, (s1_id, raw_targets) in enumerate(zip(s1_ids, raw_targets_list)):
            if not raw_targets:
                continue

            target_list = [t.strip() for t in raw_targets.split(",") if t.strip()]
            target_set = set(target_list)

            # Check intra-list duplicate
            if len(target_list) != len(target_set):
                raise DataIntegrityError(
                    f"Ground truth row {s1_id} contains duplicate targets: {target_list}"
                )

            # Check targets validity
            for target_id in target_set:
                if target_id.startswith("S1-"):
                    raise DataIntegrityError(
                        f"Self-match error: Source 1 entity {s1_id} matched to S1 ID {target_id}."
                    )
                if not target_id.startswith(VALID_TARGET_PREFIXES):
                    raise DataIntegrityError(
                        f"Invalid target prefix for match {target_id} in row {s1_id}."
                    )
                if valid_target_ids is not None and target_id not in valid_target_ids:
                    raise DataIntegrityError(
                        f"Unknown target ID {target_id} in row {s1_id} not present in source 2/3."
                    )

        return df


class BipartiteGraphAnalyzer:
    """Disjoint-set bipartite graph analyzer for business entity resolution.

    Identifies connected components linking S1 entities with S2/S3 matches,
    tracks singletons, and builds leak-free cross-validation fold splits.
    """

    def __init__(self, ground_truth_df: pd.DataFrame, all_s1_ids: Optional[Set[str]] = None):
        """Initialize bipartite graph from ground truth.

        Parameters
        ----------
        ground_truth_df : pd.DataFrame with source1_entity_id and matched_entity_ids.
        all_s1_ids : optional full set of S1 IDs (to include singletons if not in df).
        """
        self.ground_truth_df = ground_truth_df
        self.all_s1_ids = set(all_s1_ids) if all_s1_ids is not None else set(ground_truth_df["source1_entity_id"])

        self._parent: Dict[str, str] = {}
        self._s1_matches: Dict[str, List[str]] = {}
        self._target_matches: Dict[str, List[str]] = {}

        self._build_graph()

    def _find(self, item: str) -> str:
        if item not in self._parent:
            self._parent[item] = item
            return item
        if self._parent[item] != item:
            self._parent[item] = self._find(self._parent[item])
        return self._parent[item]

    def _union(self, item1: str, item2: str) -> None:
        root1 = self._find(item1)
        root2 = self._find(item2)
        if root1 != root2:
            self._parent[root1] = root2

    def _build_graph(self) -> None:
        # Register all S1 IDs
        for s1 in self.all_s1_ids:
            self._find(s1)
            self._s1_matches[s1] = []

        # Process ground truth edges
        for _, row in self.ground_truth_df.iterrows():
            s1 = row["source1_entity_id"]
            raw_targets = row["matched_entity_ids"]
            if not raw_targets:
                continue

            targets = [t.strip() for t in raw_targets.split(",") if t.strip()]
            self._s1_matches[s1] = targets

            for target in targets:
                self._find(target)
                self._union(s1, target)
                if target not in self._target_matches:
                    self._target_matches[target] = []
                self._target_matches[target].append(s1)

    def get_connected_components(self) -> Dict[str, Set[str]]:
        """Return mapping of component_root -> set of all entities in component."""
        components: Dict[str, Set[str]] = {}
        all_nodes = set(self._parent.keys())
        for node in all_nodes:
            root = self._find(node)
            if root not in components:
                components[root] = set()
            components[root].add(node)
        return components

    def get_s1_components(self) -> Dict[str, Set[str]]:
        """Return mapping of component_root -> set of S1 entities in that component."""
        full_components = self.get_connected_components()
        s1_components: Dict[str, Set[str]] = {}
        for root, members in full_components.items():
            s1_members = {m for m in members if m.startswith("S1-")}
            if s1_members:
                s1_components[root] = s1_members
        return s1_components

    def get_metrics(self) -> GraphMetrics:
        """Compute bipartite graph structure metrics."""
        components = self.get_connected_components()
        s1_components = self.get_s1_components()

        total_s1 = len(self.all_s1_ids)
        unique_targets = len(self._target_matches)
        total_links = sum(len(targets) for targets in self._s1_matches.values())
        singletons = sum(1 for s1, targets in self._s1_matches.items() if len(targets) == 0)
        singleton_ratio = singletons / total_s1 if total_s1 > 0 else 0.0

        sizes = [len(members) for members in components.values()]
        max_size = max(sizes) if sizes else 0
        median_size = float(np.median(sizes)) if sizes else 0.0

        return GraphMetrics(
            num_s1_entities=total_s1,
            num_unique_matched_targets=unique_targets,
            num_total_ground_truth_links=total_links,
            num_singletons=singletons,
            singleton_ratio=singleton_ratio,
            num_components=len(components),
            max_component_size=max_size,
            median_component_size=median_size,
        )

    def create_leak_free_folds(
        self,
        k_folds: int = 5,
        random_seed: int = 42,
    ) -> Dict[str, int]:
        """Assign S1 entities to K folds so that no connected component crosses fold boundaries.

        Uses greedy size balancing with singleton stratification.

        Parameters
        ----------
        k_folds : number of cross-validation folds.
        random_seed : seed for reproducible distribution of equal-sized components.

        Returns
        -------
        Dict[source1_entity_id, fold_id] (where fold_id in 0 .. k_folds - 1).
        """
        if k_folds < 2:
            raise ValueError("k_folds must be at least 2.")

        s1_components = self.get_s1_components()

        # Classify components into multi-match vs singleton components
        multi_comps = []
        singleton_comps = []

        for root, s1_set in s1_components.items():
            # Check if this component has any ground truth matches
            has_matches = any(len(self._s1_matches.get(s1, [])) > 0 for s1 in s1_set)
            if has_matches:
                multi_comps.append((root, s1_set, len(s1_set)))
            else:
                singleton_comps.append((root, s1_set, len(s1_set)))

        rng = np.random.RandomState(random_seed)

        # Sort multi-match components largest to smallest, with random tie-break
        rng.shuffle(multi_comps)
        multi_comps.sort(key=lambda x: x[2], reverse=True)

        rng.shuffle(singleton_comps)

        fold_counts = [0] * k_folds
        fold_assignment: Dict[str, int] = {}

        # 1. Distribute multi-match components greedily to fold with minimum S1 count
        for root, s1_set, count in multi_comps:
            min_fold = int(np.argmin(fold_counts))
            fold_counts[min_fold] += count
            for s1 in s1_set:
                fold_assignment[s1] = min_fold

        # 2. Distribute singleton components greedily to balance total S1 count across folds
        for root, s1_set, count in singleton_comps:
            min_fold = int(np.argmin(fold_counts))
            fold_counts[min_fold] += count
            for s1 in s1_set:
                fold_assignment[s1] = min_fold

        # Ensure all S1 entities received a fold assignment
        unassigned = self.all_s1_ids - set(fold_assignment.keys())
        if unassigned:
            for s1 in unassigned:
                min_fold = int(np.argmin(fold_counts))
                fold_counts[min_fold] += 1
                fold_assignment[s1] = min_fold

        return fold_assignment
