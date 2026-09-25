"""Phase E — stdlib-only text-similarity helpers for feature v2.

No numpy/sklearn here so unit tests run in minimal environments.
pair_features.py imports from this module.
"""

from __future__ import annotations


V2_FEATURE_NAMES = (
    "name_containment", "name_sorted_exact",
    "name_first_token_match", "name_last_token_match",
    "name_acronym_match", "name_len_ratio",
    "name_token_count_diff", "address_len_ratio",
    "address_number_conflict", "address_first_number_match",
    "is_s2_target",
    "strong_name_weak_address", "exact_name_number_conflict",
)
COUNTRY_FEATURES = ("country_match", "country_mismatch", "country_missing")


def _containment(left: str, right: str) -> float:
    """Max token-containment either direction (handles DBA/prefix names)."""
    first, second = set(left.split()), set(right.split())
    if not first or not second:
        return 0.0
    return max(len(first & second) / len(first), len(first & second) / len(second))


def _acronym_match(left: str, right: str) -> float:
    """Initialism agreement (IBM vs International Business Machines)."""
    left_tokens = [t for t in left.split() if t]
    right_tokens = [t for t in right.split() if t]
    if not left_tokens or not right_tokens:
        return 0.0
    left_init = "".join(t[0] for t in left_tokens)
    right_init = "".join(t[0] for t in right_tokens)
    if not left_init or not right_init:
        return 0.0
    if len(left_tokens) == 1 and left_tokens[0] == right_init:
        return 1.0
    if len(right_tokens) == 1 and right_tokens[0] == left_init:
        return 1.0
    return float(left_init == right_init)


def _length_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return min(len(left), len(right)) / max(len(left), len(right))


def strip_country_features(row: dict) -> dict:
    """Phase E ablation: remove country flags for France-robustness testing."""
    return {k: v for k, v in row.items() if k not in COUNTRY_FEATURES}
