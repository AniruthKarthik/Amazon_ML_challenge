"""Deterministic, Unicode-aware views of business records."""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from .data_contract import BusinessRecord


# Start with the legal and address abbreviations explicitly described in the
# competition problem statement. Keep the lists local and auditable.
# Phase C expansion: additional unambiguous legal suffixes (US/India/France)
# and street aliases. Deliberately EXCLUDES "co" (ambiguous: company/county):
# the "café and co" core form must be preserved per unit tests.
LEGAL_SUFFIXES = frozenset({
    "corp", "corporation", "ltd", "limited",
    # Phase C additions (single trailing tokens, unambiguous legal entities).
    "inc", "incorporated", "company", "llc", "llp", "plc",
    "gmbh", "sarl", "sas", "eurl", "srl", "pte",
    "pvt", "private",
})
LEGAL_SUFFIX_PAIRS = frozenset({
    ("pvt", "ltd"), ("private", "limited"),
    # Phase C additions.
    ("pvt", "limited"), ("private", "ltd"),
})
ADDRESS_ALIASES = {
    "rd": "road", "st": "street",
    # Phase C additions (US/India common street aliases; values are never keys,
    # preserving alias idempotence).
    "ave": "avenue", "blvd": "boulevard", "ln": "lane", "dr": "drive",
    "ct": "court", "pl": "place",
}


@dataclass(frozen=True)
class NormalizedRecord:
    raw: BusinessRecord
    business_name_clean: str
    business_name_folded: str
    business_name_core: str
    business_address_clean: str
    business_address_alias: str


def clean_text(value: str) -> str:
    """NFKC/casefold text, retaining Unicode letters, numbers, and marks."""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("&", " and ")
    chars = (
        " " if unicodedata.category(char)[0] in {"P", "S", "Z", "C"} else char
        for char in value
    )
    return " ".join("".join(chars).split())


def fold_accents(value: str) -> str:
    """Remove Latin accents while retaining non-Latin combining marks."""
    decomposed = unicodedata.normalize("NFKD", value)
    result: list[str] = []
    latin_base = False
    for char in decomposed:
        if unicodedata.combining(char):
            if not latin_base:
                result.append(char)
            continue
        result.append(char)
        latin_base = unicodedata.name(char, "").startswith("LATIN")
    return unicodedata.normalize("NFKC", "".join(result))


def core_name(clean_name: str) -> str:
    """Remove documented legal suffixes from the end of a clean name."""
    words = clean_name.split()
    while len(words) > 1:
        if len(words) > 2 and tuple(words[-2:]) in LEGAL_SUFFIX_PAIRS:
            del words[-2:]
        elif words[-1] in LEGAL_SUFFIXES:
            words.pop()
        else:
            break
    return " ".join(words)


def alias_address(clean_address: str) -> str:
    """Expand documented street abbreviations as a separate address view."""
    return " ".join(ADDRESS_ALIASES.get(word, word) for word in clean_address.split())


def normalize_record(record: BusinessRecord) -> NormalizedRecord:
    name_clean = clean_text(record.business_name)
    address_clean = clean_text(record.business_address)
    return NormalizedRecord(
        raw=record,
        business_name_clean=name_clean,
        business_name_folded=fold_accents(name_clean),
        business_name_core=core_name(name_clean),
        business_address_clean=address_clean,
        business_address_alias=alias_address(address_clean),
    )


def normalize_source(records: Mapping[str, BusinessRecord]) -> dict[str, NormalizedRecord]:
    """Normalize a source without changing its IDs, order, or raw records."""
    return {identifier: normalize_record(record) for identifier, record in records.items()}


def collision_rates(records: Iterable[NormalizedRecord]) -> dict[str, float]:
    """Fraction of nonempty records sharing each normalized value with another."""
    views = (
        "business_name_clean", "business_name_folded", "business_name_core",
        "business_address_clean", "business_address_alias",
    )
    counts_by_view = {view: Counter() for view in views}
    for record in records:
        for view in views:
            value = getattr(record, view)
            if value:
                counts_by_view[view][value] += 1
    rates = {}
    for view in views:
        counts = counts_by_view[view]
        total = sum(counts.values())
        rates[view] = sum(count for count in counts.values() if count > 1) / total if total else 0.0
    return rates
