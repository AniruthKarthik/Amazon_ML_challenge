"""Deterministic heuristic taxonomy for genuine Phase 4 retrieval misses."""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher

from .normalization import alias_address, clean_text, core_name, fold_accents


ABBREVIATIONS = {
    "corp": "corporation", "pvt": "private", "ltd": "limited",
    "rd": "road", "st": "street",
}
SCRIPTS = (
    "LATIN", "DEVANAGARI", "BENGALI", "KANNADA", "TAMIL", "TELUGU",
    "GUJARATI", "GURMUKHI", "ARABIC", "CYRILLIC", "GREEK",
)


def _scripts(value: str) -> set[str]:
    present = set()
    for char in value:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        present.update(script for script in SCRIPTS if script in name)
    return present


def classify_retrieval_miss(source_name: str, source_address: str,
                            target_name: str, target_address: str) -> dict[str, object]:
    """Assign a primary rule and auditable indicators; labels are not model errors."""
    name1, name2 = clean_text(source_name), clean_text(target_name)
    address1 = alias_address(clean_text(source_address))
    address2 = alias_address(clean_text(target_address))
    core1, core2 = core_name(name1), core_name(name2)
    name_similarity = SequenceMatcher(None, name1, name2, autojunk=False).ratio()
    address_similarity = (
        SequenceMatcher(None, address1, address2, autojunk=False).ratio()
        if address1 and address2 else None
    )
    indicators = []
    if name1 == name2 and name1:
        indicators.append("exact_name_channel_cap_or_tie")
    if core1 == core2 and core1:
        indicators.append("exact_core_channel_cap_or_tie")
    if name1 != name2 and fold_accents(name1) == fold_accents(name2):
        indicators.append("normalization_failure")
    if core1 != core2 and sorted(core1.split()) == sorted(core2.split()):
        indicators.append("token_reordering")
    expanded1 = [ABBREVIATIONS.get(token, token) for token in name1.split()]
    expanded2 = [ABBREVIATIONS.get(token, token) for token in name2.split()]
    if name1 != name2 and expanded1 == expanded2:
        indicators.append("abbreviation")
    scripts1, scripts2 = _scripts(source_name), _scripts(target_name)
    if scripts1 and scripts2 and scripts1.isdisjoint(scripts2):
        indicators.append("transliteration_or_script_change")
    if not address1 or not address2:
        indicators.append("missing_fields")
    elif address_similarity is not None and address_similarity < 0.45:
        indicators.append("address_corruption_or_variation")
    if name1 != name2 and name_similarity >= 0.65:
        indicators.append("spelling_corruption_or_variation")
    if name_similarity < 0.65 and not scripts1.isdisjoint(scripts2):
        indicators.append("semantic_linguistic_variation")
    priority = (
        "exact_name_channel_cap_or_tie", "exact_core_channel_cap_or_tie",
        "token_reordering", "abbreviation", "normalization_failure",
        "transliteration_or_script_change", "missing_fields",
        "address_corruption_or_variation", "spelling_corruption_or_variation",
        "semantic_linguistic_variation",
    )
    primary = next((label for label in priority if label in indicators),
                   "unclassified")
    return {
        "primary": primary,
        "indicators": indicators,
        "name_similarity": round(name_similarity, 4),
        "address_similarity": round(address_similarity, 4) if address_similarity is not None else None,
        "source_scripts": sorted(scripts1),
        "target_scripts": sorted(scripts2),
    }
