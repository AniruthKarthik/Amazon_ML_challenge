"""Unit and integration tests for multi-view normalization (Phase 2)."""

import pandas as pd
import pytest
from chimera_submission.code.business_entity_resolution.src.normalization import TextNormalizer


def test_clean_name_basic_and_ampersand():
    raw = "Johnson & Johnson, Inc."
    clean = TextNormalizer.clean_name(raw)
    assert clean == "johnson and johnson inc"


def test_clean_name_idempotence():
    raw = "Böhm & Co. (Logistics) Ltd."
    clean1 = TextNormalizer.clean_name(raw)
    clean2 = TextNormalizer.clean_name(clean1)
    assert clean1 == clean2


def test_fold_accents_unicode():
    raw = "Société Générale de Café Müller"
    folded = TextNormalizer.fold_accents(raw)
    assert folded == "societe generale de cafe muller"


def test_fold_accents_idempotence():
    raw = "Café au Lait"
    folded1 = TextNormalizer.fold_accents(raw)
    folded2 = TextNormalizer.fold_accents(folded1)
    assert folded1 == folded2


def test_extract_core_name_legal_suffixes():
    cases = [
        ("Acme Corporation", "acme"),
        ("Apple Inc.", "apple"),
        ("Tata Consultancy Services Pvt. Ltd.", "tata consultancy services"),
        ("Infosys Limited", "infosys"),
        ("Siemens AG", "siemens"),
        ("BMW GMBH", "bmw"),
        ("TotalEnergies SE / SA", "totalenergies se"),
        ("XYZ Logistics (Pvt) Ltd.", "xyz logistics"),
        ("SingleWordCompany", "singlewordcompany"),
    ]
    for raw, expected in cases:
        core = TextNormalizer.extract_core_name(raw)
        assert core == expected, f"Failed for {raw}: got '{core}', expected '{expected}'"


def test_extract_core_name_idempotence():
    raw = "Reliance Industries Ltd."
    core1 = TextNormalizer.extract_core_name(raw)
    core2 = TextNormalizer.extract_core_name(core1)
    assert core1 == core2


def test_clean_address_and_alias():
    raw = "123 Main St., Apt. 4B, 5th Fl., near SBI ATM"
    clean = TextNormalizer.clean_address(raw)
    alias = TextNormalizer.alias_address(raw)
    assert "street" in alias
    assert "apartment" in alias
    assert "floor" in alias
    assert "near" in alias


def test_alias_address_idempotence():
    raw = "456 Oak Rd, Ste 10"
    alias1 = TextNormalizer.alias_address(raw)
    alias2 = TextNormalizer.alias_address(alias1)
    assert alias1 == alias2


def test_empty_and_none_handling():
    for fn in [
        TextNormalizer.clean_name,
        TextNormalizer.fold_accents,
        TextNormalizer.extract_core_name,
        TextNormalizer.clean_address,
        TextNormalizer.alias_address,
    ]:
        assert fn("") == ""
        assert fn(None) == ""
        assert fn("   ") == ""


def test_normalize_dataframe_preserves_raw():
    df = pd.DataFrame([
        {
            "entity_id": "S1-001",
            "business_name": "Crédit Agricole S.A.",
            "business_address": "12 Boulevard Pasteur, Paris",
            "country": "France",
        }
    ])
    norm_df = TextNormalizer.normalize_dataframe(df)

    # Check raw preserved
    assert norm_df.loc[0, "business_name"] == "Crédit Agricole S.A."
    assert norm_df.loc[0, "business_address"] == "12 Boulevard Pasteur, Paris"
    assert norm_df.loc[0, "country"] == "France"

    # Check multi-view fields added
    assert norm_df.loc[0, "name_clean"] == "crédit agricole s a"
    assert norm_df.loc[0, "name_folded"] == "credit agricole s a"
    assert norm_df.loc[0, "name_core"] == "credit agricole"
    assert "boulevard" in norm_df.loc[0, "address_alias"]
