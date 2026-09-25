"""Multi-view text normalization module for business entity resolution.

Produces raw, clean, folded, core, and alias views safely and deterministically,
preserving discriminative signals and Unicode integrity while ensuring idempotence.
"""

from __future__ import annotations

import os
import multiprocessing as mp
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

# Legal suffixes to remove from business names (longest / multi-word first)
LEGAL_SUFFIXES = [
    r"private limited company",
    r"private limited",
    r"pvt\.?\s*ltd\.?",
    r"p\.\s*ltd\.?",
    r"limited liability company",
    r"limited liability partnership",
    r"limited partnership",
    r"public limited company",
    r"pub\.?\s*ltd\.?",
    r"plc\.?",
    r"llc\.?",
    r"llp\.?",
    r"lp\.?",
    r"corporation",
    r"incorporated",
    r"corp\.?",
    r"inc\.?",
    r"limited",
    r"ltd\.?",
    r"gesellschaft mit beschränkter haftung",
    r"gmbh",
    r"aktiengesellschaft",
    r"a\s*g",
    r"société anonyme",
    r"societe anonyme",
    r"s\s*a\s*r\s*l",
    r"s\s*a\s*s",
    r"s\s*a",
    r"l\s*l\s*c",
    r"l\s*l\s*p",
    r"l\s*p",
    r"co\s*ltd",
    r"company",
    r"c\s*o",
]

# Compile legal suffixes regex: matches suffix at word boundary at end of string
_SUFFIX_PATTERN = re.compile(
    r"(?:[\s,/-]+|\b)(?:\(?(" + r"|".join(LEGAL_SUFFIXES) + r")\)?)+[\s.]*$",
    re.IGNORECASE,
)

# Address abbreviations dictionary (standardizing common terms)
ADDRESS_ABBREVIATIONS = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bdept\b": "department",
    r"\brm\b": "room",
    r"\bbldg\b": "building",
    r"\bctr\b": "center",
    r"\bstr\b": "street",
    r"\bno\b": "number",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
}

_COMPILED_ADDR_ABBR = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items()
]


@dataclass(frozen=True)
class NormalizedEntityRecord:
    entity_id: str
    raw_name: str
    name_clean: str
    name_folded: str
    name_core: str
    raw_address: str
    address_clean: str
    address_alias: str
    country: str


class TextNormalizer:
    """Safe, multi-view text normalization for entity names and addresses."""

    @staticmethod
    def clean_name(text: Optional[str]) -> str:
        """Produce `clean` view: NFKC normalize, casefold, safe punctuation."""
        if not text:
            return ""
        # 1. NFKC normalization
        normalized = unicodedata.normalize("NFKC", str(text))
        # 2. Casefold for language-safe lowercasing
        lowered = normalized.casefold()
        # 3. Replace ampersands with 'and'
        with_and = re.sub(r"&", " and ", lowered)
        # 4. Replace punctuation with spaces, keeping alphanumeric, accents, and hyphens/apostrophes
        cleaned = re.sub(r"[^\w\s\'-]", " ", with_and)
        # 5. Normalize internal whitespace
        return " ".join(cleaned.split())

    @staticmethod
    def fold_accents(text: Optional[str]) -> str:
        """Produce `folded` view: accent/diacritic removal via NFD decomposition."""
        if not text:
            return ""
        clean = TextNormalizer.clean_name(text)
        # Decompose Unicode characters (e.g., é -> e + accent mark)
        nfd_form = unicodedata.normalize("NFD", clean)
        # Filter out combining diacritical marks
        without_accents = "".join(
            char for char in nfd_form if unicodedata.category(char) != "Mn"
        )
        return " ".join(without_accents.split())

    @staticmethod
    def extract_core_name(text: Optional[str]) -> str:
        """Produce `core` view: stripped of trailing corporate and legal suffixes."""
        if not text:
            return ""
        folded = TextNormalizer.fold_accents(text)
        # Iteratively strip legal suffixes (e.g., 'Private Limited Company')
        prev = None
        current = folded
        while current != prev:
            prev = current
            current = _SUFFIX_PATTERN.sub("", current).strip()
            # Clean trailing dashes or commas
            current = re.sub(r"[,\s\'-]+$", "", current).strip()

        return current if current else folded

    @staticmethod
    def clean_address(text: Optional[str]) -> str:
        """Produce `address_clean` view: NFKC, casefold, normalized punctuation."""
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", str(text)).casefold()
        # Replace punctuation (commas, periods, hashes) with spaces
        cleaned = re.sub(r"[^\w\s\'-]", " ", normalized)
        return " ".join(cleaned.split())

    @staticmethod
    def alias_address(text: Optional[str]) -> str:
        """Produce `address_alias` view: expands common abbreviations (st, rd, apt, etc.)."""
        if not text:
            return ""
        clean_addr = TextNormalizer.clean_address(text)
        result = clean_addr
        for pattern, replacement in _COMPILED_ADDR_ABBR:
            result = pattern.sub(replacement, result)
        return " ".join(result.split())

    @classmethod
    def _normalize_single(cls, df: pd.DataFrame) -> pd.DataFrame:
        out_df = df.copy()
        out_df["name_clean"] = out_df["business_name"].apply(cls.clean_name)
        out_df["name_folded"] = out_df["business_name"].apply(cls.fold_accents)
        out_df["name_core"] = out_df["business_name"].apply(cls.extract_core_name)
        out_df["address_clean"] = out_df["business_address"].apply(cls.clean_address)
        out_df["address_alias"] = out_df["business_address"].apply(cls.alias_address)
        return out_df

    @classmethod
    def normalize_dataframe(cls, df: pd.DataFrame, n_jobs: int = -1, verbose: bool = True) -> pd.DataFrame:
        """Add all normalized views to a DataFrame with multi-core parallelism."""
        if len(df) < 50 or n_jobs == 1:
            return cls._normalize_single(df)

        n_workers = os.cpu_count() or 4 if n_jobs == -1 else n_jobs
        n_workers = max(1, min(n_workers, 32))

        chunk_size = (len(df) + n_workers - 1) // n_workers
        chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

        if verbose and len(df) >= 200:
            print(f"  [Multi-View Normalization ({n_workers} CPU cores)] Normalizing {len(df)} entities across {len(chunks)} parallel chunks...")

        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "fork")
        with ctx.Pool(processes=n_workers) as pool:
            norm_chunks = pool.map(_normalize_chunk_worker, chunks)

        return pd.concat(norm_chunks, ignore_index=True)


def _normalize_chunk_worker(chunk: pd.DataFrame) -> pd.DataFrame:
    """Worker function for multi-core dataframe normalization."""
    return TextNormalizer._normalize_single(chunk)
