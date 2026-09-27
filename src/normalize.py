"""
normalize.py

Shared normalization used by both blocking (BM25 indexing) and features
(similarity scoring). Two entry points:

    normalize_name(text)    -> normalized business name
    normalize_address(text) -> normalized address string

Design notes:
  - Legal suffix list is intentionally generic/multilingual (not tied to
    US/India only) since the test set introduces France, which is unseen
    in training. Do NOT hardcode logic to a fixed {US, India} country set.
  - unidecode handles transliteration/diacritic noise (e.g. "Àmicale" ->
    "Amicale"), which matters for both French and Indian-transliterated
    names.
  - Suffix stripping is done on a *copy* used for blocking/matching keys;
    callers who also want the raw name for display/output should keep
    the original column untouched -- this module only ever returns the
    normalized string, never mutates input in place.
"""

from __future__ import annotations

import re
from unidecode import unidecode

# ---------------------------------------------------------------------------
# Legal suffix list -- generic, multilingual, not country-hardcoded.
# Matched as whole tokens after normalization, so order doesn't matter,
# but longer/more-specific phrases are listed first to avoid partial
# submatch issues when using phrase-based removal.
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = [
    # Common English / India / US
    "private limited", "pvt ltd", "pvt", "private", "limited", "ltd",
    "llp", "llc", "inc", "incorporated", "corp", "corporation",
    "company", "co", "industries", "enterprises", "enterprise",
    "group", "holdings", "solutions", "services", "international",
    "national", "global",
    # France
    "sarl", "sas", "sa", "sci", "eurl", "sasu", "snc",
    # Generic org markers seen across sources
    "plc", "gmbh", "ag", "bv", "nv", "oy", "ab",
]

# Sort longest-first so multi-word suffixes ("private limited") are
# stripped before their component single words ("private", "limited")
_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(s) for s in LEGAL_SUFFIXES),
                              key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# Common address abbreviation expansions -- helps token overlap between
# "Rd" and "Road", "St" and "Street", etc. Applied only in normalize_address.
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dist": "district",
    "twp": "township",
    "apt": "apartment",
    "bldg": "building",
    "hwy": "highway",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "sq": "square",
    "pl": "place",
    "no": "number",
}

_PUNCT_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _basic_clean(text) -> str:
    """Lowercase, transliterate, strip punctuation, collapse whitespace.

    Returns "" for null/empty input rather than raising, since source
    fields (especially address) can be missing.
    """
    if text is None:
        return ""
    text = str(text).strip()
    if text == "" or text.lower() == "nan":
        return ""

    text = unidecode(text)
    text = text.lower()
    text = _PUNCT_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def normalize_name(text) -> str:
    """Normalize a business_name for blocking/matching.

    Pipeline: clean -> strip legal suffixes -> collapse whitespace.
    Suffixes are removed everywhere they occur (not just at the end),
    since sources sometimes prefix them too (e.g. "PVT INDUSTRIES ... LIMITED").
    """
    cleaned = _basic_clean(text)
    if cleaned == "":
        return ""

    stripped = _SUFFIX_PATTERN.sub(" ", cleaned)
    stripped = _WHITESPACE_PATTERN.sub(" ", stripped).strip()

    # Guard: if suffix-stripping wiped out everything (e.g. name was
    # ONLY a legal suffix, unlikely but possible with junk data),
    # fall back to the cleaned-but-unstripped version so we never
    # return an empty blocking key for a non-empty input.
    return stripped if stripped != "" else cleaned


def normalize_address(text) -> str:
    """Normalize a business_address for blocking/matching.

    Pipeline: clean -> expand common abbreviations -> collapse whitespace.
    Does NOT attempt geocoding or gazetteer lookups (disallowed by the
    challenge's external-data rule) -- purely string-level normalization.
    """
    cleaned = _basic_clean(text)
    if cleaned == "":
        return ""

    tokens = cleaned.split(" ")
    expanded = [ADDRESS_ABBREVIATIONS.get(tok, tok) for tok in tokens]
    return " ".join(expanded)


def name_tokens(text) -> list[str]:
    """Convenience: normalized name split into tokens, for BM25 indexing
    or Jaccard/token-overlap features."""
    normalized = normalize_name(text)
    return normalized.split(" ") if normalized else []


if __name__ == "__main__":
    # Quick manual sanity check using rows from the sample dataset.
    samples = [
        "Brahma Infosoft",
        "Marina Ecole France Sarl",
        "SCI Ptit \u00c0micale",
        "PVT. INDUSTRIES PMB LEASING LIMITED",
        "Clm Agro [Limited]",
        "sci ligue ici parents",
        None,
        "",
    ]
    for s in samples:
        print(f"{s!r:45} -> {normalize_name(s)!r}")

    addr_samples = [
        "COIMATORE COLONY, HUNSUR TQMYSORE DIST., Karnataka",
        "63 R. DE DIEPPE, LILLE, Hauts-de-France",
        "8264 FILLY COURT, ROANOKE COUNTY, VA",
    ]
    print()
    for a in addr_samples:
        print(f"{a!r:60} -> {normalize_address(a)!r}")