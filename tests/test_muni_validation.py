"""
Regression guard for the municipality-code class of bug: 11 of 24 codes in
config.MUNI_NAMES were once mislabeled (including the JASM fab's own host
municipality, 43404). data/reference/kumamoto_muni_codes.csv is the official
Kumamoto Prefecture code list, versioned in git so a future edit that
reintroduces a mislabel — or silently drops/adds a code — fails here instead
of requiring another manual review.
"""

import csv
import os

from src import config

REFERENCE_CSV = os.path.join(config.DATA_DIR, "reference", "kumamoto_muni_codes.csv")


def _load_reference() -> dict[str, str]:
    with open(REFERENCE_CSV, encoding="utf-8") as fh:
        return {row["muni_code"]: row["official_name_romaji"] for row in csv.DictReader(fh)}


def test_reference_csv_exists_and_has_rows():
    reference = _load_reference()
    assert len(reference) > 0


def test_muni_names_matches_reference_names():
    reference = _load_reference()
    mismatches = {
        code: (config.MUNI_NAMES[code], name)
        for code, name in reference.items()
        if code in config.MUNI_NAMES and config.MUNI_NAMES[code] != name
    }
    assert not mismatches, f"MUNI_NAMES disagrees with the authoritative code list: {mismatches}"


def test_muni_names_and_reference_have_identical_code_sets():
    reference_codes = set(_load_reference())
    muni_names_codes = set(config.MUNI_NAMES)
    missing_from_muni_names = reference_codes - muni_names_codes
    extra_in_muni_names = muni_names_codes - reference_codes
    assert not missing_from_muni_names, f"Codes in the reference list but missing from MUNI_NAMES: {missing_from_muni_names}"
    assert not extra_in_muni_names, f"Codes in MUNI_NAMES but not in the reference list: {extra_in_muni_names}"


def test_jasm_fab_host_municipality_is_kikuyo_cho():
    """The specific fact this project cannot afford to get wrong again."""
    assert config.MUNI_NAMES["43404"] == "Kikuyo-cho"
