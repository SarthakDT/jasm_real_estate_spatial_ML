"""
MLIT prefectural land price survey (都道府県地価調査), Kumamoto Prefecture.

Valued July 1 by the prefecture, 2015-2026 releases (MLIT publishes no 2014
file; 2015-2017 are shapefile-only). About 480 plots a year, roughly twice
L01, so it adds plots near the fab and a mid-year reading six months after
each L01 valuation. Forest plots (land-use code 020) are excluded.

Its records also carry a price history, but the "selected year" flags that
show whether the site was a benchmark in each year are truncated in these
files, so the history is not used: the pre-announcement period comes from
the linked yearly files instead.
"""

import pandas as pd

from src import config
from src.data_sources import land_price_survey as lps

SURVEY = lps.Survey(
    code="L02",
    name="Prefectural survey (July 1)",
    raw_dir=config.PREFECTURAL_SURVEY_RAW_DIR,
    years=config.L02_YEARS,
    schemas=config._L02_SCHEMA,
    valuation_month=7,
    stage1_path=config.STAGE1_L02_PATH,
    id_prefix="L02:",
    excluded_use_codes=config.L02_EXCLUDED_USE_CODES,
)


def load_years() -> dict[int, pd.DataFrame]:
    """All raw years, parsed and with plot_ids assigned."""
    return lps.load_years(SURVEY)


def transitions(frames: dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Validated year-over-year transitions."""
    frames = frames if frames is not None else load_years()
    return lps.validate(SURVEY, lps.build_transitions(SURVEY, frames))
