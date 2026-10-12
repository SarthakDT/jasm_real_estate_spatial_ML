"""
MLIT Chika-Kouji (地価公示) benchmark land prices, Kumamoto Prefecture.

The national survey, valued January 1, 2019-2026 releases. Loading, plot
linkage and validation are shared with the prefectural survey in
src/data_sources/land_price_survey.py; this module defines the L01 survey and
reads the multi-year price history that L01 ships with each record.
"""

import json

import pandas as pd

from src import config
from src.data_sources import land_price_survey as lps

SURVEY = lps.Survey(
    code="L01",
    name="Chika-Kouji (January 1)",
    raw_dir=config.CHIKA_KOUJI_RAW_DIR,
    years=config.YEARS,
    schemas=config._SCHEMA,
    valuation_month=1,
    stage1_path=config.STAGE1_PATH,
)


def load_years() -> dict[int, pd.DataFrame]:
    """All raw years, parsed and with plot_ids assigned."""
    return lps.load_years(SURVEY)


def transitions(frames: dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Validated year-over-year transitions."""
    frames = frames if frames is not None else load_years()
    return lps.validate(SURVEY, lps.build_transitions(SURVEY, frames))


def price_history(frames: dict[int, pd.DataFrame], release_year: int, first_year: int) -> pd.DataFrame:
    """
    Long-format (plot_id, year, price) history for every plot in one release,
    from first_year up to release_year, read from the history columns MLIT
    ships with each record. Years in which the site was not a benchmark plot
    are dropped, so the history always refers to the same site.
    """
    for (lo, hi), (flag_key, first_col) in config._HISTORY.items():
        if lo <= release_year <= hi:
            break
    else:
        raise ValueError(f"No history columns defined for {release_year}.")

    plot_ids = frames[release_year].set_index("number")["plot_id"]
    with open(SURVEY.data_path(release_year), encoding="utf-8-sig") as fh:
        features = json.load(fh)["features"]

    cols = SURVEY.schema_for_year(release_year)
    rows = []
    for feat in features:
        p = feat["properties"]
        number = "-".join(str(p[cols[k]]).strip() for k in ("muni_code", "use_code", "serial"))
        flags = str(p[flag_key])
        for year in range(first_year, release_year + 1):
            offset = year - config.HISTORY_FIRST_YEAR
            price = int(p.get(f"L01_{first_col + offset:03d}") or 0)
            if price > 0 and offset < len(flags) and flags[offset] == "1":
                rows.append({"plot_id": plot_ids[number], "year": year, "price": price})
    return pd.DataFrame(rows)
