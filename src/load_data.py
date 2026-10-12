"""
Stage 1 — Load both MLIT land-price surveys.

  L01  Chika-Kouji, valued January 1  -> data/processed/stage1.csv
  L02  Prefectural survey, July 1     -> data/processed/stage1_l02.csv

About 20 sites are valued by both surveys. They are paired by location
(data/processed/common_sites.csv) and share a site_id, so later analysis does
not count one physical plot as two independent observations.
"""

import os
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from src import config
from src.data_sources import chika_kouji, land_price_survey, prefectural_survey
from src.logging_utils import get_logger

log = get_logger(__name__)


def _mlit_round(past: int, current: int) -> float:
    """% change rounded to one decimal, halves away from zero (MLIT's convention)."""
    return float((Decimal(current - past) / Decimal(past) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def year_summary(frames: dict[int, pd.DataFrame], transitions: pd.DataFrame) -> pd.DataFrame:
    """Per-year check: records, share linked to the previous year, mismatches with MLIT's published rate."""
    rows = []
    for year, df in sorted(frames.items()):
        t = transitions[(transitions["valuation_year"] == year) & transitions["published_change_pct"].notna()]
        computed = [_mlit_round(p, c) for p, c in zip(t["past_year_price"], t["current_year_price"])]
        rows.append({
            "year": year,
            "records": len(df),
            "numbers_unique": df["number"].is_unique,
            "linked_share": df["linked"].mean() if year != min(frames) else float("nan"),
            "published_rates": len(t),
            "rate_mismatches": int(sum(c != p for c, p in zip(computed, t["published_change_pct"]))),
        })
    return pd.DataFrame(rows).set_index("year")


def run() -> dict[str, pd.DataFrame]:
    log.info("Loading MLIT land-price surveys.")
    l01_frames = chika_kouji.load_years()
    l02_frames = prefectural_survey.load_years()
    l01 = chika_kouji.transitions(l01_frames)
    l02 = prefectural_survey.transitions(l02_frames)

    common = land_price_survey.match_common_sites(l01_frames, l02_frames)
    l02["site_id"] = l02["plot_id"].map(common.set_index("l02_plot_id")["l01_plot_id"]).fillna(l02["plot_id"])
    log.info("Paired %d L02 plots with an L01 site (%d carry MLIT's common-point flag).",
             len(common), int(common["flagged"].sum()))

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    l01.to_csv(config.STAGE1_PATH, index=False, encoding="utf-8-sig")
    l02.to_csv(config.STAGE1_L02_PATH, index=False, encoding="utf-8-sig")
    common.to_csv(config.COMMON_SITES_PATH, index=False, encoding="utf-8-sig")
    log.info("Wrote %s, %s and %s.", config.STAGE1_PATH, config.STAGE1_L02_PATH, config.COMMON_SITES_PATH)

    for name, frames, t in (("L01 (January)", l01_frames, l01), ("L02 (July)", l02_frames, l02)):
        summary = year_summary(frames, t)
        print()
        print(f"  {name}: {len(t)} transitions across {t['plot_id'].nunique()} plots")
        print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
        if summary["rate_mismatches"].sum():
            raise ValueError(f"{name}: computed changes disagree with MLIT's published rate.")
    return {"L01": l01, "L02": l02, "common_sites": common}


if __name__ == "__main__":
    run()
