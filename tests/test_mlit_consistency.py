"""
Checks that the ingested data agrees with what MLIT itself publishes, for both
surveys (L01 Chika-Kouji, L02 prefectural survey).

Both surveys publish each plot's official year-on-year change rate (L01 from
2022, L02 from 2021). If plots are linked to the wrong previous-year record,
columns are mapped wrongly, or prices are parsed wrongly, the change we
compute will disagree with MLIT's.
"""

from decimal import ROUND_HALF_UP, Decimal

import pytest

from src import config, geo_utils
from src.data_sources import chika_kouji, land_price_survey, prefectural_survey

SOURCES = {"L01": chika_kouji, "L02": prefectural_survey}
FIRST_PUBLISHED_RATE = {"L01": 2022, "L02": 2021}


@pytest.fixture(scope="module")
def all_frames():
    return {code: module.load_years() for code, module in SOURCES.items()}


@pytest.fixture(scope="module", params=list(SOURCES))
def survey(request, all_frames):
    module = SOURCES[request.param]
    frames = all_frames[request.param]
    return request.param, module.SURVEY, frames, land_price_survey.build_transitions(module.SURVEY, frames)


def _mlit_round(past: int, current: int) -> float:
    """% change rounded to one decimal place, halves away from zero (MLIT's convention)."""
    pct = Decimal(current - past) / Decimal(past) * 100
    return float(pct.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def test_computed_change_matches_published_rate_exactly(survey):
    code, spec, _, transitions = survey
    published = transitions[transitions["published_change_pct"].notna()]
    assert set(published["valuation_year"]) == set(range(FIRST_PUBLISHED_RATE[code], spec.years[-1] + 1))
    computed = [_mlit_round(p, c) for p, c in zip(published["past_year_price"], published["current_year_price"])]
    mismatches = published.assign(computed=computed).query("computed != published_change_pct")
    assert mismatches.empty, mismatches[["point_id", "computed", "published_change_pct"]].head()


def test_one_record_per_plot_number_per_year(survey):
    _, _, frames, _ = survey
    for year, df in frames.items():
        assert df["number"].is_unique, f"duplicate MLIT plot numbers in {year}"
        assert df["plot_id"].is_unique, f"two records share a plot_id in {year}"


def test_linked_records_point_at_a_real_previous_year_record(survey):
    _, spec, frames, _ = survey
    for year in spec.years[1:]:
        cur, prev = frames[year], frames[year - 1]
        linked = cur[cur["linked"]]
        assert linked["prev_number"].isin(prev["number"]).all()
        # A plot_id carries over exactly when the record links to last year.
        assert linked["plot_id"].isin(prev["plot_id"]).all()
        assert not cur.loc[~cur["linked"], "plot_id"].isin(prev["plot_id"]).any()


def test_replaced_sites_start_a_new_history(survey):
    """MLIT reuses a plot number when it swaps in a new site; that must not continue the old plot's history."""
    _, spec, frames, _ = survey
    reused = 0
    for year in spec.years[1:]:
        cur, prev = frames[year], frames[year - 1]
        new_sites = cur[~cur["linked"] & cur["number"].isin(prev["number"])]
        old_ids = prev.set_index("number").loc[new_sites["number"], "plot_id"]
        assert (new_sites["plot_id"].to_numpy() != old_ids.to_numpy()).all()
        reused += len(new_sites)
    assert reused > 0, "expected some reused plot numbers"


def test_l01_price_history_matches_yearly_files(all_frames):
    """The pre-2019 L01 baseline uses MLIT's price history; where it overlaps the yearly files it must agree."""
    frames = all_frames["L01"]
    history = chika_kouji.price_history(frames, 2021, config.YEARS[0])
    for year in range(config.YEARS[0], 2022):
        hist = history[history["year"] == year].set_index("plot_id")["price"]
        actual = frames[year].set_index("plot_id")["price"]
        assert hist.index.isin(actual.index).all()
        assert (hist == actual[hist.index]).all()


def test_plot_ids_never_collide_between_surveys(all_frames):
    l01 = {pid for f in all_frames["L01"].values() for pid in f["plot_id"]}
    l02 = {pid for f in all_frames["L02"].values() for pid in f["plot_id"]}
    assert not l01 & l02


def test_common_points_match_an_l01_site(all_frames):
    """
    Every L02 record MLIT flags as a common point (共通地点) in a year that L01
    also covers lies at an L01 site, and those plots are paired. Flags from
    before 2019 cannot be checked: there is no L01 data for those years.
    """
    l01, l02 = all_frames["L01"], all_frames["L02"]
    common = land_price_survey.match_common_sites(l01, l02)
    paired = set(common["l02_plot_id"])
    for year in sorted(set(l01) & set(l02)):
        flagged = l02[year][l02[year]["common_point"]]
        assert set(flagged["plot_id"]) <= paired, f"unpaired common points in {year}"
        nearest = geo_utils.project(geo_utils.to_gdf(flagged)).sjoin_nearest(
            geo_utils.project(geo_utils.to_gdf(l01[year][["plot_id", "latitude", "longitude"]])),
            distance_col="distance_m")
        assert (nearest["distance_m"] <= config.COMMON_SITE_MAX_DISTANCE_M).all()
    assert common["l01_plot_id"].is_unique, "two L02 plots paired with the same L01 site"
