"""
Checks that the ingested data agrees with what MLIT itself publishes.

From 2022 MLIT includes each plot's official year-on-year change rate. If
plots are linked to the wrong previous-year record, or prices are parsed
wrongly, the change we compute will disagree with MLIT's.
"""

from decimal import ROUND_HALF_UP, Decimal

import pytest

from src import config
from src.data_sources import chika_kouji


@pytest.fixture(scope="module")
def frames():
    return chika_kouji.load_years()


@pytest.fixture(scope="module")
def transitions(frames):
    return chika_kouji._build_transitions(frames)


def _mlit_round(past: int, current: int) -> float:
    """% change rounded to one decimal place, halves away from zero (MLIT's convention)."""
    pct = Decimal(current - past) / Decimal(past) * 100
    return float(pct.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def test_computed_change_matches_published_rate_exactly(transitions):
    published = transitions[transitions["published_change_pct"].notna()]
    assert set(published["valuation_year"]) == set(range(2022, config.YEARS[-1] + 1))
    computed = [_mlit_round(p, c) for p, c in zip(published["past_year_price"], published["current_year_price"])]
    mismatches = published.assign(computed=computed).query("computed != published_change_pct")
    assert mismatches.empty, mismatches[["point_id", "computed", "published_change_pct"]].head()


def test_one_record_per_plot_number_per_year(frames):
    for year, df in frames.items():
        assert df["number"].is_unique, f"duplicate MLIT plot numbers in {year}"
        assert df["plot_id"].is_unique, f"two records share a plot_id in {year}"


def test_linked_records_point_at_a_real_previous_year_record(frames):
    for year in config.YEARS[1:]:
        cur, prev = frames[year], frames[year - 1]
        linked = cur[cur["linked"]]
        assert linked["prev_number"].isin(prev["number"]).all()
        # A plot_id carries over exactly when the record links to last year.
        assert linked["plot_id"].isin(prev["plot_id"]).all()
        assert not cur.loc[~cur["linked"], "plot_id"].isin(prev["plot_id"]).any()


def test_replaced_sites_start_a_new_history(frames):
    """MLIT reuses a plot number when it swaps in a new site; that must not continue the old plot's history."""
    reused = 0
    for year in config.YEARS[1:]:
        cur, prev = frames[year], frames[year - 1]
        new_sites = cur[~cur["linked"] & cur["number"].isin(prev["number"])]
        old_ids = prev.set_index("number").loc[new_sites["number"], "plot_id"]
        assert (new_sites["plot_id"].to_numpy() != old_ids.to_numpy()).all()
        reused += len(new_sites)
    assert reused > 0, "expected some reused plot numbers in 2019-2026"


def test_price_history_matches_yearly_files(frames):
    """The pre-2019 baseline uses MLIT's price history; where it overlaps the yearly files it must agree."""
    history = chika_kouji.price_history(frames, 2021, config.YEARS[0])
    for year in range(config.YEARS[0], 2022):
        hist = history[history["year"] == year].set_index("plot_id")["price"]
        actual = frames[year].set_index("plot_id")["price"]
        assert hist.index.isin(actual.index).all()
        assert (hist == actual[hist.index]).all()
