"""
Stage 5 — Did land near the JASM fab appreciate faster after the announcement?

TSMC confirmed its Kumamoto fab in October 2021. Two independent official
surveys value benchmark plots each year:

  L01  Chika-Kouji, valued January 1 (about 250 plots)
  L02  Prefectural survey, valued July 1 (about 470 plots)

In both, the 2022 valuation is the first that could reflect the
announcement. This stage compares plots near the fab with the rest of the
prefecture before and after that point:

  1. Distance bands: mean annual % change by distance from the fab and year,
     for each survey.
  2. Difference-in-differences: annual % change regressed on
     near_fab x post_announcement with plot fixed effects and survey-by-year
     fixed effects, so each plot is compared with its own pre-announcement
     trend and each valuation with its own survey's prefecture-wide market.
     Standard errors are clustered by site: the ~25 sites valued by both
     surveys count as one cluster, not two.
  3. Event study: the same comparison year by year relative to 2021. Flat
     pre-2022 coefficients mean near and far plots were moving together
     before the announcement, which is what the DiD relies on.

The L01 pre-period goes back to 2013 through the price history MLIT ships with
the 2021 release; the L02 pre-period starts in 2015 from its yearly files.
"""

import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from src import config, geo_utils, plot_style
from src.data_sources import chika_kouji, land_price_survey, prefectural_survey
from src.data_sources.hazards import load_admin_boundaries
from src.logging_utils import get_logger

log = get_logger(__name__)


PANEL_FIRST_YEAR = 2013          # first L01 price year, so the first annual change is 2014
HISTORY_RELEASE_YEAR = 2021      # last release before the announcement
REFERENCE_YEAR = 2021            # event-study coefficients are relative to this year
TREATMENT_RADIUS_KM = 10
# First valuation after the April 2016 Kumamoto earthquakes, per survey: the
# July survey's 2016 valuation already came after them.
EARTHQUAKE_FIRST_VALUATION = {"L01": 2017, "L02": 2016}
# Distance rings for estimating how far the effect reaches; plots beyond the
# last ring are the comparison group.
RINGS_KM = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
BANDS = [(0, 5, "0–5 km"), (5, 10, "5–10 km"), (10, 20, "10–20 km"), (20, 40, "20–40 km"), (40, np.inf, "40+ km")]
SURVEY_LABELS = {"L01": "January survey (Chika-Kouji)", "L02": "July survey (prefectural)"}

DISTANCE_BANDS_PATH = os.path.join(config.ANALYSIS_DIR, "distance_bands.csv")
DID_RESULTS_PATH = os.path.join(config.ANALYSIS_DIR, "did_results.json")
PANEL_PATH = os.path.join(config.ANALYSIS_DIR, "price_change_panel.csv")
CUMULATIVE_PATH = os.path.join(config.ANALYSIS_DIR, "cumulative_change.csv")
BANDS_FIGURE = os.path.join(config.FIGURES_DIR, "distance_bands.png")
EVENT_STUDY_FIGURE = os.path.join(config.FIGURES_DIR, "event_study.png")
MAP_FIGURE = os.path.join(config.FIGURES_DIR, "fab_map.png")


# ── Data ──────────────────────────────────────────────────────────────────────
def _survey_panel(survey: str, prices: pd.DataFrame, frames: dict[int, pd.DataFrame],
                  site_ids: dict[str, str]) -> pd.DataFrame:
    """Annual % change per plot from (plot_id, year, price) rows, with location and site."""
    prev = prices.assign(year=prices["year"] + 1).rename(columns={"price": "prev_price"})
    panel = prices.merge(prev, on=["plot_id", "year"])
    panel["pct_change"] = (panel["price"] - panel["prev_price"]) / panel["prev_price"] * 100.0

    latest = (
        pd.concat([f[["plot_id", "year", "latitude", "longitude", "muni_code"]] for f in frames.values()])
        .sort_values("year").groupby("plot_id").last().drop(columns="year")
    )
    panel = panel.join(latest, on="plot_id")
    panel["survey"] = survey
    panel["site_id"] = panel["plot_id"].map(site_ids).fillna(panel["plot_id"])
    return panel


def build_panel(l01_frames: dict[int, pd.DataFrame], l02_frames: dict[int, pd.DataFrame],
                common_sites: pd.DataFrame) -> pd.DataFrame:
    """One row per plot and year with the annual % price change, both surveys."""
    l01_prices = pd.concat([f[["plot_id", "year", "price"]] for f in l01_frames.values()], ignore_index=True)
    history = chika_kouji.price_history(l01_frames, HISTORY_RELEASE_YEAR, PANEL_FIRST_YEAR)
    l01_prices = pd.concat([history[history["year"] < config.YEARS[0]], l01_prices], ignore_index=True)
    l02_prices = pd.concat([f[["plot_id", "year", "price"]] for f in l02_frames.values()], ignore_index=True)

    site_ids = dict(zip(common_sites["l02_plot_id"], common_sites["l01_plot_id"]))
    panel = pd.concat([
        _survey_panel("L01", l01_prices, l01_frames, {}),
        _survey_panel("L02", l02_prices, l02_frames, site_ids),
    ], ignore_index=True)
    panel["municipality"] = panel["muni_code"].map(config.MUNI_NAMES)
    panel["distance_to_jasm_km"] = geo_utils.distance_to_jasm_km(panel)
    panel["band"] = pd.cut(panel["distance_to_jasm_km"], [b[0] for b in BANDS] + [np.inf],
                           labels=[b[2] for b in BANDS], right=False)
    panel["post"] = panel["year"] >= config.FIRST_POST_VALUATION_YEAR
    panel["survey_year"] = panel["survey"] + "_" + panel["year"].astype(str)
    panel["valuation_date"] = panel["year"] + np.where(panel["survey"] == "L02", 0.5, 0.0)
    return panel.sort_values(["survey", "plot_id", "year"]).reset_index(drop=True)


def distance_band_table(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.groupby(["survey", "band", "year"], observed=True)
        .agg(mean_pct_change=("pct_change", "mean"), n_plots=("plot_id", "nunique"))
        .reset_index()
    )


# ── Regressions ───────────────────────────────────────────────────────────────
def _fit(formula: str, data: pd.DataFrame):
    """
    OLS with plot and survey-by-year fixed effects, errors clustered by site.
    Year dummies are built per survey, dropping each survey's first year, so
    they are not collinear with the plot dummies when both surveys are pooled.
    """
    data = data.copy()
    year_terms = []
    for survey, grp in data.groupby("survey"):
        for year in sorted(grp["year"].unique())[1:]:
            col = f"sy_{survey}_{year}"
            data[col] = ((data["survey"] == survey) & (data["year"] == year)).astype(int)
            year_terms.append(col)
    return smf.ols(f"{formula} + C(plot_id) + {' + '.join(year_terms)}", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(data["site_id"])[0]}
    )


def _subset(panel: pd.DataFrame, survey: str | None = None, first_year: int | None = None,
            exclude_kumamoto_city: bool = False, drop_years: tuple = ()) -> pd.DataFrame:
    d = panel
    if survey:
        d = d[d["survey"] == survey]
    if first_year:
        d = d[d["year"] >= first_year]
    if drop_years:
        d = d[~d["year"].isin(drop_years)]
    if exclude_kumamoto_city:
        d = d[~d["municipality"].str.startswith("Kumamoto-shi")]
    return d.copy()


def did_estimate(panel: pd.DataFrame, radius_km: float = TREATMENT_RADIUS_KM, survey: str | None = None,
                 first_year: int | None = None, exclude_kumamoto_city: bool = False,
                 drop_years: tuple = (), label: str = "") -> dict:
    """Average extra annual % change for plots within radius_km after the announcement."""
    d = _subset(panel, survey, first_year, exclude_kumamoto_city, drop_years)
    d["near"] = d["distance_to_jasm_km"] <= radius_km
    d["near_post"] = (d["near"] & d["post"]).astype(int)
    fit = _fit("pct_change ~ near_post", d)
    low, high = fit.conf_int().loc["near_post"]
    return {
        "label": label,
        "survey": survey or "both",
        "radius_km": radius_km,
        "years": f"{int(d['year'].min())}-{int(d['year'].max())}",
        "excludes_kumamoto_city": exclude_kumamoto_city,
        "estimate_pp": float(fit.params["near_post"]),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(fit.pvalues["near_post"]),
        "n_obs": int(fit.nobs),
        "n_plots": int(d["plot_id"].nunique()),
        "n_sites": int(d["site_id"].nunique()),
        "n_near_plots": int(d.loc[d["near"], "plot_id"].nunique()),
        "n_near_sites": int(d.loc[d["near"], "site_id"].nunique()),
    }


def event_study(panel: pd.DataFrame, radius_km: float = TREATMENT_RADIUS_KM) -> pd.DataFrame:
    """Near-vs-far difference in annual % change for each year, relative to REFERENCE_YEAR."""
    d = panel.copy()
    near = d["distance_to_jasm_km"] <= radius_km
    years = [y for y in sorted(d["year"].unique()) if y != REFERENCE_YEAR]
    for y in years:
        d[f"near_{y}"] = (near & (d["year"] == y)).astype(int)
    fit = _fit("pct_change ~ " + " + ".join(f"near_{y}" for y in years), d)
    ci = fit.conf_int()
    rows = [{"year": y, "estimate_pp": fit.params[f"near_{y}"], "ci_low": ci.loc[f"near_{y}", 0],
             "ci_high": ci.loc[f"near_{y}", 1], "p_value": fit.pvalues[f"near_{y}"]} for y in years]
    rows.append({"year": REFERENCE_YEAR, "estimate_pp": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": np.nan})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def ring_estimates(panel: pd.DataFrame, rings=RINGS_KM) -> pd.DataFrame:
    """
    Extra annual % change after the announcement for each distance ring,
    estimated jointly. Shows how far from the fab the effect reaches.
    """
    d = panel.copy()
    names = []
    for lo, hi in rings:
        name = f"ring_{lo}_{hi}"
        in_ring = d["distance_to_jasm_km"].between(lo, hi, inclusive="left")
        d[name] = (in_ring & d["post"]).astype(int)
        names.append(name)
    fit = _fit("pct_change ~ " + " + ".join(names), d)
    ci = fit.conf_int()
    rows = []
    for (lo, hi), name in zip(rings, names):
        in_ring = d["distance_to_jasm_km"].between(lo, hi, inclusive="left")
        rows.append({"ring": f"{lo}–{hi} km", "estimate_pp": fit.params[name], "ci_low": ci.loc[name, 0],
                     "ci_high": ci.loc[name, 1], "p_value": fit.pvalues[name],
                     "n_plots": d.loc[in_ring, "plot_id"].nunique(), "n_sites": d.loc[in_ring, "site_id"].nunique()})
    return pd.DataFrame(rows)


def cumulative_change(survey: str, frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Price change from the 2021 valuation (last before the announcement) to the latest, for plots in both."""
    first, last = HISTORY_RELEASE_YEAR, max(frames)
    start = frames[first].set_index("plot_id")["price"].rename("price_first")
    end = frames[last].set_index("plot_id")
    plots = end[["muni_code", "use_code", "lot_area_m2", "latitude", "longitude", "price"]].join(start, how="inner")
    plots = plots.rename(columns={"price": "price_last"})
    plots["survey"] = survey
    plots["municipality"] = plots["muni_code"].map(config.MUNI_NAMES)
    plots["land_use"] = plots["use_code"].map(config.LAND_USE_CODES).fillna("other")
    plots["distance_to_jasm_km"] = geo_utils.distance_to_jasm_km(plots)
    plots["cum_change"] = (plots["price_last"] / plots["price_first"] - 1) * 100
    return plots.drop(columns=["muni_code", "use_code"]).reset_index()


# ── Figures ───────────────────────────────────────────────────────────────────
def _mark_events(ax, survey: str | None = None) -> None:
    """Mark the 2016 earthquakes and the fab announcement. Pooled charts shade the
    year in which one survey had already been valued after the earthquakes and
    the other had not."""
    ymax = ax.get_ylim()[1]
    if survey:
        quake = EARTHQUAKE_FIRST_VALUATION[survey] - 0.5
        ax.axvline(quake, color=plot_style.TEXT_SECONDARY, linewidth=1, linestyle="--")
    else:
        quake = min(EARTHQUAKE_FIRST_VALUATION.values()) - 0.5
        ax.axvspan(quake, max(EARTHQUAKE_FIRST_VALUATION.values()) - 0.5,
                   color=plot_style.GRID, alpha=0.7, linewidth=0)
    ax.text(quake + 0.1, ymax, "Kumamoto\nearthquakes", fontsize=8, color=plot_style.TEXT_SECONDARY, va="top")
    announced = config.FIRST_POST_VALUATION_YEAR - 0.5
    ax.axvline(announced, color=plot_style.TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.text(announced + 0.1, ymax, "Fab\nannounced", fontsize=8, color=plot_style.TEXT_SECONDARY, va="top")


def plot_distance_bands(table: pd.DataFrame) -> None:
    fig, axes = plot_style.plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, facecolor=plot_style.SURFACE)
    years = sorted(table["year"].unique())
    for ax, survey in zip(axes, ("L01", "L02")):
        plot_style.style_axes(ax)
        ax.axhline(0, color=plot_style.TEXT_SECONDARY, linewidth=0.8)
        sub = table[table["survey"] == survey]
        for (_, _, label), color in zip(BANDS, plot_style.DISTANCE_RAMP):
            band = sub[sub["band"] == label]
            n = int(band.loc[band["year"] == band["year"].max(), "n_plots"].iloc[0])
            ax.plot(band["year"], band["mean_pct_change"], color=color, linewidth=2,
                    marker="o", markersize=4, label=f"{label} ({n} plots)")
        ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
        ax.set_xticks(years[::2])
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(title="Distance from fab", frameon=False, fontsize=7.5, title_fontsize=8,
                  loc="upper left", bbox_to_anchor=(0.2, 0.92))
        ax.set_title(f"{SURVEY_LABELS[survey]}, latest {int(sub['year'].max())}",
                     loc="left", fontsize=10, color=plot_style.TEXT)
    axes[0].set_ylabel("Mean annual price change (%)")
    for ax, survey in zip(axes, ("L01", "L02")):
        _mark_events(ax, survey)
    fig.suptitle("Land near the TSMC fab took off after the announcement, in both official surveys",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=plot_style.TEXT)
    fig.text(0.01, 0.905, "Mean year-on-year price change of MLIT benchmark plots by distance from the JASM fab",
             fontsize=9, color=plot_style.TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(BANDS_FIGURE, dpi=150, facecolor=plot_style.SURFACE, bbox_inches="tight", pad_inches=0.15)
    plot_style.plt.close(fig)


def plot_event_study(series: dict[str, pd.DataFrame], radius_km: float) -> None:
    fig, ax = plot_style.figure(8.5, 4.8)
    ax.axhline(0, color=plot_style.TEXT_SECONDARY, linewidth=0.8)
    offsets = np.linspace(-0.2, 0.2, len(series))
    for (label, es), color, dx in zip(series.items(), plot_style.CATEGORICAL, offsets):
        ax.errorbar(es["year"] + dx, es["estimate_pp"],
                    yerr=[es["estimate_pp"] - es["ci_low"], es["ci_high"] - es["estimate_pp"]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.3, capsize=2, markersize=4.5, label=label)
    _mark_events(ax)
    years = sorted(set().union(*(es["year"] for es in series.values())))
    ax.set_xticks(years)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("Extra annual price change (pp)")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    plot_style.title(ax, f"Plots within {radius_km:g} km of the fab vs the rest of Kumamoto",
                     f"Event-study estimates relative to {REFERENCE_YEAR}, plot and survey-by-year fixed effects, "
                     "95% CI clustered by site")
    plot_style.save(fig, EVENT_STUDY_FIGURE)


def plot_map(plots: pd.DataFrame, radius_km: float) -> None:
    first, last = HISTORY_RELEASE_YEAR, int(plots["survey"].map({"L01": config.YEARS[-1], "L02": config.L02_YEARS[-1]}).max())
    pts = geo_utils.project(geo_utils.to_gdf(plots))
    admin = load_admin_boundaries()
    fab = geo_utils.project(geo_utils.to_gdf(pd.DataFrame(
        {"latitude": [config.JASM_COORDINATES[0]], "longitude": [config.JASM_COORDINATES[1]]})))

    fig, ax = plot_style.figure(7.5, 7.5)
    ax.grid(False)
    admin.plot(ax=ax, color=plot_style.NEUTRAL, edgecolor="white", linewidth=0.5)
    # Diverging around 0 on a linear scale: red for declines, blue for rises.
    vmin = min(-10.0, float(np.floor(pts["cum_change"].min() / 10) * 10))
    vmax = float(np.ceil(pts["cum_change"].max() / 10) * 10)
    zero = -vmin / (vmax - vmin)
    cmap = LinearSegmentedColormap.from_list("div", [
        (0.0, plot_style.NEGATIVE), (zero, plot_style.NEUTRAL),
        (zero + (1 - zero) * 0.35, plot_style.SERIES), (1.0, plot_style.DISTANCE_RAMP[0]),
    ])
    norm = Normalize(vmin=vmin, vmax=vmax)
    markers = {"L01": "o", "L02": "s"}
    for survey, marker in markers.items():
        sub = pts[pts["survey"] == survey].sort_values("cum_change")
        ax.scatter(sub.geometry.x, sub.geometry.y, c=sub["cum_change"], cmap=cmap, norm=norm, s=20,
                   marker=marker, edgecolors="white", linewidths=0.4)
    fab.buffer(radius_km * 1000).boundary.plot(ax=ax, color=plot_style.TEXT_SECONDARY, linewidth=1, linestyle="--")
    fab.plot(ax=ax, marker="*", color=plot_style.TEXT, markersize=220, edgecolor="white", linewidth=0.8)
    x, y = fab.geometry.iloc[0].x, fab.geometry.iloc[0].y
    ax.annotate("JASM fab", (x, y), xytext=(12, 8), textcoords="offset points", fontsize=9, color=plot_style.TEXT)
    ax.annotate(f"{radius_km:g} km", (x + radius_km * 1000 * 0.72, y - radius_km * 1000 * 0.72),
                fontsize=8, color=plot_style.TEXT_SECONDARY)
    ax.set_axis_off()
    ax.legend(handles=[Line2D([], [], marker=m, linestyle="", color=plot_style.TEXT_SECONDARY, markersize=6,
                              label=SURVEY_LABELS[s]) for s, m in markers.items()],
              frameon=False, fontsize=8, loc="lower left")
    sm = plot_style.plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
    cbar.set_ticks([t for t in (-10, 0, 20, 40, 60, 80, 100, 120, 140) if norm.vmin <= t <= norm.vmax])
    cbar.set_label(f"Price change {first}–{last} (%)", color=plot_style.TEXT_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=plot_style.TEXT_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)
    plot_style.title(ax, f"Cumulative price change {first}–{last}",
                     f"{len(pts)} MLIT benchmark plots valued in both {first} and {last}, January and July surveys")
    plot_style.save(fig, MAP_FIGURE)


# ── Main ──────────────────────────────────────────────────────────────────────
def run() -> dict:
    log.info("Estimating the fab's effect on land prices.")
    l01_frames = chika_kouji.load_years()
    l02_frames = prefectural_survey.load_years()
    common = land_price_survey.match_common_sites(l01_frames, l02_frames)
    panel = build_panel(l01_frames, l02_frames, common)
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    panel.to_csv(PANEL_PATH, index=False, encoding="utf-8-sig")

    bands = distance_band_table(panel)
    bands.to_csv(DISTANCE_BANDS_PATH, index=False, encoding="utf-8-sig")
    plot_distance_bands(bands)

    main = did_estimate(panel, label="Main: within 10 km, both surveys")
    by_survey = [
        did_estimate(panel, survey="L01", label="January survey (L01) only"),
        did_estimate(panel, survey="L02", label="July survey (L02) only"),
    ]
    robustness = [
        did_estimate(panel, radius_km=5, label="Within 5 km"),
        did_estimate(panel, radius_km=15, label="Within 15 km"),
        did_estimate(panel, first_year=2018, label="Pre-period from 2018 (after the earthquake dip)"),
        did_estimate(panel, exclude_kumamoto_city=True, label="Excluding Kumamoto City"),
        # The 2021 valuations already show near plots pulling ahead, before the
        # October announcement; dropping them compares post-2022 with 2014-2020.
        did_estimate(panel, drop_years=(REFERENCE_YEAR,), label="Excluding the 2021 valuation"),
    ]
    es = event_study(panel)
    es_l02 = event_study(_subset(panel, survey="L02"))
    es_outside_city = event_study(_subset(panel, exclude_kumamoto_city=True))
    plot_event_study({"Both surveys": es, "July survey (L02) only": es_l02,
                      "Both surveys, excluding Kumamoto City": es_outside_city}, TREATMENT_RADIUS_KM)
    rings = ring_estimates(panel)
    cumulative = pd.concat([cumulative_change("L01", l01_frames), cumulative_change("L02", l02_frames)],
                           ignore_index=True)
    cumulative.to_csv(CUMULATIVE_PATH, index=False, encoding="utf-8-sig")
    plot_map(cumulative, TREATMENT_RADIUS_KM)

    records = lambda df: df.replace({np.nan: None}).to_dict(orient="records")
    results = {
        "treatment_radius_km": TREATMENT_RADIUS_KM,
        "first_post_valuation_year": config.FIRST_POST_VALUATION_YEAR,
        "main": main,
        "by_survey": by_survey,
        "robustness": robustness,
        "rings": records(rings),
        "event_study": records(es),
        "event_study_l02": records(es_l02),
        "event_study_excluding_kumamoto_city": records(es_outside_city),
    }
    with open(DID_RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log.info("Wrote %s and figures in %s.", DID_RESULTS_PATH, config.FIGURES_DIR)

    print()
    for survey in ("L01", "L02"):
        wide = bands[bands["survey"] == survey].pivot(index="band", columns="year", values="mean_pct_change")
        print(f"  Mean annual % change by distance from the fab, {SURVEY_LABELS[survey]}")
        print(wide.loc[:, 2018:].round(1).to_string())
        print()
    print("  Difference-in-differences (extra annual % change near the fab after 2021)")
    for r in [main] + by_survey + robustness:
        print(f"    {r['label']:<50} {r['estimate_pp']:+.2f} pp  "
              f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  p={r['p_value']:.3g}  "
              f"near plots={r['n_near_plots']} (sites={r['n_near_sites']})")
    print()
    print("  Event study (pp vs 2021): both | L02 only | both excluding Kumamoto City")
    both = es.merge(es_l02, on="year", how="left", suffixes=("", "_l02")).merge(
        es_outside_city, on="year", suffixes=("", "_outside_city"))
    print(both[["year", "estimate_pp", "p_value", "estimate_pp_l02", "p_value_l02",
                "estimate_pp_outside_city", "p_value_outside_city"]].round(2).to_string(index=False))
    print()
    print("  Extra annual % change after 2021 by distance ring (both surveys)")
    print(rings.round(2).to_string(index=False))
    print()
    for survey, sub in cumulative.groupby("survey"):
        near = sub["distance_to_jasm_km"] <= TREATMENT_RADIUS_KM
        print(f"  Median cumulative change, {SURVEY_LABELS[survey]}: within {TREATMENT_RADIUS_KM} km "
              f"{sub.loc[near, 'cum_change'].median():.1f}% ({near.sum()} plots), "
              f"beyond {sub.loc[~near, 'cum_change'].median():.1f}% ({(~near).sum()} plots)")
    return results


if __name__ == "__main__":
    run()
