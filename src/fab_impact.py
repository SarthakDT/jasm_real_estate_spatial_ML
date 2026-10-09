"""
Stage 5 — Did land near the JASM fab appreciate faster after the announcement?

TSMC confirmed its Kumamoto fab in October 2021, so the 2022 valuation is the
first that could reflect it. This stage compares plots near the fab with the
rest of the prefecture before and after that point:

  1. Distance bands: mean annual % change by distance from the fab and year.
  2. Difference-in-differences: annual % change regressed on
     near_fab x post_announcement with plot and year fixed effects, so each
     plot is compared with its own pre-announcement trend and each year with
     the prefecture-wide market. Standard errors are clustered by plot.
  3. Event study: the same comparison year by year relative to 2021. Flat
     pre-2022 coefficients mean near and far plots were moving together
     before the announcement, which is what the DiD relies on.

The pre-period goes back to PANEL_FIRST_YEAR using the price history MLIT
publishes with each plot in the 2021 release (the last before the
announcement), restricted to years in which the site was a benchmark.
"""

import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap, Normalize

from src import config, geo_utils, plot_style
from src.data_sources import chika_kouji
from src.data_sources.hazards import load_admin_boundaries
from src.logging_utils import get_logger, status

log = get_logger(__name__)


def _status(msg, *args): status(log, msg, *args)


PANEL_FIRST_YEAR = 2013          # first price year, so the first annual change is 2014
HISTORY_RELEASE_YEAR = 2021      # last release before the announcement
REFERENCE_YEAR = 2021            # event-study coefficients are relative to this year
TREATMENT_RADIUS_KM = 10
EARTHQUAKE_FIRST_VALUATION = 2017  # Kumamoto earthquakes, April 2016
BANDS = [(0, 5, "0–5 km"), (5, 10, "5–10 km"), (10, 20, "10–20 km"), (20, 40, "20–40 km"), (40, np.inf, "40+ km")]

DISTANCE_BANDS_PATH = os.path.join(config.ANALYSIS_DIR, "distance_bands.csv")
DID_RESULTS_PATH = os.path.join(config.ANALYSIS_DIR, "did_results.json")
PANEL_PATH = os.path.join(config.ANALYSIS_DIR, "price_change_panel.csv")
BANDS_FIGURE = os.path.join(config.FIGURES_DIR, "distance_bands.png")
EVENT_STUDY_FIGURE = os.path.join(config.FIGURES_DIR, "event_study.png")
MAP_FIGURE = os.path.join(config.FIGURES_DIR, "fab_map.png")


# ── Data ──────────────────────────────────────────────────────────────────────
def build_panel(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """One row per plot and year with the annual % price change, 2014-2026."""
    yearly = pd.concat([f[["plot_id", "year", "price"]] for f in frames.values()], ignore_index=True)
    history = chika_kouji.price_history(frames, HISTORY_RELEASE_YEAR, PANEL_FIRST_YEAR)
    history = history[history["year"] < config.YEARS[0]]
    prices = pd.concat([history, yearly], ignore_index=True)

    prev = prices.assign(year=prices["year"] + 1).rename(columns={"price": "prev_price"})
    panel = prices.merge(prev, on=["plot_id", "year"])
    panel["pct_change"] = (panel["price"] - panel["prev_price"]) / panel["prev_price"] * 100.0

    latest = (
        pd.concat([f[["plot_id", "year", "latitude", "longitude", "muni_code"]] for f in frames.values()])
        .sort_values("year").groupby("plot_id").last().drop(columns="year")
    )
    panel = panel.join(latest, on="plot_id")
    panel["municipality"] = panel["muni_code"].map(config.MUNI_NAMES)
    panel["distance_to_jasm_km"] = geo_utils.distance_to_jasm_km(panel)
    panel["band"] = pd.cut(panel["distance_to_jasm_km"], [b[0] for b in BANDS] + [np.inf],
                           labels=[b[2] for b in BANDS], right=False)
    panel["post"] = panel["year"] >= config.FIRST_POST_VALUATION_YEAR
    return panel.sort_values(["plot_id", "year"]).reset_index(drop=True)


def distance_band_table(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.groupby(["band", "year"], observed=True)
        .agg(mean_pct_change=("pct_change", "mean"), n_plots=("plot_id", "nunique"))
        .reset_index()
    )


# ── Regressions ───────────────────────────────────────────────────────────────
def _fit(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(data["plot_id"])[0]}
    )


def did_estimate(panel: pd.DataFrame, radius_km: float = TREATMENT_RADIUS_KM,
                 first_year: int | None = None, exclude_kumamoto_city: bool = False,
                 label: str = "") -> dict:
    """Average extra annual % change for plots within radius_km after the announcement."""
    d = panel.copy()
    if first_year:
        d = d[d["year"] >= first_year]
    if exclude_kumamoto_city:
        d = d[~d["municipality"].str.startswith("Kumamoto-shi")]
    d["near"] = d["distance_to_jasm_km"] <= radius_km
    d["near_post"] = (d["near"] & d["post"]).astype(int)
    fit = _fit("pct_change ~ near_post + C(plot_id) + C(year)", d)
    low, high = fit.conf_int().loc["near_post"]
    return {
        "label": label,
        "radius_km": radius_km,
        "years": f"{int(d['year'].min())}-{int(d['year'].max())}",
        "excludes_kumamoto_city": exclude_kumamoto_city,
        "estimate_pp": float(fit.params["near_post"]),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(fit.pvalues["near_post"]),
        "n_obs": int(fit.nobs),
        "n_plots": int(d["plot_id"].nunique()),
        "n_near_plots": int(d.loc[d["near"], "plot_id"].nunique()),
    }


def event_study(panel: pd.DataFrame, radius_km: float = TREATMENT_RADIUS_KM) -> pd.DataFrame:
    """Near-vs-far difference in annual % change for each year, relative to REFERENCE_YEAR."""
    d = panel.copy()
    near = d["distance_to_jasm_km"] <= radius_km
    years = [y for y in sorted(d["year"].unique()) if y != REFERENCE_YEAR]
    for y in years:
        d[f"near_{y}"] = (near & (d["year"] == y)).astype(int)
    fit = _fit("pct_change ~ " + " + ".join(f"near_{y}" for y in years) + " + C(plot_id) + C(year)", d)
    ci = fit.conf_int()
    rows = [{"year": y, "estimate_pp": fit.params[f"near_{y}"], "ci_low": ci.loc[f"near_{y}", 0],
             "ci_high": ci.loc[f"near_{y}", 1], "p_value": fit.pvalues[f"near_{y}"]} for y in years]
    rows.append({"year": REFERENCE_YEAR, "estimate_pp": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": np.nan})
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# ── Figures ───────────────────────────────────────────────────────────────────
def _mark_events(ax) -> None:
    ymax = ax.get_ylim()[1]
    for x, text in ((EARTHQUAKE_FIRST_VALUATION - 0.5, "Kumamoto\nearthquakes"),
                    (config.FIRST_POST_VALUATION_YEAR - 0.5, "Fab\nannounced")):
        ax.axvline(x, color=plot_style.TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.text(x + 0.1, ymax, text, fontsize=8, color=plot_style.TEXT_SECONDARY, va="top")


def plot_distance_bands(table: pd.DataFrame) -> None:
    fig, ax = plot_style.figure(8, 4.8)
    ax.axhline(0, color=plot_style.TEXT_SECONDARY, linewidth=0.8)
    for (_, _, label), color in zip(BANDS, plot_style.DISTANCE_RAMP):
        band = table[table["band"] == label]
        n = int(band.loc[band["year"] == band["year"].max(), "n_plots"].iloc[0])
        ax.plot(band["year"], band["mean_pct_change"], color=color, linewidth=2,
                marker="o", markersize=4, label=f"{label} ({n} plots)")
    _mark_events(ax)
    ax.set_ylabel("Mean annual price change (%)")
    ax.set_xticks(sorted(table["year"].unique()))
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.legend(title="Distance from fab", frameon=False, fontsize=8, title_fontsize=8,
              loc="upper left", bbox_to_anchor=(0.27, 0.9))
    plot_style.title(ax, "Land near the TSMC fab took off after the announcement",
                     "Kumamoto MLIT benchmark plots, mean year-on-year price change by distance from the JASM fab")
    plot_style.save(fig, BANDS_FIGURE)


def plot_event_study(series: dict[str, pd.DataFrame], radius_km: float) -> None:
    fig, ax = plot_style.figure(8, 4.8)
    ax.axhline(0, color=plot_style.TEXT_SECONDARY, linewidth=0.8)
    offsets = np.linspace(-0.12, 0.12, len(series))
    for (label, es), color, dx in zip(series.items(), plot_style.CATEGORICAL, offsets):
        ax.errorbar(es["year"] + dx, es["estimate_pp"],
                    yerr=[es["estimate_pp"] - es["ci_low"], es["ci_high"] - es["estimate_pp"]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.5, capsize=2, markersize=5, label=label)
    _mark_events(ax)
    years = next(iter(series.values()))["year"]
    ax.set_xticks(years)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("Extra annual price change (pp)")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    plot_style.title(ax, f"Plots within {radius_km:g} km of the fab vs the rest of Kumamoto",
                     f"Event-study estimates relative to {REFERENCE_YEAR}, plot and year fixed effects, "
                     "95% CI clustered by plot")
    plot_style.save(fig, EVENT_STUDY_FIGURE)


def plot_map(frames: dict[int, pd.DataFrame], radius_km: float) -> pd.DataFrame:
    first, last = HISTORY_RELEASE_YEAR, config.YEARS[-1]
    a = frames[first].set_index("plot_id")["price"]
    b = frames[last].set_index("plot_id")[["price", "latitude", "longitude"]]
    plots = b.join(a.rename("price_first"), how="inner")
    plots["cum_change"] = (plots["price"] / plots["price_first"] - 1) * 100
    pts = geo_utils.project(geo_utils.to_gdf(plots.reset_index()))
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
    pts.sort_values("cum_change").plot(ax=ax, column="cum_change", cmap=cmap, norm=norm, markersize=22,
                                       edgecolor="white", linewidth=0.4)
    fab.buffer(radius_km * 1000).boundary.plot(ax=ax, color=plot_style.TEXT_SECONDARY, linewidth=1, linestyle="--")
    fab.plot(ax=ax, marker="*", color=plot_style.TEXT, markersize=220, edgecolor="white", linewidth=0.8)
    x, y = fab.geometry.iloc[0].x, fab.geometry.iloc[0].y
    ax.annotate("JASM fab", (x, y), xytext=(12, 8), textcoords="offset points", fontsize=9, color=plot_style.TEXT)
    ax.annotate(f"{radius_km:g} km", (x + radius_km * 1000 * 0.72, y - radius_km * 1000 * 0.72),
                fontsize=8, color=plot_style.TEXT_SECONDARY)
    ax.set_axis_off()
    sm = ax.collections[1]
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
    cbar.set_ticks([t for t in (-10, 0, 20, 40, 60, 80, 100, 120, 140) if norm.vmin <= t <= norm.vmax])
    cbar.set_label(f"Price change {first}–{last} (%)", color=plot_style.TEXT_SECONDARY, fontsize=9)
    cbar.ax.tick_params(colors=plot_style.TEXT_SECONDARY, labelsize=8)
    cbar.outline.set_visible(False)
    plot_style.title(ax, f"Cumulative price change {first}–{last}",
                     f"{len(pts)} MLIT benchmark plots present in both years")
    plot_style.save(fig, MAP_FIGURE)
    return plots


# ── Main ──────────────────────────────────────────────────────────────────────
def run() -> dict:
    log.info("Estimating the fab's effect on land prices.")
    frames = chika_kouji.load_years()
    panel = build_panel(frames)
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    panel.to_csv(PANEL_PATH, index=False, encoding="utf-8-sig")

    bands = distance_band_table(panel)
    bands.to_csv(DISTANCE_BANDS_PATH, index=False, encoding="utf-8-sig")
    plot_distance_bands(bands)

    main = did_estimate(panel, label="Main: within 10 km, 2014-2026")
    robustness = [
        did_estimate(panel, radius_km=5, label="Within 5 km"),
        did_estimate(panel, radius_km=15, label="Within 15 km"),
        did_estimate(panel, first_year=2018, label="Pre-period from 2018 (after the earthquake dip)"),
        did_estimate(panel, exclude_kumamoto_city=True, label="Excluding Kumamoto City"),
    ]
    es = event_study(panel)
    outside_city = panel[~panel["municipality"].str.startswith("Kumamoto-shi")]
    es_outside_city = event_study(outside_city)
    plot_event_study({"All plots": es, "Excluding Kumamoto City": es_outside_city}, TREATMENT_RADIUS_KM)
    cumulative = plot_map(frames, TREATMENT_RADIUS_KM)

    results = {
        "treatment_radius_km": TREATMENT_RADIUS_KM,
        "first_post_valuation_year": config.FIRST_POST_VALUATION_YEAR,
        "main": main,
        "robustness": robustness,
        "event_study": es.replace({np.nan: None}).to_dict(orient="records"),
        "event_study_excluding_kumamoto_city": es_outside_city.replace({np.nan: None}).to_dict(orient="records"),
    }
    with open(DID_RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    _status("Wrote %s and figures in %s.", DID_RESULTS_PATH, config.FIGURES_DIR)

    wide = bands.pivot(index="band", columns="year", values="mean_pct_change")
    print()
    print("  Mean annual % change by distance from the fab")
    print(wide.loc[:, 2018:].round(1).to_string())
    print()
    print("  Difference-in-differences (extra annual % change near the fab after 2021)")
    for r in [main] + robustness:
        print(f"    {r['label']:<50} {r['estimate_pp']:+.2f} pp  "
              f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  p={r['p_value']:.3g}  "
              f"near plots={r['n_near_plots']}")
    print()
    print("  Event study (pp vs 2021): all plots | excluding Kumamoto City")
    both = es.merge(es_outside_city, on="year", suffixes=("_all", "_outside_city"))
    print(both[["year", "estimate_pp_all", "p_value_all", "estimate_pp_outside_city", "p_value_outside_city"]]
          .round(2).to_string(index=False))
    near = cumulative.join(panel.groupby("plot_id")["distance_to_jasm_km"].first())
    print()
    print(f"  Median cumulative change {HISTORY_RELEASE_YEAR}-{config.YEARS[-1]}: "
          f"within {TREATMENT_RADIUS_KM} km {near.loc[near['distance_to_jasm_km'] <= TREATMENT_RADIUS_KM, 'cum_change'].median():.1f}%, "
          f"beyond {near.loc[near['distance_to_jasm_km'] > TREATMENT_RADIUS_KM, 'cum_change'].median():.1f}%")
    return results


if __name__ == "__main__":
    run()
