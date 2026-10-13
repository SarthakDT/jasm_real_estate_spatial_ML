"""
Dashboard for the JASM land-price analysis.

Run the pipeline first (`python main.py --all`), then:
    streamlit run dashboard.py
"""

import json
import os

import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config, fab_impact, plot_style, train

st.set_page_config(page_title="TSMC fab and Kumamoto land prices", page_icon=":material/factory:", layout="wide")

REQUIRED = [fab_impact.PANEL_PATH, fab_impact.DISTANCE_BANDS_PATH, fab_impact.DID_RESULTS_PATH, fab_impact.CUMULATIVE_PATH,
            config.MODEL_COMPARISON_PATH, config.FORECAST_PREDICTIONS_PATH]
FIRST, LAST = fab_impact.HISTORY_RELEASE_YEAR, max(config.YEARS[-1], config.L02_YEARS[-1])
ANNOUNCE_X = config.FIRST_POST_VALUATION_YEAR - 0.5
SURVEYS = {"L01": "January survey (Chika-Kouji)", "L02": "July survey (prefectural)"}


@st.cache_data
def load_outputs():
    panel = pd.read_csv(fab_impact.PANEL_PATH, encoding="utf-8-sig")
    bands = pd.read_csv(fab_impact.DISTANCE_BANDS_PATH, encoding="utf-8-sig")
    with open(fab_impact.DID_RESULTS_PATH, encoding="utf-8") as fh:
        did = json.load(fh)
    comparison = pd.read_csv(config.MODEL_COMPARISON_PATH)
    forecast = pd.read_csv(config.FORECAST_PREDICTIONS_PATH, encoding="utf-8-sig")
    best_name = open(config.BEST_MODEL_NAME_PATH, encoding="utf-8").read().strip()
    cum = pd.read_csv(fab_impact.CUMULATIVE_PATH, encoding="utf-8-sig")
    return panel, bands, did, comparison, forecast, best_name, cum


def event_rules(top: float, survey: str = "L01") -> alt.Chart:
    """Earthquake and announcement markers. The July survey's 2016 valuation already followed the earthquakes."""
    quake_x = fab_impact.EARTHQUAKE_FIRST_VALUATION[survey] - 0.5
    events = pd.DataFrame({"x": [quake_x, ANNOUNCE_X], "label": ["Kumamoto earthquakes", "Fab announced"], "y": [top, top]})
    rules = alt.Chart(events).mark_rule(strokeDash=[4, 4], color=plot_style.TEXT_SECONDARY).encode(x="x:Q")
    text = alt.Chart(events).mark_text(align="left", dx=4, dy=8, fontSize=11, color=plot_style.TEXT_SECONDARY).encode(
        x="x:Q", y=alt.Y("y:Q"), text="label:N")
    return rules + text


def year_axis():
    return alt.X("year:Q", title=None, axis=alt.Axis(format="d", tickMinStep=1))


if not all(os.path.isfile(p) for p in REQUIRED):
    st.error("Pipeline outputs are missing. Run `python scripts/download_data.py` and then `python main.py --all`.")
    st.stop()

panel, bands, did, comparison, forecast, best_name, cum = load_outputs()
main = did["main"]
by_survey = {r["survey"]: r for r in did["by_survey"]}
radius = did["treatment_radius_km"]
r2 = comparison.pivot(index="model_key", columns="split", values="r2")
latest_l02 = bands[(bands["survey"] == "L02") & (bands["band"] == fab_impact.BANDS[0][2])].set_index("year")["mean_pct_change"]
latest_year = int(latest_l02.index.max())

st.title("Did TSMC's Kumamoto fab move land prices?")
st.caption(
    "MLIT benchmark land prices for Kumamoto Prefecture from two official surveys: Chika-Kouji (valued January 1) "
    "and the prefectural survey (valued July 1). TSMC announced the JASM fab in Kikuyo in October 2021; "
    "the 2022 valuation is the first after it in both."
)

with st.container(horizontal=True):
    st.metric(f"Extra annual growth within {radius} km", f"{main['estimate_pp']:+.1f} pp",
              f"95% CI {main['ci_low']:+.1f} to {main['ci_high']:+.1f}", delta_color="off", border=True,
              help="Difference-in-differences on both surveys: plot and survey-by-year fixed effects, "
                   "standard errors clustered by site.")
    st.metric("Same estimate, January / July survey alone",
              f"{by_survey['L01']['estimate_pp']:+.1f} / {by_survey['L02']['estimate_pp']:+.1f} pp",
              "two independent surveys", delta_color="off", border=True)
    st.metric(f"Latest reading: July {latest_year}, within 5 km", f"{latest_l02[latest_year]:+.1f}%",
              f"July {latest_year - 1}: {latest_l02[latest_year - 1]:+.1f}%", delta_color="off", border=True,
              help="Mean annual change of July-survey plots within 5 km of the fab.")
    st.metric(f"Forecast R², {best_name} vs persistence",
              f"{r2.loc['xgboost' if best_name == 'XGBoost' else 'random_forest', 'forecast']:.2f}",
              f"persistence {r2.loc['persistence', 'forecast']:.2f}", delta_color="off", border=True)

tab_impact, tab_models, tab_errors, tab_data = st.tabs(
    [":material/factory: Fab impact", ":material/leaderboard: Model evaluation",
     ":material/error: Forecast errors", ":material/table: Data"]
)

# ── Fab impact ────────────────────────────────────────────────────────────────
with tab_impact:
    band_order = [b[2] for b in fab_impact.BANDS]
    survey = st.segmented_control("Survey", list(SURVEYS), default="L02", format_func=lambda s: SURVEYS[s],
                                  key="survey") or "L02"
    survey_bands = bands[bands["survey"] == survey]
    survey_cum = cum[cum["survey"] == survey]
    near = survey_cum["distance_to_jasm_km"] <= radius
    with st.container(border=True):
        st.subheader(f"Mean annual price change by distance from the fab, {SURVEYS[survey].lower()}")
        st.caption(f"Median change {FIRST}–{LAST}: {survey_cum.loc[near, 'cum_change'].median():+.0f}% for "
                   f"{near.sum()} plots within {radius} km, {survey_cum.loc[~near, 'cum_change'].median():+.0f}% "
                   f"for {(~near).sum()} plots farther away.")
        top = float(survey_bands["mean_pct_change"].max())
        lines = alt.Chart(survey_bands).mark_line(point=True, strokeWidth=2).encode(
            x=year_axis(),
            y=alt.Y("mean_pct_change:Q", title="Mean annual change (%)"),
            color=alt.Color("band:N", title="Distance from fab", sort=band_order,
                            scale=alt.Scale(domain=band_order, range=plot_style.DISTANCE_RAMP)),
            tooltip=[alt.Tooltip("band:N", title="Distance"), alt.Tooltip("year:Q", format="d"),
                     alt.Tooltip("mean_pct_change:Q", title="Mean change (%)", format=".1f"),
                     alt.Tooltip("n_plots:Q", title="Plots")],
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=plot_style.TEXT_SECONDARY).encode(y="y:Q")
        st.altair_chart(zero + lines + event_rules(top, survey), width="stretch")

    with st.container(border=True):
        st.subheader(f"Event study: plots within {radius} km vs the rest")
        st.caption(f"Extra annual price change relative to {fab_impact.REFERENCE_YEAR}, with 95% confidence intervals. "
                   "Before the 2016 earthquakes near and far plots moved together; near plots then ran about "
                   "1 pp a year behind until 2021, and pulled ahead from 2022.")
        samples = {"Both surveys": ("event_study", -0.2), "July survey (L02) only": ("event_study_l02", 0.0),
                   "Both, excluding Kumamoto City": ("event_study_excluding_kumamoto_city", 0.2)}
        es = pd.concat([pd.DataFrame(did[key]).assign(sample=name) for name, (key, _) in samples.items()])
        es["year_shift"] = es["year"] + es["sample"].map({name: dx for name, (_, dx) in samples.items()})
        color = alt.Color("sample:N", title=None, sort=list(samples),
                          scale=alt.Scale(domain=list(samples), range=plot_style.CATEGORICAL[:3]))
        base = alt.Chart(es).encode(x=alt.X("year_shift:Q", title=None, axis=alt.Axis(format="d", tickMinStep=1)),
                                    color=color)
        bars = base.mark_rule(strokeWidth=2).encode(y=alt.Y("ci_low:Q", title="Extra annual change (pp)"), y2="ci_high:Q")
        dots = base.mark_circle(size=70, opacity=1).encode(
            y="estimate_pp:Q",
            tooltip=[alt.Tooltip("sample:N"), alt.Tooltip("year:Q", format="d"),
                     alt.Tooltip("estimate_pp:Q", title="Estimate (pp)", format="+.2f"),
                     alt.Tooltip("ci_low:Q", title="95% CI low", format="+.2f"),
                     alt.Tooltip("ci_high:Q", title="95% CI high", format="+.2f")],
        )
        st.altair_chart(zero + bars + dots + event_rules(float(es["ci_high"].max())), width="stretch")

    col_map, col_table = st.columns([3, 2])
    with col_map:
        with st.container(border=True):
            st.subheader(f"Price change {FIRST}–{LAST}, {SURVEYS[survey].lower()}")
            fig = px.scatter_map(
                survey_cum, lat="latitude", lon="longitude", color="cum_change", hover_name="municipality",
                hover_data={"cum_change": ":.1f", "distance_to_jasm_km": ":.1f", "latitude": False, "longitude": False},
                labels={"cum_change": "Change (%)", "distance_to_jasm_km": "Distance to fab (km)"},
                color_continuous_scale=[plot_style.NEGATIVE, plot_style.NEUTRAL, plot_style.SERIES, plot_style.DISTANCE_RAMP[0]],
                color_continuous_midpoint=0, zoom=8, height=520, map_style="carto-positron",
            )
            fig.add_trace(go.Scattermap(lat=[config.JASM_COORDINATES[0]], lon=[config.JASM_COORDINATES[1]],
                                        mode="markers+text", marker={"size": 14, "color": "black"},
                                        text=["JASM fab"], textposition="top right", name="JASM fab", showlegend=False))
            fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
            st.plotly_chart(fig, width="stretch")
    with col_table:
        with st.container(border=True):
            st.subheader("Difference-in-differences")
            st.caption("Average extra annual % change near the fab after 2021.")
            did_table = pd.DataFrame([main] + did["by_survey"] + did["robustness"])[
                ["label", "estimate_pp", "ci_low", "ci_high", "p_value", "n_near_plots"]]
            st.dataframe(
                did_table, hide_index=True, width="stretch",
                column_config={
                    "label": st.column_config.TextColumn("Specification", width="large"),
                    "estimate_pp": st.column_config.NumberColumn("Estimate (pp)", format="%+.2f"),
                    "ci_low": st.column_config.NumberColumn("CI low", format="%+.2f"),
                    "ci_high": st.column_config.NumberColumn("CI high", format="%+.2f"),
                    "p_value": st.column_config.NumberColumn("p", format="%.1e"),
                    "n_near_plots": st.column_config.NumberColumn("Near plots"),
                },
            )

# ── Model evaluation ─────────────────────────────────────────────────────────
with tab_models:
    st.markdown(
        "Every candidate is scored on the same splits. The random-row split is shown only for reference: "
        "it puts other years of the same plot in training and flatters every model."
    )
    metric = st.segmented_control("Metric", ["r2", "mae", "rmse"], default="r2",
                                  format_func={"r2": "R²", "mae": "MAE (pp)", "rmse": "RMSE (pp)"}.get)
    table = comparison.pivot(index="model", columns="split_label", values=metric or "r2")
    table = table.reindex(index=list(train.CANDIDATES.values()), columns=list(train.SPLITS.values()))
    st.dataframe(table, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns})
    st.caption("Municipality-year mean is not available for the forecast split: same-year means do not exist yet "
               "when forecasting.")

    with st.container(border=True):
        st.subheader("What drives the forecast")
        st.image(config.CHART_PATH)

# ── Forecast errors ──────────────────────────────────────────────────────────
with tab_errors:
    forecast["abs_error"] = forecast["residual"].abs()
    forecast["persistence_abs_error"] = (forecast["actual"] - forecast["pred_persistence"]).abs()
    year = st.segmented_control("Valuation year", sorted(forecast["valuation_year"].unique()),
                                default=int(forecast["valuation_year"].max()), format_func=str)
    view = forecast if year is None else forecast[forecast["valuation_year"] == year]

    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.subheader(f"Actual vs predicted ({best_name})")
            lo, hi = float(min(view["actual"].min(), view["predicted"].min())), float(max(view["actual"].max(), view["predicted"].max()))
            diag = alt.Chart(pd.DataFrame({"x": [lo, hi]})).mark_line(strokeDash=[4, 4], color=plot_style.TEXT_SECONDARY).encode(x="x:Q", y="x:Q")
            pts = alt.Chart(view).mark_circle(size=40, color=plot_style.SERIES, opacity=0.7).encode(
                x=alt.X("actual:Q", title="Actual change (%)"), y=alt.Y("predicted:Q", title="Predicted change (%)"),
                tooltip=["municipality", alt.Tooltip("actual:Q", format=".1f"), alt.Tooltip("predicted:Q", format=".1f"),
                         alt.Tooltip("pred_persistence:Q", title="Persistence", format=".1f")])
            st.altair_chart(diag + pts, width="stretch")
    with col_b:
        with st.container(border=True):
            st.subheader("Where the errors are")
            fig = px.scatter_map(view, lat="latitude", lon="longitude", color="abs_error", size="abs_error",
                                 hover_name="municipality", color_continuous_scale=plot_style.DISTANCE_RAMP[::-1],
                                 labels={"abs_error": "|error| (pp)"}, zoom=8, height=380, map_style="carto-positron")
            fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
            st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.subheader("Mean absolute error by municipality")
        by_muni = (view.groupby("municipality")[["abs_error", "persistence_abs_error"]].mean()
                   .assign(plots=view.groupby("municipality").size()).sort_values("abs_error", ascending=False))
        st.dataframe(by_muni, width="stretch", column_config={
            "abs_error": st.column_config.NumberColumn(f"{best_name} (pp)", format="%.2f"),
            "persistence_abs_error": st.column_config.NumberColumn("Persistence (pp)", format="%.2f"),
        })

# ── Data ─────────────────────────────────────────────────────────────────────
with tab_data:
    munis = st.multiselect("Municipality", sorted(panel["municipality"].dropna().unique()))
    shown = panel if not munis else panel[panel["municipality"].isin(munis)]
    st.dataframe(shown, hide_index=True, width="stretch", height=500)
    st.caption(f"{len(shown):,} plot-years. Source: MLIT Chika-Kouji (地価公示), Japan Public Data License 1.0.")
