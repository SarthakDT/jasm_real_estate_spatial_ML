"""
JASM Spatial ML — Interactive Dashboard
Kumamoto Prefecture Chika-Kouji land-price forecasting pipeline.

Run with:
    streamlit run dashboard.py

Expects to sit at the project root, alongside data/ and models/.
"""

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.models.predict_dispatch import predict as model_predict
from src.spatial_cv import spatial_train_test_split
from src.train import N_SPATIAL_BLOCKS, N_TEST_BLOCKS

# ── Paths / constants (centralized in src/config.py) ────────────────────────
STAGE2_PATH = config.STAGE2_PATH
ANOMALY_PATH = config.ANOMALY_PATH
MODEL_PATH = config.MODEL_PATH
FEATURE_COLS = config.FEATURE_COLS
TARGET_COL = config.TARGET_COL
JASM_CENTROID = config.JASM_COORDINATES

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JASM Spatial ML Dashboard",
    page_icon="🗾",
    layout="wide",
)


# ── Cached loaders ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.isfile(STAGE2_PATH):
        return None, None
    df = pd.read_csv(STAGE2_PATH, encoding="utf-8-sig")
    # stage2.csv includes pre-announcement transition rows (added for the
    # fab event-study analysis); this dashboard's model-driven tabs only
    # cover the post-announcement window the model was trained on, matching
    # src/train.py's filter.
    df = df[df["years_since_announcement"] >= 0].reset_index(drop=True)
    anomalies = None
    if os.path.isfile(ANOMALY_PATH):
        anomalies = pd.read_csv(ANOMALY_PATH, encoding="utf-8-sig")
    return df, anomalies


@st.cache_resource
def load_model():
    # Loads models/best_model.pkl — whichever model won the Stage 3 comparison
    # (RandomForest / XGBoost / GaussianProcess / MixedEffects / SpatialError)
    # on the spatial-block test split. Falls back to the always-saved RF at
    # MODEL_PATH if best_model.pkl isn't present yet.
    best_path = os.path.join(config.MODEL_DIR, "best_model.pkl")
    path = best_path if os.path.isfile(best_path) else MODEL_PATH
    if not os.path.isfile(path):
        return None, None
    model_name = "RandomForest"
    name_path = os.path.join(config.MODEL_DIR, "best_model_name.txt")
    if path == best_path and os.path.isfile(name_path):
        model_name = open(name_path, encoding="utf-8").read().strip()
    return joblib.load(path), model_name


@st.cache_data
def compute_test_split(df: pd.DataFrame):
    """Reconstruct the identical spatial-block split used in src/train.py —
    NOT a plain random split, since nearby benchmark plots are spatially
    autocorrelated (see src/spatial_diagnostics.py) and the model itself was
    trained on this spatial split, not a random one.
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    train_idx, test_idx, _ = spatial_train_test_split(df, test_blocks=N_TEST_BLOCKS, n_blocks=N_SPATIAL_BLOCKS)
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


# ── Load everything ────────────────────────────────────────────────────────
df, anomalies_df = load_data()
model, model_name = load_model()

if df is None or model is None:
    st.error(
        "Couldn't find `data/processed/stage2.csv` and/or `models/jasm_rf_model.pkl`. "
        "Place `dashboard.py` in the project root (next to `main.py`) and make sure "
        "the pipeline has been run at least once (`python main.py --all`)."
    )
    st.stop()

X_train, X_test, y_train, y_test = compute_test_split(df)
train_pred = model_predict(model, model_name, X_train, df_slice=df.loc[X_train.index])
test_pred = model_predict(model, model_name, X_test, df_slice=df.loc[X_test.index])

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_mae = mean_absolute_error(y_test, test_pred)

eval_df = X_test.copy()
eval_df["municipality"] = df.loc[X_test.index, "municipality"]
eval_df["point_id"] = df.loc[X_test.index, "point_id"]
eval_df["actual"] = y_test
eval_df["predicted"] = test_pred
eval_df["residual"] = eval_df["actual"] - eval_df["predicted"]
eval_df["abs_error"] = eval_df["residual"].abs()

# ── Sidebar filters ─────────────────────────────────────────────────────────
st.sidebar.title("🗾 JASM Spatial ML")
st.sidebar.caption("Kumamoto Chika-Kouji land-price forecasting")

all_munis = sorted(df["municipality"].unique())
selected_munis = st.sidebar.multiselect(
    "Municipality", options=all_munis, default=[]
)
ysa_min, ysa_max = int(df["years_since_announcement"].min()), int(
    df["years_since_announcement"].max()
)
ysa_range = st.sidebar.slider(
    "Years since announcement (2021 anchor)",
    min_value=ysa_min,
    max_value=ysa_max,
    value=(ysa_min, ysa_max),
)

filtered = df.copy()
if selected_munis:
    filtered = filtered[filtered["municipality"].isin(selected_munis)]
filtered = filtered[
    (filtered["years_since_announcement"] >= ysa_range[0])
    & (filtered["years_since_announcement"] <= ysa_range[1])
]

st.sidebar.divider()
st.sidebar.metric("Rows in view", f"{len(filtered):,}")
st.sidebar.metric("Municipalities in view", filtered["municipality"].nunique())

# ── Header ───────────────────────────────────────────────────────────────
st.title("JASM Spatial ML — Model & Data Dashboard")
st.caption(
    f"{model_name} forecast of year-over-year benchmark land-price change "
    "(winner of a 5-model spatial-block comparison — see models/model_comparison.csv), "
    "Kumamoto Prefecture, 2019–2026 MLIT Chika-Kouji data."
)

# ── KPI row ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Transition rows", f"{len(df):,}")
k2.metric("Municipalities", df["municipality"].nunique())
k3.metric("Train R²", f"{train_r2:.3f}")
k4.metric("Test R²", f"{test_r2:.3f}")
k5.metric("Test RMSE (pp)", f"{test_rmse:.2f}")
k6.metric("Test MAE (pp)", f"{test_mae:.2f}")

st.divider()

tab_geo, tab_perf, tab_anom, tab_trend, tab_data = st.tabs(
    ["🗺️ Geospatial", "🌲 Model Performance", "🚩 Anomalies", "📈 Price Trends", "📋 Raw Data"]
)

# ── Tab 1: Geospatial ───────────────────────────────────────────────────
with tab_geo:
    st.subheader("Benchmark plots colored by year-over-year price change")
    fig = px.scatter_map(
        filtered,
        lat="latitude",
        lon="longitude",
        color="target_price_pct_change",
        size="past_year_price",
        size_max=18,
        hover_name="municipality",
        hover_data={
            "point_id": True,
            "past_year_price": ":,",
            "current_year_price": ":,",
            "target_price_pct_change": ":.2f",
            "latitude": False,
            "longitude": False,
        },
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        zoom=8.2,
        height=560,
        map_style="carto-positron",
    )
    fig.add_trace(
        go.Scattermap(
            lat=[JASM_CENTROID[0]],
            lon=[JASM_CENTROID[1]],
            mode="markers+text",
            marker=dict(size=16, color="black", symbol="star"),
            text=["JASM fab"],
            textposition="top center",
            name="JASM fab centroid",
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Spatial decay vs. distance to JASM")
        fig2 = px.scatter(
            filtered,
            x="distance_to_jasm_km",
            y="spatial_decay",
            color="target_price_pct_change",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            labels={
                "distance_to_jasm_km": "Distance to JASM (km)",
                "spatial_decay": "Spatial decay = 1 / (dist_km + 1)",
            },
        )
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.subheader("Mean abs. price change by municipality")
        muni_avg = (
            filtered.groupby("municipality")["target_price_pct_change"]
            .mean()
            .sort_values()
        )
        fig3 = px.bar(
            muni_avg,
            orientation="h",
            labels={"value": "Mean % price change", "municipality": ""},
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

# ── Tab 2: Model performance ─────────────────────────────────────────────
with tab_perf:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Feature importance")
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
            fig4 = px.bar(
                importances,
                orientation="h",
                labels={"value": "Importance", "index": ""},
                color=importances.values,
                color_continuous_scale="Blues",
            )
            fig4.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info(f"{model_name} doesn't expose feature_importances_ (only tree-ensemble models do).")

    with c2:
        st.subheader("Actual vs. predicted (test set)")
        fig5 = px.scatter(
            eval_df,
            x="actual",
            y="predicted",
            color="abs_error",
            color_continuous_scale="Oranges",
            hover_data=["municipality", "point_id"],
            labels={"actual": "Actual % change", "predicted": "Predicted % change"},
        )
        lo = min(eval_df["actual"].min(), eval_df["predicted"].min())
        hi = max(eval_df["actual"].max(), eval_df["predicted"].max())
        fig5.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                        line=dict(dash="dash", color="gray"))
        )
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Residual distribution")
    fig6 = px.histogram(eval_df, x="residual", nbins=40)
    fig6.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig6, use_container_width=True)

    with st.expander("Model & split details"):
        st.write(
            f"""
            - Model: **{model_name}** — winner of a 5-model spatial-block comparison
              (RandomForest, XGBoost, GaussianProcess, MixedEffects, SpatialError) on
              held-out Test R² in `models/model_comparison.csv` (Stage 3)
            - Split: spatial-block ({N_TEST_BLOCKS} of {N_SPATIAL_BLOCKS} KMeans-derived
              spatial blocks held out) — NOT a random split, since nearby plots are
              spatially autocorrelated (see the Moran's I diagnostics in
              `data/spatial_diagnostics.json`)
            - Train/test rows: {len(X_train):,} / {len(X_test):,}
            - Features used: {", ".join(FEATURE_COLS)}
            - Target: `target_price_pct_change` = ((current − past) / past) × 100
            """
        )

# ── Tab 3: Anomalies ───────────────────────────────────────────────────
with tab_anom:
    st.subheader("Top 5% highest-error predictions (test set)")
    threshold = eval_df["abs_error"].quantile(0.95)
    live_anomalies = eval_df[eval_df["abs_error"] >= threshold].sort_values(
        "abs_error", ascending=False
    )
    st.caption(f"Live-computed threshold: abs_error ≥ {threshold:.2f} pp — {len(live_anomalies)} rows")

    display_df = live_anomalies[
        ["point_id", "municipality", "latitude", "longitude", "past_year_price",
         "years_since_announcement", "actual", "predicted", "residual", "abs_error"]
    ].reset_index(drop=True)
    st.dataframe(
        display_df.style.format(
            {
                "latitude": "{:.5f}",
                "longitude": "{:.5f}",
                "past_year_price": "¥{:,.0f}",
                "actual": "{:.2f}",
                "predicted": "{:.2f}",
                "residual": "{:.2f}",
                "abs_error": "{:.2f}",
            }
        ),
        use_container_width=True,
        height=380,
    )

    fig7 = px.scatter_map(
        live_anomalies,
        lat="latitude",
        lon="longitude",
        color="abs_error",
        size="abs_error",
        hover_name="municipality",
        color_continuous_scale="Reds",
        zoom=8.2,
        height=480,
        map_style="carto-positron",
    )
    fig7.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig7, use_container_width=True)

    csv = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download anomalies as CSV", csv, "evaluation_anomalies_live.csv", "text/csv"
    )

# ── Tab 4: Price trends ───────────────────────────────────────────────
with tab_trend:
    st.subheader("Price trajectory by municipality")
    trend_munis = st.multiselect(
        "Municipalities to plot",
        options=all_munis,
        default=all_munis[:5],
        key="trend_muni_select",
    )
    if trend_munis:
        trend_df = df[df["municipality"].isin(trend_munis)].copy()
        trend_df["year"] = 2021 + trend_df["years_since_announcement"]
        agg = trend_df.groupby(["municipality", "year"])["current_year_price"].mean().reset_index()
        fig8 = px.line(
            agg, x="year", y="current_year_price", color="municipality", markers=True,
            labels={"current_year_price": "Mean current-year price (¥/m²)"},
        )
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("Pick at least one municipality above.")

    st.subheader("Distribution of % price change by years-since-announcement")
    fig9 = px.box(
        filtered, x="years_since_announcement", y="target_price_pct_change",
        points="outliers",
        labels={"years_since_announcement": "Years since 2021 announcement",
                "target_price_pct_change": "% price change"},
    )
    st.plotly_chart(fig9, use_container_width=True)

# ── Tab 5: Raw data ───────────────────────────────────────────────────
with tab_data:
    st.subheader("Filtered processed_stage2.csv")
    st.dataframe(filtered, use_container_width=True, height=500)
    st.caption(f"{len(filtered):,} of {len(df):,} rows shown (sidebar filters applied)")
