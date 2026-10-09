# JASM Spatial ML

A geospatial machine-learning project analyzing land-price dynamics in Kumamoto Prefecture, Japan, centered on the economic-geography effect of the **JASM** semiconductor fab (the TSMC/Sony/Denso joint venture in Kikuyo-cho) on nearby land values, housing affordability, and infrastructure demand.

Kumamoto is in the middle of a real, current policy problem: the fab is driving rapid land-price appreciation and housing pressure in adjacent municipalities, layered on top of unresolved 2016-earthquake and 2020-flood disaster-recovery geography. This project aims to go beyond a single price-prediction model and build toward analysis a prefectural planner could actually use — proximity effects, displacement risk, hazard-aware growth siting, and infrastructure strain forecasts.

## Current state

The pipeline (`main.py` → `src/`) runs four stages end-to-end:

1. **Ingestion** (`src/load_data.py` → `src/data_sources/chika_kouji.py`) — parses 8 years (2019–2026) of MLIT *Chika-Kouji* (地価公示) benchmark land-price GeoJSON for Kumamoto Prefecture (code 43), builds year-over-year price-transition rows per benchmark plot, validates the schema, writes `data/processed/stage1.csv`.
2. **Feature engineering** (`src/features.py`) — converts to a GeoDataFrame, computes a vectorized projected-CRS distance and inverse-decay proximity feature to the JASM fab site, the target variable (% price change), and joins static hazard/land-use features from NLNI layers (`src/data_sources/hazards.py`) — flood risk class, landslide zone flag, urban planning zone type. Writes `data/processed/stage2.csv` (flat) and `stage2.geoparquet` (with geometry, for spatial modules).
3. **Training** (`src/train.py`) — trains and compares **six** models on an identical **spatial-block** train/test split (`src/spatial_cv.py`): the original Random Forest, a tuned XGBoost, a Gaussian Process with an ARD Matérn kernel (`src/models/gp.py`), a mixed-effects model with a random intercept per municipality (`src/models/mixed_effects.py`), and two interpretable spatial-econometrics baselines (`src/models/spatial_baseline.py`) — a spatial-lag model (`spreg.GM_Lag`, in-sample only — see its docstring for why) and a spatial-error model (`spreg.GM_Error`, genuinely evaluated out-of-sample). Writes `models/model_comparison.csv`, always saves the RF to `models/jasm_rf_model.pkl` (backward-compatible), saves whichever model wins to `models/best_model.pkl`, and fits a MAPIE conformal-interval wrapper around the winner — sweeping `plus`/`base`/`minmax` and keeping the tightest that still meets target coverage (`models/mapie_model.pkl`, sweep results in `models/mapie_sweep.json`).
4. **Evaluation** (`src/evaluate.py`) — reconstructs the identical spatial test split, exports top-5% error anomalies, a feature-importance chart, and Moran's I on both raw prices and the winning model's residuals (`src/spatial_diagnostics.py`, → `data/spatial_diagnostics.json`).

A standalone **out-of-time backtest** (`src/backtest.py`, run via `python -m src.backtest`) retrains the winning model on data through 2024 and forecasts 2025–2026 against the actuals already sitting in the raw data — a genuine test of forward-looking trust, not just held-out geography.

An interactive Streamlit dashboard (`dashboard.py`) visualizes the geospatial data, model performance, anomalies, and price trends.

**Honest, load-bearing finding from switching to a spatial-block split**: the original pipeline's plain random 75/25 split reported Test R²=0.626. Nearby benchmark plots are significantly spatially autocorrelated (Moran's I=0.49, p=0.001 on raw prices), so that number was inflated by leakage — a model can partly "memorize" a neighborhood rather than generalize to unseen geography. Under a proper spatial-block split (2 of 8 KMeans-derived blocks held out entirely), Random Forest's Test R² drops to **~0.13**, and a tuned XGBoost overfits outright (Test R² negative) with only ~1,000 spatially-held-out training rows. The **residual Moran's I for the winning model is still 0.575 (p=0.001)** — i.e. even the best model on offer leaves substantial spatial structure unexplained. The **out-of-time backtest tells a more encouraging story**: forecasting 2025 (one year ahead) scores R²≈0.66, though it degrades to R²≈0.26 for 2026 (two years ahead) — the model generalizes across time noticeably better than across untrained geography. All of this is reported directly, not smoothed over, because a government-facing tool that overclaims a single inflated R² is a bigger credibility risk than one that's candid about where it's weak.

**Random Forest stays the production model after real competition, not by default.** A Gaussian Process, a mixed-effects model, and a spatial-error model were all built specifically to try to beat it (see `docs/ANALYSIS_REPORT.md` §9 for the full account) — none did. The GP result is the most instructive: it was built to directly fix the residual-autocorrelation problem above, but its own residual Moran's I came in *higher* (0.73) than RF's, even after two rounds of re-tuning. The mechanism: a stationary spatial kernel has no support for a genuinely unseen spatial block, so it reverts toward the global mean uniformly across every point in that block — which is itself a spatially-correlated error pattern. This is reported as a real negative finding, not hidden.

**Correctness fix**: `src/config.py`'s `MUNI_NAMES` mapping (JIS X0402 municipality code → name) had 11 of 24 codes mislabeled in the original codebase — most importantly, code `43404` (the fab's actual host municipality) was mislabeled "Mashiki-machi" instead of "Kikuyo-cho", and the real Mashiki-machi (`43443`) was mislabeled "Nagomi-machi". Verified and corrected against the official Kumamoto Prefecture municipal code list and a point-in-polygon check of the JASM coordinates against NLNI admin boundary polygons. This only changes display labels, not any model input/output — all metrics are unaffected — but every municipality-level chart/table now shows correct names.

`data/processed/stage2.csv` also carries pre-2021 (pre-fab-announcement) transition rows now, so a future fab-impact event study has a baseline to compare against. Model training itself is unaffected — `train.py`/`evaluate.py` explicitly filter to `years_since_announcement >= 0` before building the feature matrix.

## Data sources & licensing

| Source | Publisher | License |
|---|---|---|
| Chika-Kouji (地価公示) benchmark land prices | MLIT | Japan Public Data License (PDL) 1.0 — commercial use permitted |
| A31 flood inundation risk zones, A33 landslide hazard zones, A29 urban planning zones, N03 admin boundaries — National Land Numerical Information (国土数値情報) | MLIT | PDL 1.0 |

Additional sources planned as part of the current expansion (see Roadmap) — MLIT reinfolib transaction prices, e-Stat demographic and housing-survey data, and PLATEAU building footprints — are all published under PDL 1.0 or equivalent open terms by Japanese government bodies.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Two data sources require a free API key. Copy `.env.example` to `.env` and fill in:

- `REINFOLIB_API_KEY` — register at https://www.reinfolib.mlit.go.jp/api/request
- `ESTAT_APP_ID` — register at https://www.e-stat.go.jp/api/

(Neither key is required to run the current pipeline — both are only used by data sources being added as part of the expansion described below.)

## Running the pipeline

```bash
python main.py --all          # run all 4 stages
python main.py --stage 2      # run a single stage (1-4)
python -m src.backtest        # out-of-time backtest (train <=2024, forecast 2025-2026)
streamlit run dashboard.py    # launch the dashboard
```

## Roadmap

This project is being expanded from a single-model price predictor into a broader Kumamoto policy-analysis tool. **Done**: proper spatial cross-validation and Moran's I diagnostics, a spatial-econometrics baseline alongside the ML models, gradient boosting with uncertainty intervals, an out-of-time backtest against 2025–2026 actuals, NLNI hazard/land-use layers. **Remaining**: three more data sources (MLIT reinfolib transaction prices, e-Stat demographics, PLATEAU building footprints), and four policy-analysis modules (fab proximity event study, affordability/displacement risk index, hazard-aware growth siting, infrastructure strain forecast) surfaced in an expanded dashboard with a what-if predictor and an exportable policy briefing.
