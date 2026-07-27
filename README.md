# JASM Spatial ML

A geospatial machine learning pipeline built around official Japanese land valuation data (地価公示 / Chika-Kouji) for Kumamoto Prefecture. It tracks year-over-year benchmark land price changes across hundreds of survey plots and models those transitions using spatial and temporal features to predict future land valuation movement.

**JASM** = **J**apanese **A**sset **S**patial **M**achine Learning.

## Problem

Land valuation in Japan relies on annually published government benchmark prices at fixed survey plots. Kumamoto Prefecture is a particularly interesting case post-2016 earthquake — land prices across the prefecture didn't recover uniformly; some zones rebounded sharply while others stagnated.

**Research question:** Can a model trained on a benchmark plot's location, its municipality, and its historical price trajectory predict the direction and magnitude of its next-year price change?

## Data

- **Source:** Ministry of Land, Infrastructure, Transport and Tourism (MLIT), Japan — National Land Numerical Information
- **Scope:** Kumamoto Prefecture (code `43`), 2019–2026 (8 annual snapshots)
- **Format:** GeoJSON + Shapefile + XML metadata, one folder per year

MLIT changed its GeoJSON schema twice over this period; the pipeline maps all three variants to one canonical schema.

## Pipeline

```
src/
├── load_data.py   # Stage 1: ingestion, schema mapping, transition-row construction, validation
├── features.py    # Stage 2: geospatial feature engineering (distance/decay to JASM site, target %)
├── train.py        # Stage 3: Random Forest training + serialization
└── evaluate.py      # Stage 4: evaluation & spatial diagnostics
```

Run via the orchestrator:

```bash
python main.py --all        # run the full pipeline
python main.py --stage 2    # run a single stage (1-4)
```

## Results (current model)

| Metric | Value |
|---|---|
| Train R² | 0.85 |
| Test R² | 0.63 |
| Test RMSE | 2.74 pp |
| Test MAE | 1.34 pp |

Top features: `spatial_decay`, `distance_to_jasm_km`, `years_since_announcement` — spatial proximity to the reference site accounts for ~46% of predictive weight.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `geopy`, `joblib`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`

## Status

Stages 1–3 complete. Stage 4 (evaluation & spatial diagnostics) in progress.
