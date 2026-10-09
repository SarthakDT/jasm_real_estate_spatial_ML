# Did TSMC's Kumamoto fab move land prices?

A geospatial analysis of Japanese government benchmark land prices (MLIT *Chika-Kouji*, 地価公示) from 2013 to 2026 around the JASM semiconductor fab that TSMC, Sony and Denso built in Kikuyo, Kumamoto.

TSMC confirmed the fab in October 2021. MLIT values land as of January 1, so the 2022 valuation is the first that could reflect it.

![Mean annual price change by distance from the fab](docs/figures/distance_bands.png)

## Findings

**Land near the fab appreciated much faster after the announcement.** Plots within 5 km of the fab rose 10–14% a year in 2023–2025, against under 1% a year for plots more than 20 km away. Between the 2021 and 2026 valuations the median plot within 10 km gained 20%, against 5% for the rest of the prefecture.

| Mean annual change | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| 0–5 km (13–16 plots) | 2.4% | 1.9% | 3.8% | **10.4%** | **13.3%** | **14.1%** | 7.6% |
| 5–10 km | 1.2% | 0.5% | 0.9% | 2.5% | 3.9% | 4.9% | 3.9% |
| 10–20 km | 3.3% | 0.8% | 1.2% | 2.0% | 2.9% | 3.5% | 3.9% |
| 20–40 km | −0.1% | −0.2% | −0.1% | 0.1% | 0.3% | 0.8% | 0.7% |
| 40+ km | −0.4% | −1.6% | −0.8% | −0.5% | −0.5% | −0.4% | −0.7% |

**The jump is attributable to the announcement, not just a general rise.** A difference-in-differences regression compares each plot with its own history and each year with the prefecture-wide market (plot and year fixed effects, standard errors clustered by plot, 271 plots, 3,051 plot-years from 2014 to 2026):

| Specification | Extra annual price growth near the fab after 2021 | 95% CI |
|---|---|---|
| **Within 10 km (main)** | **+3.0 pp** | 1.9 to 4.2 |
| Within 5 km | +6.6 pp | 4.0 to 9.2 |
| Within 15 km | +1.2 pp | 0.6 to 1.9 |
| Pre-period from 2018 (after the earthquake dip) | +3.3 pp | 2.1 to 4.5 |
| Excluding Kumamoto City | +4.9 pp | 3.0 to 6.8 |

The effect fades with distance and holds in every specification (all p < 0.001).

![Event study](docs/figures/event_study.png)

**The pre-announcement comparison is not perfectly clean, and this is why.** The event study estimates the near-vs-far gap year by year, relative to 2021. Across all plots, near plots grew about 1 pp a year more slowly than the rest in 2017–2020. That came from Kumamoto City in the comparison group, which boomed during the rebuilding after the April 2016 earthquakes and then dipped in the 2021 (COVID) valuation. Without Kumamoto City, the 2014–2020 gaps are flat and indistinguishable from zero, except 2017: the first valuation after the earthquakes, whose epicentre was next to the fab towns. Either way, the post-announcement gaps (up to +7 pp a year) are several times any pre-announcement difference. The effect starts small in 2022, peaks in 2024–2025, and narrows in 2026.

![Price change 2021-2026](docs/figures/fab_map.png)

## How good is the model, honestly?

The project also trains models to predict each plot's annual price change. The original version reported R² = 0.63 from a random train/test split. That number leaked: each plot appears once a year, so the random split put most test plots' other years into training.

Every candidate is now scored on the same three honest splits, next to three baselines that need no model. Rows are post-announcement valuations (2022–2026) that have a previous-year change (1,196 rows).

| R² | Unseen plots | Unseen areas | Forecast 2025–26 | Random rows (leaky, reference only) |
|---|---|---|---|---|
| Persistence: this year = last year's change | 0.69 | 0.69 | **0.72** | 0.69 |
| Municipality-year mean | 0.45 | −0.09 | n/a | 0.47 |
| Training mean | 0.00 | −0.07 | −0.13 | 0.00 |
| Random Forest | **0.79** | 0.69 | 0.64 | 0.79 |
| XGBoost | 0.78 | 0.61 | 0.69 | 0.82 |

- **Unseen plots** (5-fold, grouped by plot): Random Forest beats persistence (R² 0.79 vs 0.69, MAE 0.88 vs 0.96 pp). The plot attributes MLIT records (floor-area ratio, land use, road width, last year's price) help explain *which* plots move.
- **Unseen areas** (8 spatial blocks, each held out in turn): the models only tie persistence. Annual changes are strongly spatially correlated (Moran's I between 0.44 and 0.58 in every year, p = 0.001), so a held-out area is genuinely hard.
- **Forecast** (train on 2022–2024, predict 2025 and 2026): **neither model beats persistence.** 2022–2024 were accelerating years near the fab and 2026 slowed sharply, so models trained on the acceleration overshoot. Errors concentrate in Ozu, Kikuyo and Koshi, the towns closest to the fab, and in 2026 they cluster in space (residual Moran's I 0.37), which is the slowdown around the fab. Modelling the deviation from persistence instead of the level did worse (R² 0.46).

![Permutation importance](docs/figures/feature_importance.png)

On the forecast years, shuffling last year's change drops R² by 0.86. Shuffling distance to the fab changes nothing: once a plot's momentum is known, the fab adds no further information about next year. The fab effect shows up in the data as momentum, which is why the difference-in-differences above, not the forecasting model, is the right tool for measuring it.

## Data and validation

- **Source:** MLIT Chika-Kouji benchmark land prices for Kumamoto Prefecture, 2019–2026 releases (about 250 plots a year), plus the price history MLIT publishes with each plot in the 2021 release, which extends the pre-announcement period back to 2013. NLNI flood and landslide hazard zones and administrative boundaries. All published under the Japan Public Data License 1.0.
- **Plot linkage:** plots are followed across years with MLIT's official plot numbers and the previous-year number each record carries. MLIT renumbers plots and reuses numbers when it replaces a site. Matching on rounded coordinates, as the first version did, silently lost 63 of 1,712 transitions, many at the 2024 switch from the JGD2000 to the JGD2011 datum.
- **Checked against MLIT's own figures:** from 2022 MLIT publishes each plot's official change rate. The computed change matches it **exactly** for all 1,216 transitions, using MLIT's rounding (`tests/test_mlit_consistency.py`).
- **Municipality names:** 11 of the 24 municipality codes were mislabelled in the original code, including the fab's host town (43404, Kikuyo). The codes are now checked against the prefecture's official list (`tests/test_muni_validation.py`).

More detail is in [docs/methods.md](docs/methods.md).

## Limitations

- Benchmark prices are appraisals, not transactions, and appraisers see last year's value. That makes changes smoother and more persistent than market prices.
- Only 17 plots lie within 5 km of the fab, so the estimates closest to the fab have wide intervals.
- "Near the fab" is defined by straight-line distance. Other investment followed the fab into the same area, such as suppliers, housing and road projects. The estimate is the combined effect of the fab and everything that came with it.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # or: source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py  # fetch the MLIT files into data/raw/ (about 50 MB)
python main.py --all             # run all five stages
pytest                           # consistency checks plus an end-to-end run
streamlit run dashboard.py       # interactive dashboard
```

| Stage | Module | Output |
|---|---|---|
| 1. Load | `src/data_sources/chika_kouji.py` | one row per plot and year, linked by MLIT plot number |
| 2. Features | `src/features.py` | annual % change, last year's change, distance to the fab, MLIT plot attributes, hazard flags |
| 3. Models | `src/train.py` | `data/analysis/model_comparison.csv` |
| 4. Forecast errors | `src/evaluate.py` | worst forecasts, permutation importance, residual Moran's I |
| 5. Fab impact | `src/fab_impact.py` | distance bands, difference-in-differences, event study, figures |
