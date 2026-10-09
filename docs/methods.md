# Methods

How the numbers in the README are produced, and the choices behind them.

## Data

**MLIT Chika-Kouji (地価公示), Kumamoto Prefecture (code 43), 2019–2026 releases.** Each release lists about 250 benchmark plots with a price per m² valued as of January 1. Records also carry the plot's attributes and a price history back to 1983. The GeoJSON column layout changed twice, in 2022 and 2024. `config._SCHEMA` maps each era's property keys to canonical names. Every mapping was checked against the element names in that era's GML file and against value ranges in all eight years.

| Field | Used for |
|---|---|
| Plot number (municipality, land-use code, serial) and the previous year's number | linking plots across years |
| Price, published year-on-year change rate (2022+) | target and validation |
| Lot area, road width, nearest-station distance, floor-area and building-coverage ratios | model features |
| Land-use category (用途区分), zoning (用途地域), planning area (都市計画区域) | model features (ordinal codes) |
| Price history and "selected year" flags | pre-2019 baseline for the fab-impact analysis |

**NLNI layers:** A31 flood inundation depth (FY2012) and A33 landslide warning zones (FY2018), joined to plots by point-in-polygon; N03 administrative boundaries (2023) for the map. `scripts/download_data.py` fetches everything from https://nlftp.mlit.go.jp/ksj/.

## Linking plots across years

A plot's `plot_id` is its official number in the first year it appears, plus that year (for example `43404-000-002/2019`). Each later record inherits the id of the record its previous-year number points to.

- **Continuing plots** link every year.
- **Renumbered plots** (five in 2025) keep their history through the previous-number field.
- **Replaced sites** (site status 4 or 5) have a blank previous number, so they start a new history, even when MLIT reuses the old plot number.

Matching on rounded coordinates instead lost 63 of 1,712 transitions. Many were at the 2024 change from the JGD2000 to the JGD2011 datum, which moves coordinates in the fifth decimal place.

**Validation.** The computed change matches MLIT's published change rate exactly for all 1,216 transitions from 2022 onwards, using MLIT's rounding: one decimal place, halves away from zero. MLIT's price history agrees with the yearly files wherever the two overlap. Both checks are in `tests/test_mlit_consistency.py`.

## Timing of the announcement

TSMC confirmed the Kumamoto fab in October 2021. Valuations are as of January 1, so 2021 is the last valuation before the announcement and 2022 the first after. The first version of this project labelled 2021 as "year 0 after the announcement". It is now treated as pre-announcement.

## Fab impact (`src/fab_impact.py`)

**Panel.** Annual % change per plot from 2014 to 2026. Changes up to 2019 come from the price history in the 2021 release, kept only for years in which the site was a benchmark; later years come from the yearly files. That gives 271 plots and 3,051 plot-years. Distance to the fab is the straight-line distance from each plot's latest location to the JASM site, computed in JGD2011 / Japan Plane Rectangular CS II (EPSG:6670).

**Difference-in-differences.**

    pct_change[i,t] = β · near[i] · post[t] + plot[i] + year[t] + ε

`near` means within 10 km; `post` means valuation year 2022 or later. Plot fixed effects absorb each plot's level of growth; year fixed effects absorb prefecture-wide market swings. β is the average extra annual growth near the fab after the announcement. Standard errors are clustered by plot. Robustness runs use 5 km and 15 km cut-offs, a pre-period starting in 2018 (after the earthquake dip), and a sample without Kumamoto City.

**Event study.** The same model with `near × year` dummies, with 2021 as the reference year. It shows whether near and far plots moved together before the announcement:

- **All plots:** near plots ran about 1 pp a year behind in 2017–2020. Kumamoto City, which is in the comparison group, boomed during the post-earthquake rebuilding and dipped in 2021.
- **Excluding Kumamoto City:** the pre-announcement estimates are near zero, except for 2017, the first valuation after the April 2016 earthquakes.

## Model evaluation (`src/train.py`)

**Rows:** post-announcement valuations (2022–2026) with a previous-year change, 1,196 in total.

**Features:** last year's change, distance to the fab, coordinates, last year's price, valuation year, lot area, station distance, road width, floor-area ratio, land use, zoning group, planning area, flood class and landslide flag. The original `spatial_decay = 1/(distance+1)` was dropped: it is a monotone transform of distance, so it adds nothing for a tree model.

**Candidates:**

- Persistence (this year's change = last year's)
- Municipality-year mean
- Training mean
- Random Forest (fixed hyperparameters)
- XGBoost (randomized search with plot-grouped inner CV)

**Splits:**

| Split | What it tests |
|---|---|
| Unseen plots | 5-fold CV grouped by plot |
| Unseen areas | 8 KMeans blocks on projected coordinates, each held out once |
| Forecast | train on 2022–2024, predict 2025 and 2026, each from the previous year's change |
| Random rows | plain 5-fold CV; leaky because a plot's other years land in training; reference only |

**Spatial autocorrelation.** Moran's I uses an 8-nearest-neighbour weights matrix and is computed separately for each year. Pooling years would make every plot its own nearest neighbour.

**Forecast errors (`src/evaluate.py`).** Permutation importance is computed on the 2025–2026 forecasts (20 shuffles, R² scoring). Unlike impurity importance from training, it measures what a feature contributes on years the model has not seen.

## Things tried that did not help

- **Modelling the deviation from persistence.** Predicting this year's change minus last year's, then adding last year's back, scored R² 0.46 on the forecast split, worse than both plain models. The model learned the 2022–2024 acceleration and could not anticipate the 2026 slowdown.
- **Spatial models in an earlier version:** a Gaussian process, a mixed-effects model with municipality intercepts, and spatial lag and error models. None beat Random Forest on held-out areas, and they were removed to keep the comparison focused.
