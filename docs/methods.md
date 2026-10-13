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

**MLIT prefectural land price survey (都道府県地価調査, L02), 2015–2026 releases.** Valued July 1 by the prefecture, about 470 plots a year. Notes on the files:

- **Missing years and formats.** MLIT publishes no 2014 file, and 2015–2017 are shapefiles only (read as cp932).
- **Four column layouts:** 2015–16, 2017–20, 2021–25 and 2026. They were mapped by pairing 120 records per year between the GML (which names each field) and the GeoJSON or shapefile, then checked against value ranges in all twelve years. The schema is `config._L02_SCHEMA`.
- **Fields missing from 2015–16.** Zoning and planning area are absent, so stage 2 takes them from the plot's next record.
- **Published change rate** from 2021.
- **Forest plots** (land-use code 020, numbered prefecture-wide under code 43900) are dropped: the January survey has none.
- **Price history not used.** The history columns' "selected year" flags are truncated in these files, so the history cannot be checked for site changes. The pre-announcement period comes from the linked yearly files instead.

The survey covers all 49 Kumamoto municipalities, including a category of plots outside any planning area (都計外).

**NLNI layers:** A31 flood inundation depth (FY2012) and A33 landslide warning zones (FY2018), joined to plots by point-in-polygon; N03 administrative boundaries (2023) for the map. `scripts/download_data.py` fetches everything from https://nlftp.mlit.go.jp/ksj/.

## Linking plots across years

A plot's `plot_id` is its official number in the first year it appears, plus that year (for example `43404-000-002/2019`). Each later record inherits the id of the record its previous-year number points to.

- **Continuing plots** link every year.
- **Renumbered plots** (five in 2025) keep their history through the previous-number field.
- **Replaced sites** (site status 4 or 5) have a blank previous number, so they start a new history, even when MLIT reuses the old plot number.

Matching on rounded coordinates instead lost 63 of 1,712 transitions. Many were at the 2024 change from the JGD2000 to the JGD2011 datum, which moves coordinates in the fifth decimal place.

The July survey is linked the same way; its plot ids are prefixed `L02:`.

**Validation.** The computed change matches MLIT's published change rate exactly in every case: 1,216 January-survey transitions from 2022 and 2,734 July-survey transitions from 2021. Matching uses MLIT's rounding: one decimal place, halves away from zero. For L02 this check also verifies the column mapping. MLIT's L01 price history agrees with the yearly files wherever the two overlap. The checks are in `tests/test_mlit_consistency.py`.

## Sites valued by both surveys

About 25 physical sites are valued in both January and July (MLIT's "common points", 共通地点). An L02 plot is paired with the L01 plot within 10 m of it in the same calendar year, and the pairs are checked against MLIT's own common-point flag.

- **Why 10 m.** Flagged pairs agree to 0.01 m in 2019 but drift up to ~7 m apart later, as each survey revises its coordinates. No L02 record lies between 10 and 50 m from an L01 site, so the threshold is unambiguous.
- **What pairing changes.** A paired L02 plot gets the L01 plot's `site_id`.
- **Six unflagged pairs** are new 2023–24 L02 sites in the fab towns, placed at existing L01 sites.

Pairs are written to `data/processed/common_sites.csv`.

## Timing of the announcement

TSMC confirmed the Kumamoto fab in October 2021. January-survey valuations are as of January 1 and July-survey valuations as of July 1, so in both surveys 2021 is the last valuation before the announcement and 2022 the first after. The first version of this project labelled 2021 as "year 0 after the announcement". It is now treated as pre-announcement.

## Fab impact (`src/fab_impact.py`)

**Panel.** Annual % change per plot, both surveys:

- **January survey, 2014–2026.** Changes up to 2019 come from the price history in the 2021 release, kept only for years in which the site was a benchmark. Later years come from the yearly files.
- **July survey, 2016–2026,** from its yearly files (2015 is the first base year).

That gives 812 plots (787 sites) and 8,078 plot-years. Distance to the fab is the straight-line distance from each plot's latest location to the JASM site, computed in JGD2011 / Japan Plane Rectangular CS II (EPSG:6670).

**Difference-in-differences.**

    pct_change[i,t] = β · near[i] · post[t] + plot[i] + survey_year[s,t] + ε

- **`near`:** within 10 km of the fab. **`post`:** valuation year 2022 or later.
- **Plot fixed effects** absorb each plot's level of growth.
- **Survey-by-year fixed effects** absorb each survey's prefecture-wide market swings. January and July markets differ, so each survey gets its own year effects. Each survey's first year is dropped, so the year dummies are not collinear with the plot dummies.
- **β** is the average extra annual growth near the fab after the announcement.
- **Standard errors** are clustered by site, so a site valued in both surveys is one cluster.

The model is run on both surveys pooled (main) and on each survey alone. Robustness runs:

- 5 km and 15 km cut-offs
- a pre-period starting in 2018 (after the earthquake dip)
- a sample without Kumamoto City
- a sample without the 2021 valuations

**Distance rings.** The same model with separate `ring × post` terms for 0–5, 5–10, 10–15, 15–20 and 20–30 km. Plots beyond 30 km are the comparison group.

**Event study.** The same model with `near × year` dummies, with 2021 as the reference year. It is run on both surveys pooled, on the July survey alone, and pooled without Kumamoto City. It shows whether near and far plots moved together before the announcement:

- **2014–2015:** the estimates are near zero (January survey only).
- **2016–2020:** the estimates sit at about −1 pp in every year and every sample. Near plots ran behind for the five years after the April 2016 earthquakes, whose epicentre was next to the fab towns. The July survey's 2016 valuation (July 2016) already falls after them; the January survey's first post-earthquake valuation is 2017.
- **2021:** near plots are back level.

The earlier January-only version attributed this pattern to Kumamoto City in the comparison group. With the July survey added, the dip persists without Kumamoto City, so the earthquake is the better explanation.

Since the pre-period average includes these depressed years, β is conservative relative to a pre-earthquake baseline. Dropping 2021 gives +3.9 pp against +3.7 pp.

## Model evaluation (`src/train.py`)

**Rows:** January-survey post-announcement valuations (2022–2026) with a previous-year change, 1,196 in total. Models are trained on the January survey only.

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
| July-survey plots, same years | train on all January-survey rows, test on 2,122 July-survey rows (2022–2026), excluding sites valued by both surveys |
| July-survey plots, forecast | train on January-survey rows to 2024, test on July-survey rows for 2025–2026 |

The external tests use different plots, different appraisers and a valuation date six months later. Two caveats:

- **Municipality-year mean** is half a year out of phase there.
- **Planning area:** 40% of July-survey rows lie outside any planning area, a category the January survey never contains.

**Spatial autocorrelation.** Moran's I uses an 8-nearest-neighbour weights matrix and is computed separately for each year. Pooling years would make every plot its own nearest neighbour.

**Forecast errors (`src/evaluate.py`).** Permutation importance is computed on the 2025–2026 forecasts (20 shuffles, R² scoring). Unlike impurity importance from training, it measures what a feature contributes on years the model has not seen.

## Things tried that did not help

- **Modelling the deviation from persistence.** Predicting this year's change minus last year's, then adding last year's back, scored R² 0.46 on the forecast split, worse than both plain models. The model learned the 2022–2024 acceleration and could not anticipate the 2026 slowdown.
- **Spatial models in an earlier version:** a Gaussian process, a mixed-effects model with municipality intercepts, and spatial lag and error models. None beat Random Forest on held-out areas, and they were removed to keep the comparison focused.
