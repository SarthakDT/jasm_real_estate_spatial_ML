"""
Central configuration for the JASM land-price pipeline.

All paths, MLIT schema mappings, municipality metadata, and shared
hyperparameters live here so that ingestion, feature engineering, training,
evaluation, and the dashboard stay in sync instead of each redefining its
own copy.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")

CHIKA_KOUJI_RAW_DIR = os.path.join(RAW_DIR, "chika_kouji")
PREFECTURAL_SURVEY_RAW_DIR = os.path.join(RAW_DIR, "prefectural_survey")

# L01 (January survey) keeps the original file names; L02 (July survey) gets its own.
STAGE1_PATH = os.path.join(PROCESSED_DIR, "stage1.csv")
STAGE2_PATH = os.path.join(PROCESSED_DIR, "stage2.csv")
STAGE2_GEOPARQUET_PATH = os.path.join(PROCESSED_DIR, "stage2.geoparquet")
STAGE1_L02_PATH = os.path.join(PROCESSED_DIR, "stage1_l02.csv")
STAGE2_L02_PATH = os.path.join(PROCESSED_DIR, "stage2_l02.csv")
COMMON_SITES_PATH = os.path.join(PROCESSED_DIR, "common_sites.csv")
SPATIAL_DIAGNOSTICS_PATH = os.path.join(ANALYSIS_DIR, "spatial_diagnostics.json")
FORECAST_PREDICTIONS_PATH = os.path.join(ANALYSIS_DIR, "forecast_predictions.csv")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
FORECAST_MODEL_PATH = os.path.join(MODEL_DIR, "forecast_model.pkl")
BEST_MODEL_NAME_PATH = os.path.join(MODEL_DIR, "best_model_name.txt")
MODEL_COMPARISON_PATH = os.path.join(ANALYSIS_DIR, "model_comparison.csv")

# Figures embedded in the README are committed, unlike other outputs.
FIGURES_DIR = os.path.join(BASE_DIR, "docs", "figures")
CHART_PATH = os.path.join(FIGURES_DIR, "feature_importance.png")

# ── Coordinate reference systems ─────────────────────────────────────────────
WGS84 = "EPSG:4326"
# JGD2011 / Japan Plane Rectangular CS Zone II — covers Kumamoto Prefecture.
PROJECTED_CRS = "EPSG:6670"

# ── Kumamoto / JASM domain constants ─────────────────────────────────────────
# TSMC confirmed its Kumamoto fab plan in October 2021. Both surveys value
# land on a fixed date (January 1 or July 1), so 2021 valuations predate the
# announcement and 2022 is the first that could reflect it.
FIRST_POST_VALUATION_YEAR = 2022   # holds for both surveys: L02's July 2021 valuation is also pre-announcement
YEARS = list(range(2019, 2027))    # L01 (Chika-Kouji, valued January 1)
L02_YEARS = list(range(2015, 2027))  # L02 (prefectural survey, valued July 1); MLIT publishes no 2014 file
# Kumamoto Prefecture's extent from MLIT's N03 boundaries (129.94-131.33 E,
# 32.10-33.20 N), rounded outward. The prefectural survey reaches the
# western Amakusa islands and the eastern Aso villages.
LAT_MIN, LAT_MAX = 32.0, 33.25
LON_MIN, LON_MAX = 129.9, 131.4

# WGS84 centroid of the JASM (TSMC/Sony/Denso JV) semiconductor fab site,
# Kikuyo-cho, Kumamoto.
JASM_COORDINATES = (32.8715, 130.8250)

# MLIT L01 GeoJSON property keys per release era. The column layout changed in
# 2022 (published change rate added) and again in 2024 (municipality code moved
# to the front and previous-year municipality added). Positions were checked
# against the element names in each era's GML file and against value ranges in
# all eight years.
_SCHEMA: dict[tuple, dict] = {
    (2019, 2021): {
        "use_code": "L01_001", "serial": "L01_002",
        "prev_use_code": "L01_003", "prev_serial": "L01_004",
        "year": "L01_005", "price": "L01_006", "site_status": "L01_007",
        "muni_code": "L01_021", "lot_area_m2": "L01_024", "current_use": "L01_025",
        "road_width_m": "L01_038", "station_name": "L01_045", "station_distance_m": "L01_046",
        "zoning": "L01_047", "planning_area": "L01_049", "bcr": "L01_052", "far": "L01_053",
    },
    (2022, 2023): {
        "use_code": "L01_001", "serial": "L01_002",
        "prev_use_code": "L01_003", "prev_serial": "L01_004",
        "year": "L01_005", "price": "L01_006", "published_change_pct": "L01_007",
        "site_status": "L01_008",
        "muni_code": "L01_022", "lot_area_m2": "L01_026", "current_use": "L01_027",
        "road_width_m": "L01_041", "station_name": "L01_048", "station_distance_m": "L01_049",
        "zoning": "L01_050", "planning_area": "L01_052", "bcr": "L01_056", "far": "L01_057",
    },
    (2024, 2026): {
        "muni_code": "L01_001", "use_code": "L01_002", "serial": "L01_003",
        "prev_muni_code": "L01_004", "prev_use_code": "L01_005", "prev_serial": "L01_006",
        "year": "L01_007", "price": "L01_008", "published_change_pct": "L01_009",
        "site_status": "L01_010",
        "lot_area_m2": "L01_027", "current_use": "L01_028",
        "road_width_m": "L01_042", "station_name": "L01_048", "station_distance_m": "L01_050",
        "zoning": "L01_051", "planning_area": "L01_053", "bcr": "L01_057", "far": "L01_058",
    },
}

# MLIT L02 (prefectural land price survey, 都道府県地価調査) property keys per
# era. 2015-2016 come as shapefiles without zoning or planning-area fields; the
# change rate appears in 2021; the 2026 release shifts the later columns by
# one. Positions were found by pairing 120 records per year between the GML
# (which names each field) and the GeoJSON/shapefile, then checked against
# value ranges in all twelve years.
_L02_COMMON = {
    "use_code": "L02_001", "serial": "L02_002",
    "prev_use_code": "L02_003", "prev_serial": "L02_004",
    "year": "L02_005", "price": "L02_006",
}
_L02_SCHEMA: dict[tuple, dict] = {
    (2015, 2016): {
        **_L02_COMMON, "site_status": "L02_007", "muni_code": "L02_017",
        "lot_area_m2": "L02_020", "current_use": "L02_021", "road_width_m": "L02_034",
        "station_name": "L02_039", "station_distance_m": "L02_040",
        "bcr": "L02_042", "far": "L02_043", "common_point": "L02_044",
    },
    (2017, 2020): {
        **_L02_COMMON, "site_status": "L02_007", "muni_code": "L02_021",
        "lot_area_m2": "L02_024", "current_use": "L02_025", "road_width_m": "L02_038",
        "station_name": "L02_043", "station_distance_m": "L02_044", "zoning": "L02_045",
        "planning_area": "L02_047", "bcr": "L02_050", "far": "L02_051", "common_point": "L02_052",
    },
    (2021, 2025): {
        **_L02_COMMON, "published_change_pct": "L02_007", "site_status": "L02_008",
        "muni_code": "L02_020", "lot_area_m2": "L02_024", "current_use": "L02_025",
        "road_width_m": "L02_039", "station_name": "L02_044", "station_distance_m": "L02_045",
        "zoning": "L02_046", "planning_area": "L02_048", "bcr": "L02_050", "far": "L02_051",
        "common_point": "L02_053",
    },
    (2026, 2026): {
        **_L02_COMMON, "published_change_pct": "L02_007", "site_status": "L02_008",
        "muni_code": "L02_020", "lot_area_m2": "L02_024", "current_use": "L02_025",
        "road_width_m": "L02_039", "station_name": "L02_045", "station_distance_m": "L02_046",
        "zoning": "L02_047", "planning_area": "L02_049", "bcr": "L02_051", "far": "L02_052",
        "common_point": "L02_054",
    },
}
# Forest plots (林地): numbered prefecture-wide under code 43900 with land-use
# code 020. L01 has none, and they are not land the fab would plausibly move,
# so they are dropped at load.
L02_EXCLUDED_USE_CODES = ("020",)

# An L02 site within this distance of an L01 site in the same year is the
# same physical plot valued by both surveys (a "common point", 共通地点).
# The two surveys' coordinates for MLIT-flagged common points agree to 0.01 m
# in 2019 but drift up to ~7 m apart from 2020 as each revises its geocoding;
# unrelated benchmark plots are far further apart than this.
COMMON_SITE_MAX_DISTANCE_M = 10.0

# Each record also carries the plot's price history: a "selected year" flag
# string (one character per year from 1983, "1" = the site was a benchmark that
# year) followed by one price column per year from 1983. Values are
# (flag column, property number of the 1983 price) per era.
_HISTORY: dict[tuple, tuple] = {
    (2019, 2021): ("L01_055", 56),
    (2022, 2023): ("L01_060", 61),
    (2024, 2026): ("L01_061", 62),
}
HISTORY_FIRST_YEAR = 1983

# MLIT land-use category (用途区分) codes found in the Kumamoto files.
LAND_USE_CODES = {
    "000": "residential",
    "003": "prospective residential",  # 宅地見込地
    "005": "commercial",
    "009": "industrial",
}

# 都市計画区域区分: urbanization promotion area, urbanization control area,
# and planning areas with no such split.
PLANNING_AREAS = {"市街化": "urbanization", "調区": "control", "非線引": "undivided", "都計外": "outside"}

# JIS X 0402 municipality codes present in Kumamoto prefecture (43) dataset.
#
# Cross-verified 2026-08-05 against the official Kumamoto Prefecture municipal
# code list (pref.kumamoto.jp) and, for the JASM-fab municipality specifically,
# a point-in-polygon check of JASM_COORDINATES against NLNI N03 admin boundary
# polygons (data/raw/hazards/N03_43_GML/). The previous version of this dict
# had 11 of 24 codes mislabeled — most importantly 43404 (the fab's actual
# host municipality, Kikuyo-cho) was mislabeled "Mashiki-machi", and 43443
# (the real Mashiki-machi) was mislabeled "Nagomi-machi". Fixed here; the
# fix only changes string labels, not any model input/output.
MUNI_NAMES: dict[str, str] = {
    "43101": "Kumamoto-shi Chuo-ku",
    "43102": "Kumamoto-shi Higashi-ku",
    "43103": "Kumamoto-shi Nishi-ku",
    "43104": "Kumamoto-shi Minami-ku",
    "43105": "Kumamoto-shi Kita-ku",
    "43202": "Yatsushiro-shi",
    "43203": "Hitoyoshi-shi",
    "43204": "Arao-shi",
    "43205": "Minamata-shi",
    "43206": "Tamana-shi",
    "43208": "Yamaga-shi",
    "43210": "Kikuchi-shi",
    "43211": "Uto-shi",
    "43213": "Uki-shi",
    "43214": "Aso-shi",
    "43215": "Amakusa-shi",
    "43216": "Koshi-shi",
    "43368": "Nagasu-machi",
    "43403": "Ozu-machi",
    "43404": "Kikuyo-cho",
    "43441": "Mifune-machi",
    "43442": "Kashima-machi",
    "43443": "Mashiki-machi",
    "43482": "Ashikita-machi",
    # Added for the L02 survey, which covers every municipality. Japanese
    # names from MLIT's N03 administrative boundaries (2023-01-01).
    "43212": "Kamiamakusa-shi",
    "43348": "Misato-machi",
    "43364": "Gyokuto-machi",
    "43367": "Nankan-machi",
    "43369": "Nagomi-machi",
    "43423": "Minamioguni-machi",
    "43424": "Oguni-machi",
    "43425": "Ubuyama-mura",
    "43428": "Takamori-machi",
    "43432": "Nishihara-mura",
    "43433": "Minamiaso-mura",
    "43444": "Kosa-machi",
    "43447": "Yamato-cho",
    "43468": "Hikawa-cho",
    "43484": "Tsunagi-machi",
    "43501": "Nishiki-machi",
    "43505": "Taragi-machi",
    "43506": "Yunomae-machi",
    "43507": "Mizukami-mura",
    "43510": "Sagara-mura",
    "43511": "Itsuki-mura",
    "43512": "Yamae-mura",
    "43513": "Kuma-mura",
    "43514": "Asagiri-cho",
    "43531": "Reihoku-machi",
}
_VALID_MUNIS = set(MUNI_NAMES.values())

# ── Model / training constants ────────────────────────────────────────────────
FEATURE_COLS = [
    "prev_year_pct_change",
    "distance_to_jasm_km",
    "latitude",
    "longitude",
    "past_year_price",
    "valuation_year",
    "lot_area_m2",
    "station_distance_m",
    "road_width_m",
    "far",
    "land_use_code",
    "zoning_group",
    "planning_area_code",
    "flood_risk_class",
    "in_landslide_zone",
]
TARGET_COL = "target_price_pct_change"

RF_PARAMS = dict(n_estimators=300, max_depth=7, min_samples_split=5, random_state=42, n_jobs=-1)
RANDOM_STATE = 42
# The forecast split trains on valuations up to this year and predicts the rest.
FORECAST_TRAIN_MAX_YEAR = 2024
