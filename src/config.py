"""
Central configuration for the JASM land-price pipeline.

All paths, MLIT schema mappings, municipality metadata, and shared
hyperparameters live here so that ingestion, feature engineering, training,
evaluation, and the dashboard stay in sync instead of each redefining its
own copy.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")

CHIKA_KOUJI_RAW_DIR = os.path.join(RAW_DIR, "chika_kouji")

STAGE1_PATH = os.path.join(PROCESSED_DIR, "stage1.csv")
STAGE2_PATH = os.path.join(PROCESSED_DIR, "stage2.csv")
STAGE2_GEOPARQUET_PATH = os.path.join(PROCESSED_DIR, "stage2.geoparquet")
ANOMALY_PATH = os.path.join(PROCESSED_DIR, "evaluation_anomalies.csv")
SPATIAL_DIAGNOSTICS_PATH = os.path.join(DATA_DIR, "spatial_diagnostics.json")
BACKTEST_PATH = os.path.join(DATA_DIR, "backtest_results.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "jasm_rf_model.pkl")
CHART_PATH = os.path.join(MODEL_DIR, "feature_importance.png")
MODEL_COMPARISON_PATH = os.path.join(MODEL_DIR, "model_comparison.csv")

# ── Coordinate reference systems ─────────────────────────────────────────────
WGS84 = "EPSG:4326"
# JGD2011 / Japan Plane Rectangular CS Zone II — covers Kumamoto Prefecture.
PROJECTED_CRS = "EPSG:6670"

# ── Kumamoto / JASM domain constants ─────────────────────────────────────────
# TSMC confirmed its Kumamoto fab plan in October 2021. MLIT benchmark prices
# are valuations as of January 1, so the 2021 values predate the announcement
# and 2022 is the first valuation that could reflect it.
ANNOUNCEMENT_DATE = "2021-10"
FIRST_POST_VALUATION_YEAR = 2022
YEARS = list(range(2019, 2027))
LAT_MIN, LAT_MAX = 32.0, 33.2
LON_MIN, LON_MAX = 130.0, 131.2

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
PLANNING_AREAS = {"市街化": "urbanization", "調区": "control", "非線引": "undivided"}

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

RF_PARAMS = dict(n_estimators=100, max_depth=7, min_samples_split=5, random_state=42)
TEST_SIZE = 0.25
RANDOM_STATE = 42
ANOMALY_PCT = 0.95


# ── API credentials ────────────────────────────────────────────────────────
def get_reinfolib_key() -> str:
    key = os.environ.get("REINFOLIB_API_KEY")
    if not key:
        raise RuntimeError(
            "REINFOLIB_API_KEY is not set. Register for a free key at "
            "https://www.reinfolib.mlit.go.jp/api/request and add it to .env "
            "(copy .env.example)."
        )
    return key


def get_estat_app_id() -> str:
    app_id = os.environ.get("ESTAT_APP_ID")
    if not app_id:
        raise RuntimeError(
            "ESTAT_APP_ID is not set. Register for a free application ID at "
            "https://www.e-stat.go.jp/api/ and add it to .env (copy .env.example)."
        )
    return app_id
