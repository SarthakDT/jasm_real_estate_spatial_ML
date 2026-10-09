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
ANNOUNCEMENT_YEAR = 2021
YEARS = list(range(2019, 2027))
LAT_MIN, LAT_MAX = 32.0, 33.2
LON_MIN, LON_MAX = 130.0, 131.2

# WGS84 centroid of the JASM (TSMC/Sony/Denso JV) semiconductor fab site,
# Kikuyo-cho, Kumamoto.
JASM_COORDINATES = (32.8715, 130.8250)

# MLIT GeoJSON column names shifted twice across the 2019-2026 data releases.
# Verified by inspecting the first feature of each year's file.
_SCHEMA: dict[tuple, dict] = {
    (2019, 2021): {"year": "L01_005", "price": "L01_006", "muni_code": "L01_021"},
    (2022, 2023): {"year": "L01_005", "price": "L01_006", "muni_code": "L01_022"},
    (2024, 2026): {"year": "L01_007", "price": "L01_008", "muni_code": "L01_001"},
}

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
    "latitude",
    "longitude",
    "past_year_price",
    "years_since_announcement",
    "distance_to_jasm_km",
    "spatial_decay",
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
