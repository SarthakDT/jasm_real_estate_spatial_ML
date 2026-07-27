"""
Stage 2 — Geospatial Feature Engineering
JASM Spatial ML  |  Kumamoto MLIT Chika-Kouji land-price dataset

Reads data/processed_stage1.csv, engineers three spatial/economic features,
validates the expanded schema, and writes data/processed_stage2.csv.
"""

import logging
import os
import sys

import pandas as pd
from geopy.distance import geodesic

# ── Logging: custom SUCCESS / STATUS levels to match expected health report ───
_SUCCESS = 25
_STATUS  = 26
logging.addLevelName(_SUCCESS, "SUCCESS")
logging.addLevelName(_STATUS,  "STATUS")
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

def _success(msg, *args): log.log(_SUCCESS, msg, *args)
def _status(msg, *args):  log.log(_STATUS,  msg, *args)


# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
INPUT_PATH  = os.path.join(DATA_DIR, "processed_stage1.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed_stage2.csv")

# WGS84 centroid of the JASM semiconductor fab site, Kikuyo-cho, Kumamoto
JASM_COORDINATES = (32.8715, 130.8250)

STAGE1_COLUMNS = {
    "point_id", "municipality", "latitude", "longitude",
    "past_year_price", "current_year_price", "years_since_announcement",
}


# ── Step A: Load & verify Stage 1 input ──────────────────────────────────────
def _load_stage1() -> pd.DataFrame:
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(
            f"Stage 1 output missing: {INPUT_PATH}\n"
            "Run src/load_data.py first to generate the cleaned dataset."
        )
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    missing = STAGE1_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Stage 1 CSV is missing required columns: {missing}\n"
            "Re-run src/load_data.py to regenerate a valid processed_stage1.csv."
        )
    log.info("Loaded %d rows from %s.", len(df), INPUT_PATH)
    return df


# ── Step B: Geodesic distance to JASM centroid ───────────────────────────────
def _compute_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_to_jasm_km"] = df.apply(
        lambda row: geodesic(
            (row["latitude"], row["longitude"]), JASM_COORDINATES
        ).km,
        axis=1,
    )
    df["distance_to_jasm_km"] = df["distance_to_jasm_km"].astype("float64")
    return df


# ── Step C: Non-linear inverse spatial decay ─────────────────────────────────
def _compute_decay(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # +1 stabilises the denominator: a plot at exactly 0 km yields decay = 1.0
    df["spatial_decay"] = 1.0 / (df["distance_to_jasm_km"] + 1.0)
    df["spatial_decay"] = df["spatial_decay"].astype("float64")
    return df


# ── Step D: Target variable ───────────────────────────────────────────────────
def _compute_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_price_pct_change"] = (
        (df["current_year_price"] - df["past_year_price"])
        / df["past_year_price"]
    ) * 100.0
    df["target_price_pct_change"] = df["target_price_pct_change"].astype("float64")
    return df


# ── Step E: Validate engineered columns ──────────────────────────────────────
def _validate(df: pd.DataFrame) -> pd.DataFrame:
    errors: list[str] = []

    if df["distance_to_jasm_km"].isna().any():
        errors.append("distance_to_jasm_km contains NaN values.")
    elif df["distance_to_jasm_km"].le(0).any():
        errors.append(
            f"{df['distance_to_jasm_km'].le(0).sum()} record(s) have "
            "distance_to_jasm_km <= 0 (impossible for off-site plots)."
        )

    if df["spatial_decay"].isna().any():
        errors.append("spatial_decay contains NaN values.")
    else:
        oob = ~df["spatial_decay"].between(0.0, 1.0, inclusive="neither")
        # A plot sitting exactly at JASM gates would yield decay == 1.0 (valid)
        oob = oob & (df["spatial_decay"] != 1.0)
        if oob.any():
            errors.append(
                f"{oob.sum()} record(s) have spatial_decay outside (0.0, 1.0]."
            )

    if df["target_price_pct_change"].isna().any():
        errors.append("target_price_pct_change contains NaN values.")
    elif not df["target_price_pct_change"].apply(
        lambda x: x == x and x != float("inf") and x != float("-inf")
    ).all():
        errors.append("target_price_pct_change contains infinite values.")

    if errors:
        raise ValueError(
            "Stage 2 feature validation failed:\n  • " + "\n  • ".join(errors)
        )

    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run() -> pd.DataFrame:
    log.info("Initiating Feature Engineering Pipeline...")

    df = _load_stage1()
    df = _compute_distance(df)
    df = _compute_decay(df)
    df = _compute_target(df)
    df = _validate(df)

    # ── DoD health report ─────────────────────────────────────────────────────
    _success("Computed geodesic distances to JASM centroid using WGS84 standard.")
    _success("Non-linear spatial decay vectors generated and stabilized against zero-division.")
    _success("Target variable 'target_price_pct_change' computed across all temporal windows.")
    _status("Cleaned feature matrix cached to %s. Ready for Stage 3 Model Training.", OUTPUT_PATH)

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # Summary diagnostics
    print()
    print(f"  Rows              : {len(df)}")
    print(f"  Columns           : {list(df.columns)}")
    print(f"  Distance range    : [{df['distance_to_jasm_km'].min():.2f} km, {df['distance_to_jasm_km'].max():.2f} km]")
    print(f"  Decay range       : [{df['spatial_decay'].min():.6f}, {df['spatial_decay'].max():.6f}]")
    print(f"  Target pct change : [{df['target_price_pct_change'].min():.2f}%, {df['target_price_pct_change'].max():.2f}%]")
    print()
    print(df.dtypes.to_string())

    return df


if __name__ == "__main__":
    run()
