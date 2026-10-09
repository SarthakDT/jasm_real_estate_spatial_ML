"""
Stage 2 — Geospatial Feature Engineering
JASM Spatial ML  |  Kumamoto MLIT Chika-Kouji land-price dataset

Reads data/processed/stage1.csv, converts it to a GeoDataFrame, engineers
distance/decay/target features (vectorized in a projected CRS) plus static
hazard/land-use features joined from NLNI layers, validates the expanded
schema, and writes data/processed/stage2.csv (flat) and stage2.geoparquet
(with geometry, for downstream spatial modules).
"""

import os

import pandas as pd

from src import config, geo_utils
from src.data_sources.hazards import join_hazards_to_points
from src.logging_utils import get_logger, success, status

log = get_logger(__name__)

def _success(msg, *args): success(log, msg, *args)
def _status(msg, *args):  status(log, msg, *args)


STAGE1_COLUMNS = {
    "point_id", "municipality", "latitude", "longitude",
    "past_year_price", "current_year_price", "valuation_year", "post_announcement",
}


# ── Step A: Load & verify Stage 1 input ──────────────────────────────────────
def _load_stage1() -> pd.DataFrame:
    if not os.path.isfile(config.STAGE1_PATH):
        raise FileNotFoundError(
            f"Stage 1 output missing: {config.STAGE1_PATH}\n"
            "Run src/load_data.py first to generate the cleaned dataset."
        )
    df = pd.read_csv(config.STAGE1_PATH, encoding="utf-8-sig")
    missing = STAGE1_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Stage 1 CSV is missing required columns: {missing}\n"
            "Re-run src/load_data.py to regenerate a valid stage1.csv."
        )
    log.info("Loaded %d rows from %s.", len(df), config.STAGE1_PATH)
    return df


# ── Step B: Vectorized projected-CRS distance to JASM centroid ──────────────
def _compute_distance(gdf):
    jasm = geo_utils.to_gdf(
        pd.DataFrame({"latitude": [config.JASM_COORDINATES[0]], "longitude": [config.JASM_COORDINATES[1]]})
    )
    jasm_projected = geo_utils.project(jasm).geometry.iloc[0]
    gdf["distance_to_jasm_km"] = gdf.geometry.distance(jasm_projected) / 1000.0
    gdf["distance_to_jasm_km"] = gdf["distance_to_jasm_km"].astype("float64")
    return gdf


# ── Step C: Non-linear inverse spatial decay ─────────────────────────────────
def _compute_decay(df):
    df["spatial_decay"] = 1.0 / (df["distance_to_jasm_km"] + 1.0)
    df["spatial_decay"] = df["spatial_decay"].astype("float64")
    return df


# ── Step D: Target variable ───────────────────────────────────────────────────
def _compute_target(df):
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

    if df["flood_risk_class"].isna().any():
        errors.append("flood_risk_class contains NaN values.")
    if df["in_landslide_zone"].isna().any():
        errors.append("in_landslide_zone contains NaN values.")

    if errors:
        raise ValueError(
            "Stage 2 feature validation failed:\n  • " + "\n  • ".join(errors)
        )

    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run() -> pd.DataFrame:
    log.info("Building features.")

    df = _load_stage1()
    gdf = geo_utils.to_gdf(df)
    gdf = geo_utils.project(gdf)

    gdf = _compute_distance(gdf)
    gdf = _compute_decay(gdf)
    gdf = _compute_target(gdf)

    gdf = join_hazards_to_points(gdf)
    _success("Joined NLNI hazard/land-use layers (flood, landslide, urban planning zone).")

    gdf = _validate(gdf)

    _status("Wrote %s.", config.STAGE2_PATH)

    # Save: flat CSV (backward-compatible, no geometry) + geoparquet (with geometry)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    flat_df = pd.DataFrame(gdf.drop(columns="geometry"))
    flat_df.to_csv(config.STAGE2_PATH, index=False, encoding="utf-8-sig")
    gdf.to_parquet(config.STAGE2_GEOPARQUET_PATH, index=False)

    # Summary diagnostics
    print()
    print(f"  Rows              : {len(flat_df)}")
    print(f"  Columns           : {list(flat_df.columns)}")
    print(f"  Distance range    : [{flat_df['distance_to_jasm_km'].min():.2f} km, {flat_df['distance_to_jasm_km'].max():.2f} km]")
    print(f"  Decay range       : [{flat_df['spatial_decay'].min():.6f}, {flat_df['spatial_decay'].max():.6f}]")
    print(f"  Target pct change : [{flat_df['target_price_pct_change'].min():.2f}%, {flat_df['target_price_pct_change'].max():.2f}%]")
    print(f"  Flood risk class  : {flat_df['flood_risk_class'].value_counts().to_dict()}")
    print(f"  In landslide zone : {flat_df['in_landslide_zone'].sum()} of {len(flat_df)} points")
    print()
    print(flat_df.dtypes.to_string())

    return flat_df


if __name__ == "__main__":
    run()
