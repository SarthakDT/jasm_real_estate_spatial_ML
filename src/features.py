"""
Stage 2 — Feature engineering
MLIT Chika-Kouji benchmark land prices, Kumamoto Prefecture

Reads data/processed/stage1.csv and adds:
  - the target: year-over-year % price change
  - prev_year_pct_change: the same plot's change the year before (the lag)
  - distance to the JASM fab
  - numeric codes for MLIT's land-use, zoning and planning-area categories
  - flood and landslide hazard flags from NLNI polygons

Writes data/processed/stage2.csv (flat) and stage2.geoparquet (with geometry).
"""

import os

import numpy as np
import pandas as pd

from src import config, geo_utils
from src.data_sources.hazards import join_hazards_to_points
from src.logging_utils import get_logger, status

log = get_logger(__name__)


def _status(msg, *args): status(log, msg, *args)


STAGE1_COLUMNS = {
    "point_id", "plot_id", "municipality", "valuation_year", "post_announcement",
    "latitude", "longitude", "past_year_price", "current_year_price",
    "land_use", "lot_area_m2", "road_width_m", "station_distance_m",
    "zoning", "planning_area", "far",
}

# Ordinal codes for tree models. Zoning (用途地域) is grouped from MLIT's 13
# zone abbreviations into broad classes ordered roughly by permitted density.
LAND_USE_CODE = {"residential": 0, "prospective residential": 1, "commercial": 2, "industrial": 3}
PLANNING_AREA_CODE = {"urbanization": 0, "undivided": 1, "control": 2}
ZONING_GROUP = {
    "none": 0,
    "1低専": 1, "2低専": 1, "田園住": 1,   # low-rise residential
    "1中専": 2, "2中専": 2,                 # mid-rise residential
    "1住居": 3, "2住居": 3, "準住居": 3,   # mixed residential
    "近商": 4, "商業": 4,                   # commercial
    "準工": 5, "工業": 5, "工専": 5,       # industrial
}


def _load_stage1() -> pd.DataFrame:
    if not os.path.isfile(config.STAGE1_PATH):
        raise FileNotFoundError(f"Stage 1 output missing: {config.STAGE1_PATH}. Run stage 1 first.")
    df = pd.read_csv(config.STAGE1_PATH, encoding="utf-8-sig")
    missing = STAGE1_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"stage1.csv is missing columns: {missing}. Re-run stage 1.")
    log.info("Loaded %d rows from %s.", len(df), config.STAGE1_PATH)
    return df


def _add_target_and_lag(df: pd.DataFrame) -> pd.DataFrame:
    df[config.TARGET_COL] = (df["current_year_price"] - df["past_year_price"]) / df["past_year_price"] * 100.0
    prev = df[["plot_id", "valuation_year", config.TARGET_COL]].rename(
        columns={config.TARGET_COL: "prev_year_pct_change"}
    )
    prev["valuation_year"] += 1
    return df.merge(prev, on=["plot_id", "valuation_year"], how="left")


def _encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    df["land_use_code"] = df["land_use"].map(LAND_USE_CODE)
    df["planning_area_code"] = df["planning_area"].map(PLANNING_AREA_CODE)
    df["zoning_group"] = df["zoning"].map(ZONING_GROUP)
    for col, source in (("land_use_code", "land_use"), ("planning_area_code", "planning_area"), ("zoning_group", "zoning")):
        unknown = df[col].isna()
        if unknown.any():
            raise ValueError(f"Unmapped {source} values: {df.loc[unknown, source].unique().tolist()}")
        df[col] = df[col].astype("int64")
    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    errors: list[str] = []
    if df["distance_to_jasm_km"].isna().any() or df["distance_to_jasm_km"].le(0).any():
        errors.append("distance_to_jasm_km has missing or non-positive values.")
    if not np.isfinite(df[config.TARGET_COL]).all():
        errors.append(f"{config.TARGET_COL} has missing or infinite values.")
    lag_missing = df["post_announcement"] & df["prev_year_pct_change"].isna()
    if lag_missing.sum() > 0.05 * df["post_announcement"].sum():
        errors.append(f"{lag_missing.sum()} post-announcement rows have no previous-year change.")
    for col in ("flood_risk_class", "in_landslide_zone"):
        if df[col].isna().any():
            errors.append(f"{col} has missing values.")
    if errors:
        raise ValueError("Stage 2 validation failed:\n  • " + "\n  • ".join(errors))
    return df


def run() -> pd.DataFrame:
    log.info("Building features.")
    df = _load_stage1()
    df["distance_to_jasm_km"] = geo_utils.distance_to_jasm_km(df)
    df = _add_target_and_lag(df)
    df = _encode_categories(df)

    gdf = join_hazards_to_points(geo_utils.to_gdf(df))
    df = _validate(pd.DataFrame(gdf.drop(columns="geometry")))

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    df.to_csv(config.STAGE2_PATH, index=False, encoding="utf-8-sig")
    gdf[df.columns.tolist() + ["geometry"]].to_parquet(config.STAGE2_GEOPARQUET_PATH, index=False)
    _status("Wrote %s.", config.STAGE2_PATH)

    model_rows = df[df["post_announcement"] & df["prev_year_pct_change"].notna()]
    print()
    print(f"  Rows                      : {len(df)} ({len(model_rows)} post-announcement rows with a lag)")
    print(f"  Distance to fab           : {df['distance_to_jasm_km'].min():.1f}-{df['distance_to_jasm_km'].max():.1f} km")
    print(f"  Annual % change           : {df[config.TARGET_COL].min():.1f} to {df[config.TARGET_COL].max():.1f}")
    print(f"  Flood risk class          : {df['flood_risk_class'].value_counts().sort_index().to_dict()}")
    print(f"  Plots in landslide zones  : {df.loc[df['in_landslide_zone'], 'plot_id'].nunique()}")
    return df


if __name__ == "__main__":
    run()
