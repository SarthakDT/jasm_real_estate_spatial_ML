"""
Stage 1 — Data Ingestion & Schema Validation
JASM Spatial ML  |  Kumamoto MLIT Chika-Kouji land-price dataset

Reads 8 years of MLIT GeoJSON (2019-2026), constructs consecutive-year
price-transition rows, enforces the Stage 1 data contract, and writes
data/processed/stage1.csv.
"""

import json
import os

import pandas as pd

from src import config
from src.data_sources.base import validate_kumamoto_bbox
from src.logging_utils import get_logger, success, status

log = get_logger(__name__)


def _success(msg, *args): success(log, msg, *args)
def _status(msg, *args):  status(log, msg, *args)


# ── Step A: Raw input verification ────────────────────────────────────────────
def _schema_for_year(year: int) -> dict:
    for (lo, hi), cols in config._SCHEMA.items():
        if lo <= year <= hi:
            return cols
    raise ValueError(f"No column schema defined for year {year}.")


def _geojson_path(year: int) -> str:
    short = str(year)[2:]
    path = os.path.join(config.CHIKA_KOUJI_RAW_DIR, f"L01-{short}_43_GML", f"L01-{short}_43.geojson")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"GeoJSON missing: {path}\n"
            "Run `python scripts/download_data.py` to fetch the MLIT archives."
        )
    return path


def _load_year(year: int) -> pd.DataFrame:
    """Parse one year's GeoJSON into a flat DataFrame with canonical columns."""
    path = _geojson_path(year)
    cols = _schema_for_year(year)

    with open(path, encoding="utf-8-sig") as fh:
        gj = json.load(fh)

    rows = []
    for feat in gj["features"]:
        p = feat["properties"]
        c = feat["geometry"]["coordinates"]   # [longitude, latitude]
        rows.append({
            "latitude":  float(c[1]),
            "longitude": float(c[0]),
            "year":      p.get(cols["year"]),
            "price":     p.get(cols["price"]),
            "muni_code": str(p.get(cols["muni_code"], "")).strip(),
        })

    df = pd.DataFrame(rows)

    # Step B: strip ¥-signs, commas, whitespace before numeric cast
    df["price"] = (
        df["price"].astype(str).str.replace(r"[^\d]", "", regex=True)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["year"]  = pd.to_numeric(df["year"],  errors="coerce")

    df = df.dropna(subset=["price", "year"])
    df["price"] = df["price"].astype("int64")
    df["year"]  = df["year"].astype("int64")

    log.info("  %d: %d records loaded", year, len(df))
    return df


# ── Transition builder ────────────────────────────────────────────────────────
def _build_transitions(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pairs consecutive (t-1, t) year records for each benchmark plot.

    Transitions are kept for every consecutive year pair in the raw data,
    including pre-announcement pairs (years_since_announcement < 0) — the
    fab event-study analysis needs a pre-2021 baseline. ML training
    (src/train.py) explicitly filters years_since_announcement >= 0 before
    building its feature matrix, so model behavior is unchanged.
    """
    df_raw = df_raw.copy()
    # Stable cross-year plot identifier: municipality + rounded coordinates
    df_raw["plot_key"] = (
        df_raw["muni_code"]
        + "_" + df_raw["latitude"].round(5).astype(str)
        + "_" + df_raw["longitude"].round(5).astype(str)
    )

    rows = []
    for plot_key, grp in df_raw.groupby("plot_key", sort=False):
        grp = grp.sort_values("year").reset_index(drop=True)
        for i in range(1, len(grp)):
            prev, curr = grp.iloc[i - 1], grp.iloc[i]
            if int(curr["year"]) - int(prev["year"]) != 1:
                continue   # gap in data — skip non-consecutive pair
            ysa       = int(curr["year"]) - config.ANNOUNCEMENT_YEAR
            muni_code = str(curr["muni_code"])
            lat       = round(float(curr["latitude"]),  5)
            lon       = round(float(curr["longitude"]), 5)
            rows.append({
                "point_id":                  f"{muni_code}_{lat}_{lon}_Y{int(curr['year'])}",
                "municipality":              config.MUNI_NAMES.get(muni_code, muni_code),
                "latitude":                  lat,
                "longitude":                 lon,
                "past_year_price":           int(prev["price"]),
                "current_year_price":        int(curr["price"]),
                "years_since_announcement":  ysa,
            })

    return pd.DataFrame(rows)


# ── Step C: Null evacuation + full schema validation ─────────────────────────
def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces the Stage 1 data contract.
    Returns a cleaned, fully-typed DataFrame.
    Raises ValueError listing every violation found.
    """
    required = {
        "point_id", "municipality", "latitude", "longitude",
        "past_year_price", "current_year_price", "years_since_announcement",
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()

    # Coerce all numeric columns (catches stray strings/objects)
    df["latitude"]               = pd.to_numeric(df["latitude"],               errors="coerce")
    df["longitude"]              = pd.to_numeric(df["longitude"],               errors="coerce")
    df["past_year_price"]        = pd.to_numeric(df["past_year_price"],         errors="coerce")
    df["current_year_price"]     = pd.to_numeric(df["current_year_price"],      errors="coerce")
    df["years_since_announcement"] = pd.to_numeric(
        df["years_since_announcement"], errors="coerce"
    )

    # Null evacuation: drop rows where either target price is missing
    n_before = len(df)
    df = df.dropna(subset=["current_year_price", "past_year_price"])
    n_dropped = n_before - len(df)
    if n_dropped:
        log.warning(
            "Dropped %d row(s) with null price values "
            "(likely rural plots that skipped an appraisal cycle).",
            n_dropped,
        )

    # Impute auxiliary NaNs with the local-municipality historical median.
    # A global fallback handles the rare case where an entire municipality is null.
    for col in ("latitude", "longitude", "years_since_announcement"):
        if df[col].isna().any():
            group_median = df.groupby("municipality")[col].transform("median")
            df[col] = df[col].fillna(group_median).fillna(df[col].median())

    # Lock in final data types
    df["latitude"]               = df["latitude"].astype("float64")
    df["longitude"]              = df["longitude"].astype("float64")
    df["past_year_price"]        = df["past_year_price"].astype("int64")
    df["current_year_price"]     = df["current_year_price"].astype("int64")
    df["years_since_announcement"] = df["years_since_announcement"].astype("int64")
    df["point_id"]               = df["point_id"].astype(str)
    df["municipality"]           = df["municipality"].astype(str)

    # ── Rule checks ───────────────────────────────────────────────────────────
    errors: list[str] = []

    dup_count = df["point_id"].duplicated().sum()
    if dup_count:
        errors.append(f"{dup_count} duplicate point_id value(s) detected.")

    bad_muni = ~df["municipality"].isin(config._VALID_MUNIS)
    if bad_muni.any():
        errors.append(
            f"Unrecognised municipality values: "
            f"{df.loc[bad_muni, 'municipality'].unique().tolist()}"
        )

    # CRS audit: points outside the WGS84 Kumamoto bounding box indicate
    # possible Tokyo Datum contamination, which causes ~400-450 m positional drift.
    oob = validate_kumamoto_bbox(df)
    if oob.any():
        errors.append(
            f"{oob.sum()} record(s) fall outside Kumamoto WGS84 bounds "
            f"(lat {config.LAT_MIN}–{config.LAT_MAX}, lon {config.LON_MIN}–{config.LON_MAX}). "
            "Suspected Tokyo Datum contamination (~400–450 m positional drift)."
        )

    if df["past_year_price"].le(0).any():
        errors.append("past_year_price contains values ≤ 0.")
    if df["current_year_price"].le(0).any():
        errors.append("current_year_price contains values ≤ 0.")

    # Range widened to include pre-announcement pairs (as low as -2, since the
    # earliest raw year is 2019 and ANNOUNCEMENT_YEAR is 2021) — see
    # _build_transitions() docstring.
    bad_ysa = ~df["years_since_announcement"].between(-2, 6)
    if bad_ysa.any():
        errors.append(
            f"years_since_announcement out of range [-2, 6]: "
            f"{df.loc[bad_ysa, 'years_since_announcement'].unique().tolist()}"
        )

    if errors:
        raise ValueError(
            "Stage 1 schema validation failed:\n  • " + "\n  • ".join(errors)
        )

    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run() -> pd.DataFrame:
    log.info("Loading MLIT Chika-Kouji files for %d-%d.", config.YEARS[0], config.YEARS[-1])

    # Load all 8 years
    frames = [_load_year(y) for y in config.YEARS]
    df_raw = pd.concat(frames, ignore_index=True)
    log.info("Combined raw dataset: %d records across %d year files.", len(df_raw), len(config.YEARS))

    # Build year-transition pairs (past_year_price → current_year_price)
    df = _build_transitions(df_raw)
    log.info("Year-transition pairs constructed: %d rows.", len(df))

    # Validate and clean
    df = _validate(df)

    _success("Validation passed: no duplicate point_ids, all coordinates inside Kumamoto.")
    _status("Wrote %s.", config.STAGE1_PATH)

    # Save
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    df.to_csv(config.STAGE1_PATH, index=False, encoding="utf-8-sig")

    # Summary diagnostics
    print()
    print(f"  Rows       : {len(df)}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"  Years (YSA): {sorted(df['years_since_announcement'].unique())}")
    print(f"  Munis      : {df['municipality'].nunique()} municipalities")
    print(f"  Lat range  : [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}]")
    print(f"  Lon range  : [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]")
    print()
    print(df.dtypes.to_string())

    return df


if __name__ == "__main__":
    run()
