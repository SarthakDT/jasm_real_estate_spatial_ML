"""
Stage 1 — Data ingestion and validation
MLIT Chika-Kouji (地価公示) benchmark land prices, Kumamoto Prefecture

Reads the 2019-2026 GeoJSON releases, links each benchmark plot to its
previous-year record using MLIT's own plot numbers, and writes one row per
year-over-year price transition to data/processed/stage1.csv.

Plot linkage: each record carries its official number (municipality code,
land-use code, serial) and the number it had the previous year. Following
those links, rather than matching coordinates, keeps a plot's history intact
when MLIT renumbers it, and starts a new history when MLIT replaces a plot
with a different site (those records have no previous-year number).
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


_NUMERIC = ["price", "published_change_pct", "lot_area_m2", "road_width_m",
            "station_distance_m", "bcr", "far"]


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
    cols = _schema_for_year(year)
    with open(_geojson_path(year), encoding="utf-8-sig") as fh:
        gj = json.load(fh)

    rows = []
    for feat in gj["features"]:
        p = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        row = {name: p.get(key) for name, key in cols.items()}
        row["latitude"], row["longitude"] = float(lat), float(lon)
        rows.append(row)
    df = pd.DataFrame(rows)

    for col in ("muni_code", "use_code", "serial", "prev_use_code", "prev_serial", "site_status"):
        df[col] = df[col].astype(str).str.strip()
    # Before 2024 the previous-year number has no municipality part; plots
    # never moved between municipalities in this period.
    if "prev_muni_code" not in df:
        df["prev_muni_code"] = df["muni_code"]
    df["prev_muni_code"] = df["prev_muni_code"].astype(str).str.strip()

    df["price"] = df["price"].astype(str).str.replace(r"[^\d]", "", regex=True)
    if "published_change_pct" not in df:
        df["published_change_pct"] = float("nan")
    for col in _NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int64")
    if (df["year"] != year).any():
        raise ValueError(f"{year} file contains records for other years: {df['year'].unique()}")

    df["number"] = df["muni_code"] + "-" + df["use_code"] + "-" + df["serial"]
    df["prev_number"] = df["prev_muni_code"] + "-" + df["prev_use_code"] + "-" + df["prev_serial"]

    log.info("  %d: %d records loaded", year, len(df))
    return df


def _assign_plot_ids(frames: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """
    Give every record a plot_id that stays the same along its chain of
    previous-year links. A plot's id is its official number in the first year
    it appears, plus that year, because MLIT reuses a number when it replaces
    a plot with a new site.
    """
    years = sorted(frames)
    prev_ids: dict[str, str] = {}
    for year in years:
        df = frames[year]
        linked = df["prev_number"].map(prev_ids)
        df["linked"] = linked.notna()
        df["plot_id"] = linked.fillna(df["number"] + "/" + str(year))
        prev_ids = dict(zip(df["number"], df["plot_id"]))
    return frames


STAGE1_COLUMNS = [
    "point_id", "plot_id", "number", "muni_code", "municipality",
    "valuation_year", "post_announcement", "latitude", "longitude",
    "past_year_price", "current_year_price", "published_change_pct",
    "land_use", "current_use", "lot_area_m2", "road_width_m",
    "station_name", "station_distance_m", "zoning", "planning_area", "bcr", "far",
]


def _build_transitions(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """One row per plot per year where the plot also has a previous-year price."""
    years = sorted(frames)
    out = []
    for year in years[1:]:
        prev_price = frames[year - 1].set_index("plot_id")["price"]
        cur = frames[year][frames[year]["linked"]].copy()
        cur["past_year_price"] = cur["plot_id"].map(prev_price)
        out.append(cur)
    df = pd.concat(out, ignore_index=True)

    df = df.rename(columns={"year": "valuation_year", "price": "current_year_price"})
    df["point_id"] = df["plot_id"] + "_Y" + df["valuation_year"].astype(str)
    df["municipality"] = df["muni_code"].map(config.MUNI_NAMES).fillna(df["muni_code"])
    df["post_announcement"] = df["valuation_year"] >= config.FIRST_POST_VALUATION_YEAR
    df["land_use"] = df["use_code"].map(config.LAND_USE_CODES).fillna("other")
    df["planning_area"] = df["planning_area"].map(config.PLANNING_AREAS).fillna("other")
    df["zoning"] = df["zoning"].replace("_", "none")
    return df[STAGE1_COLUMNS]


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the Stage 1 data contract. Raises ValueError listing every violation."""
    errors: list[str] = []

    dup_count = df["point_id"].duplicated().sum()
    if dup_count:
        errors.append(f"{dup_count} duplicate point_id value(s).")

    bad_muni = ~df["municipality"].isin(config._VALID_MUNIS)
    if bad_muni.any():
        errors.append(f"Unrecognised municipality codes: {df.loc[bad_muni, 'muni_code'].unique().tolist()}")

    # The 2019-2023 files are in JGD2000 (EPSG:4612) and 2024+ in JGD2011;
    # both agree with WGS84 to well under a metre here, so no transform is
    # needed. This check only catches coordinates that are outright wrong.
    oob = validate_kumamoto_bbox(df)
    if oob.any():
        errors.append(f"{oob.sum()} record(s) fall outside Kumamoto Prefecture's bounding box.")

    for col in ("past_year_price", "current_year_price"):
        if df[col].isna().any() or df[col].le(0).any():
            errors.append(f"{col} has missing or non-positive values.")

    for col in ("lot_area_m2", "road_width_m", "station_distance_m", "bcr", "far"):
        if df[col].isna().any():
            errors.append(f"{col} has {df[col].isna().sum()} missing value(s).")

    bad_year = ~df["valuation_year"].between(config.YEARS[0] + 1, config.YEARS[-1])
    if bad_year.any():
        errors.append(f"valuation_year out of range: {df.loc[bad_year, 'valuation_year'].unique().tolist()}")

    if errors:
        raise ValueError("Stage 1 validation failed:\n  • " + "\n  • ".join(errors))

    df = df.copy()
    df["past_year_price"] = df["past_year_price"].astype("int64")
    df["current_year_price"] = df["current_year_price"].astype("int64")
    return df


def load_years() -> dict[int, pd.DataFrame]:
    """All raw years, parsed and with plot_ids assigned."""
    return _assign_plot_ids({y: _load_year(y) for y in config.YEARS})


def price_history(frames: dict[int, pd.DataFrame], release_year: int, first_year: int) -> pd.DataFrame:
    """
    Long-format (plot_id, year, price) history for every plot in one release,
    from first_year up to release_year, read from the history columns MLIT
    ships with each record. Years in which the site was not a benchmark plot
    are dropped, so the history always refers to the same site.
    """
    for (lo, hi), (flag_key, first_col) in config._HISTORY.items():
        if lo <= release_year <= hi:
            break
    else:
        raise ValueError(f"No history columns defined for {release_year}.")

    plot_ids = frames[release_year].set_index("number")["plot_id"]
    with open(_geojson_path(release_year), encoding="utf-8-sig") as fh:
        features = json.load(fh)["features"]

    cols = _schema_for_year(release_year)
    rows = []
    for feat in features:
        p = feat["properties"]
        number = "-".join(str(p[cols[k]]).strip() for k in ("muni_code", "use_code", "serial"))
        flags = str(p[flag_key])
        for year in range(first_year, release_year + 1):
            offset = year - config.HISTORY_FIRST_YEAR
            price = int(p.get(f"L01_{first_col + offset:03d}") or 0)
            if price > 0 and offset < len(flags) and flags[offset] == "1":
                rows.append({"plot_id": plot_ids[number], "year": year, "price": price})
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    log.info("Loading MLIT Chika-Kouji files for %d-%d.", config.YEARS[0], config.YEARS[-1])
    frames = load_years()
    log.info("Raw records: %d across %d files.", sum(len(f) for f in frames.values()), len(frames))

    df = _validate(_build_transitions(frames))
    _success("Validation passed: %d transitions across %d plots, all coordinates inside Kumamoto.",
             len(df), df["plot_id"].nunique())

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    df.to_csv(config.STAGE1_PATH, index=False, encoding="utf-8-sig")
    _status("Wrote %s.", config.STAGE1_PATH)

    print()
    print(f"  Transitions : {len(df)}")
    print(f"  Plots       : {df['plot_id'].nunique()}")
    print(f"  Per year    : {df['valuation_year'].value_counts().sort_index().to_dict()}")
    print(f"  Land use    : {df['land_use'].value_counts().to_dict()}")
    print(f"  Munis       : {df['municipality'].nunique()} municipalities")
    return df


if __name__ == "__main__":
    run()
