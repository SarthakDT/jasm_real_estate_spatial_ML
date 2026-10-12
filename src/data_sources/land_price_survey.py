"""
Shared loader for MLIT's two official land-price surveys.

  L01  Chika-Kouji (地価公示): national survey, valued January 1
  L02  Prefectural survey (都道府県地価調査): valued July 1

Both publish one record per benchmark plot per year with the plot's official
number (municipality code, land-use code, serial) and the number it had the
previous year. Following those links, rather than matching coordinates, keeps
a plot's history intact when MLIT renumbers it, and starts a new history when
MLIT replaces a plot with a different site (those records have no
previous-year number).
"""

import glob
import json
import os
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd

from src import config, geo_utils
from src.logging_utils import get_logger

log = get_logger(__name__)

_NUMERIC = ["price", "published_change_pct", "lot_area_m2", "road_width_m",
            "station_distance_m", "bcr", "far"]
_OPTIONAL = ["published_change_pct", "zoning", "planning_area", "prev_muni_code", "common_point"]

STAGE1_COLUMNS = [
    "point_id", "plot_id", "site_id", "survey", "number", "muni_code", "municipality",
    "valuation_year", "post_announcement", "latitude", "longitude",
    "past_year_price", "current_year_price", "published_change_pct",
    "land_use", "current_use", "lot_area_m2", "road_width_m",
    "station_name", "station_distance_m", "zoning", "planning_area", "bcr", "far",
]


@dataclass(frozen=True)
class Survey:
    code: str                      # "L01" or "L02"
    name: str
    raw_dir: str
    years: list[int]
    schemas: dict[tuple, dict]
    valuation_month: int           # 1 = January 1, 7 = July 1
    stage1_path: str
    id_prefix: str = ""            # prepended to plot ids so the surveys never collide
    excluded_use_codes: tuple = field(default_factory=tuple)

    def schema_for_year(self, year: int) -> dict:
        for (lo, hi), cols in self.schemas.items():
            if lo <= year <= hi:
                return cols
        raise ValueError(f"No {self.code} column schema defined for {year}.")

    def data_path(self, year: int) -> str:
        """The year's GeoJSON, or its shapefile for years MLIT published only as shapefiles."""
        folder = os.path.join(self.raw_dir, f"{self.code}-{str(year)[2:]}_43_GML")
        found = glob.glob(os.path.join(folder, "*.geojson")) or glob.glob(os.path.join(folder, "*.shp"))
        if not found:
            raise FileNotFoundError(
                f"No {self.code} data for {year} in {folder}\n"
                "Run `python scripts/download_data.py` to fetch the MLIT archives."
            )
        return found[0]

    def read_raw(self, year: int) -> pd.DataFrame:
        """Raw properties of every record, plus latitude and longitude."""
        path = self.data_path(year)
        if path.endswith(".geojson"):
            with open(path, encoding="utf-8-sig") as fh:
                features = json.load(fh)["features"]
            df = pd.DataFrame([f["properties"] for f in features])
            coords = [f["geometry"]["coordinates"] for f in features]
        else:
            gdf = gpd.read_file(path, encoding="cp932")
            df = pd.DataFrame(gdf.drop(columns="geometry"))
            coords = [(p.x, p.y) for p in gdf.geometry]
        df["longitude"] = [float(c[0]) for c in coords]
        df["latitude"] = [float(c[1]) for c in coords]
        return df


def load_year(survey: Survey, year: int) -> pd.DataFrame:
    """One year's records with canonical column names."""
    cols = survey.schema_for_year(year)
    raw = survey.read_raw(year)
    df = pd.DataFrame({name: raw[key] for name, key in cols.items()})
    df["latitude"], df["longitude"] = raw["latitude"], raw["longitude"]

    for col in ("muni_code", "use_code", "serial", "prev_use_code", "prev_serial", "site_status"):
        df[col] = df[col].astype(str).str.strip()
    for col in _OPTIONAL:
        if col not in df:
            df[col] = pd.NA
    # Only L01 from 2024 carries the previous year's municipality; otherwise
    # plots never moved between municipalities in this period.
    df["prev_muni_code"] = df["prev_muni_code"].fillna(df["muni_code"]).astype(str).str.strip()
    df["common_point"] = df["common_point"].astype(str).str.strip().str.lower().isin(("true", "1"))

    df["price"] = df["price"].astype(str).str.replace(r"[^\d]", "", regex=True)
    for col in _NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("int64")
    if (df["year"] != year).any():
        raise ValueError(f"{survey.code} {year} file contains records for other years: {df['year'].unique()}")

    df = df[~df["use_code"].isin(survey.excluded_use_codes)].reset_index(drop=True)
    df["number"] = df["muni_code"] + "-" + df["use_code"] + "-" + df["serial"]
    df["prev_number"] = df["prev_muni_code"] + "-" + df["prev_use_code"] + "-" + df["prev_serial"]
    log.info("  %s %d: %d records loaded", survey.code, year, len(df))
    return df


def assign_plot_ids(frames: dict[int, pd.DataFrame], id_prefix: str = "") -> dict[int, pd.DataFrame]:
    """
    Give every record a plot_id that stays the same along its chain of
    previous-year links. A plot's id is its official number in the first year
    it appears, plus that year, because MLIT reuses a number when it replaces
    a plot with a new site.
    """
    prev_ids: dict[str, str] = {}
    for year in sorted(frames):
        df = frames[year]
        linked = df["prev_number"].map(prev_ids)
        df["linked"] = linked.notna()
        df["plot_id"] = linked.fillna(id_prefix + df["number"] + "/" + str(year))
        prev_ids = dict(zip(df["number"], df["plot_id"]))
    return frames


def load_years(survey: Survey) -> dict[int, pd.DataFrame]:
    """All raw years of a survey, parsed and with plot_ids assigned."""
    return assign_plot_ids({y: load_year(survey, y) for y in survey.years}, survey.id_prefix)


def build_transitions(survey: Survey, frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """One row per plot per year where the plot also has a previous-year price."""
    out = []
    for year in sorted(frames)[1:]:
        prev_price = frames[year - 1].set_index("plot_id")["price"]
        cur = frames[year][frames[year]["linked"]].copy()
        cur["past_year_price"] = cur["plot_id"].map(prev_price)
        out.append(cur)
    df = pd.concat(out, ignore_index=True)

    df = df.rename(columns={"year": "valuation_year", "price": "current_year_price"})
    df["survey"] = survey.code
    df["site_id"] = df["plot_id"]
    df["point_id"] = df["plot_id"] + "_Y" + df["valuation_year"].astype(str)
    df["municipality"] = df["muni_code"].map(config.MUNI_NAMES).fillna(df["muni_code"])
    df["post_announcement"] = df["valuation_year"] >= config.FIRST_POST_VALUATION_YEAR
    df["land_use"] = df["use_code"].map(config.LAND_USE_CODES).fillna("other")
    df["planning_area"] = df["planning_area"].map(config.PLANNING_AREAS)
    df["zoning"] = df["zoning"].replace("_", "none")
    return df[STAGE1_COLUMNS]


def validate_kumamoto_bbox(df: pd.DataFrame) -> pd.Series:
    """Mask of rows outside Kumamoto Prefecture's bounding box: a coarse check for garbled coordinates."""
    return (
        ~df["latitude"].between(config.LAT_MIN, config.LAT_MAX, inclusive="neither")
        | ~df["longitude"].between(config.LON_MIN, config.LON_MAX, inclusive="neither")
    )


def validate(survey: Survey, df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the Stage 1 data contract. Raises ValueError listing every violation."""
    errors: list[str] = []

    dup_count = df["point_id"].duplicated().sum()
    if dup_count:
        errors.append(f"{dup_count} duplicate point_id value(s).")

    bad_muni = ~df["municipality"].isin(config._VALID_MUNIS)
    if bad_muni.any():
        errors.append(f"Unrecognised municipality codes: {df.loc[bad_muni, 'muni_code'].unique().tolist()}")

    # Files up to 2023 are in JGD2000 (EPSG:4612) and from 2024 in JGD2011;
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

    unknown_planning = df["planning_area"].isna() & df["valuation_year"].ge(min(
        lo for (lo, _), cols in survey.schemas.items() if "planning_area" in cols))
    if unknown_planning.any():
        errors.append(f"{unknown_planning.sum()} record(s) have an unrecognised planning area.")

    bad_year = ~df["valuation_year"].between(survey.years[0] + 1, survey.years[-1])
    if bad_year.any():
        errors.append(f"valuation_year out of range: {df.loc[bad_year, 'valuation_year'].unique().tolist()}")

    if errors:
        raise ValueError(f"{survey.code} stage 1 validation failed:\n  • " + "\n  • ".join(errors))

    df = df.copy()
    df["past_year_price"] = df["past_year_price"].astype("int64")
    df["current_year_price"] = df["current_year_price"].astype("int64")
    return df


def match_common_sites(l01_frames: dict[int, pd.DataFrame], l02_frames: dict[int, pd.DataFrame],
                       max_m: float = config.COMMON_SITE_MAX_DISTANCE_M) -> pd.DataFrame:
    """
    Pair L02 plots with the L01 plot at the same location (MLIT's "common
    points", valued by both surveys). Compares records of the same calendar
    year, which are always in the same datum. Returns one row per L02 plot
    that has a partner: l02_plot_id, l01_plot_id, distance_m, flagged (MLIT's
    own common-point flag on any of the plot's L02 records).
    """
    pairs = []
    for year in sorted(set(l01_frames) & set(l02_frames)):
        a = geo_utils.project(geo_utils.to_gdf(l02_frames[year][["plot_id", "common_point", "latitude", "longitude"]]))
        b = geo_utils.project(geo_utils.to_gdf(l01_frames[year][["plot_id", "latitude", "longitude"]]))
        near = a.sjoin_nearest(b, distance_col="distance_m", lsuffix="l02", rsuffix="l01")
        near = near[near["distance_m"] <= max_m]
        pairs.append(near[["plot_id_l02", "plot_id_l01", "distance_m", "common_point"]].assign(year=year))
    pairs = pd.concat(pairs, ignore_index=True)
    return (
        pairs.sort_values("distance_m")
        .groupby("plot_id_l02")
        .agg(l01_plot_id=("plot_id_l01", "first"), distance_m=("distance_m", "min"),
             flagged=("common_point", "any"), years=("year", "nunique"))
        .reset_index().rename(columns={"plot_id_l02": "l02_plot_id"})
    )
