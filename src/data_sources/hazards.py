"""
National Land Numerical Information (国土数値情報) hazard & land-use layers
for Kumamoto Prefecture — manually downloaded (no API), same pattern as the
Chika-Kouji GML folders.

Layers used:
  A31  Flood inundation risk zones (洪水浸水想定区域), FY2012 vintage
  A33  Landslide hazard warning zones (土砂災害警戒区域), FY2018 vintage
  A29  Urban planning zones (用途地域), FY2019 vintage
  N03  Administrative boundary polygons (行政区域), FY2023 vintage

Source: https://nlftp.mlit.go.jp/ksj/ (Japan Public Data License 1.0)
"""

import glob
import os

import geopandas as gpd
import pandas as pd

from src import config, geo_utils
from src.logging_utils import get_logger

log = get_logger(__name__)

HAZARD_DIR = os.path.join(config.RAW_DIR, "hazards")
A31_SHP = os.path.join(HAZARD_DIR, "A31_43_GML", "A31-12_43.shp")
A33_SHP = os.path.join(HAZARD_DIR, "A33_43_GML", "A33-18_43Polygon.shp")
A29_GLOB = os.path.join(HAZARD_DIR, "A29_43_GML", "A29-19_43*.geojson")
N03_GEOJSON = os.path.join(HAZARD_DIR, "N03_43_GML", "N03-23_43_230101.geojson")

# A31_001 flood-depth-rank codes (NLNI spec): 11=0-0.5m ... 15=10m+.
# Mapped to a simple ordinal 1-5 (higher = deeper / more severe).
_FLOOD_RANK_MAP = {11: 1, 12: 2, 13: 3, 14: 4, 15: 5}


def _require(path: str) -> str:
    if not os.path.isfile(path) and not glob.glob(path):
        raise FileNotFoundError(
            f"Hazard data missing: {path}\n"
            "Download the NLNI archive(s) from https://nlftp.mlit.go.jp/ksj/ "
            "and extract into data/raw/hazards/."
        )
    return path


def _load_flood_zones() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(_require(A31_SHP), encoding="cp932")
    gdf["flood_risk_class"] = gdf["A31_001"].map(_FLOOD_RANK_MAP).fillna(0).astype(int)
    return geo_utils.project(gdf[["flood_risk_class", "geometry"]])


def _load_landslide_zones() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(_require(A33_SHP), encoding="cp932")
    # A33_001: 1=warning zone (イエローゾーン), 2=special warning zone (レッドゾーン)
    gdf = gdf[gdf["A33_001"].isin([1, 2])]
    return geo_utils.project(gdf[["geometry"]])


def _load_urban_planning_zones() -> gpd.GeoDataFrame:
    paths = _require(A29_GLOB) and sorted(glob.glob(A29_GLOB))
    frames = [gpd.read_file(p) for p in paths]
    gdf = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=frames[0].crs)
    gdf = gdf.rename(columns={"A29_003": "urban_planning_zone_type"})
    return geo_utils.project(gdf[["urban_planning_zone_type", "geometry"]])


def load_admin_boundaries() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(_require(N03_GEOJSON))
    gdf = gdf.rename(columns={"N03_007": "muni_code", "N03_004": "muni_name_ja"})
    gdf["muni_code"] = gdf["muni_code"].astype(str)
    return geo_utils.project(gdf[["muni_code", "muni_name_ja", "geometry"]])


def join_hazards_to_points(points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add flood_risk_class, in_landslide_zone, urban_planning_zone_type to a
    GeoDataFrame of benchmark points (any CRS in, PROJECTED_CRS used internally).
    Returns the input GeoDataFrame (original CRS/columns preserved) with the
    three new columns appended.
    """
    original_crs = points_gdf.crs
    pts = geo_utils.project(points_gdf).reset_index(drop=True)
    pts["_pid"] = pts.index

    flood = _load_flood_zones()
    joined = gpd.sjoin(pts[["_pid", "geometry"]], flood, how="left", predicate="intersects")
    flood_max = joined.groupby("_pid")["flood_risk_class"].max()
    pts["flood_risk_class"] = pts["_pid"].map(flood_max).fillna(0).astype(int)

    landslide = _load_landslide_zones()
    joined = gpd.sjoin(pts[["_pid", "geometry"]], landslide, how="left", predicate="intersects")
    in_zone = joined.groupby("_pid").apply(lambda g: g["index_right"].notna().any(), include_groups=False)
    pts["in_landslide_zone"] = pts["_pid"].map(in_zone).fillna(False).astype(bool)

    urban = _load_urban_planning_zones()
    joined = gpd.sjoin(pts[["_pid", "geometry"]], urban, how="left", predicate="intersects")
    zone_type = joined.groupby("_pid")["urban_planning_zone_type"].first()
    pts["urban_planning_zone_type"] = pts["_pid"].map(zone_type)

    result = points_gdf.reset_index(drop=True).copy()
    result["flood_risk_class"] = pts["flood_risk_class"]
    result["in_landslide_zone"] = pts["in_landslide_zone"]
    result["urban_planning_zone_type"] = pts["urban_planning_zone_type"]
    return result.set_crs(original_crs, allow_override=True)


if __name__ == "__main__":
    admin = load_admin_boundaries()
    log.info("Loaded %d admin boundary polygons.", len(admin))
    flood = _load_flood_zones()
    log.info("Loaded %d flood risk polygons.", len(flood))
    landslide = _load_landslide_zones()
    log.info("Loaded %d landslide hazard polygons.", len(landslide))
    urban = _load_urban_planning_zones()
    log.info("Loaded %d urban planning zone polygons.", len(urban))
