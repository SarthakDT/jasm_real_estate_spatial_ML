"""
National Land Numerical Information (国土数値情報) hazard & land-use layers
for Kumamoto Prefecture, fetched by scripts/download_data.py.

Layers used:
  A31  Flood inundation risk zones (洪水浸水想定区域), FY2012 vintage
  A33  Landslide hazard warning zones (土砂災害警戒区域), FY2018 vintage
  N03  Administrative boundary polygons (行政区域), FY2023 vintage

Source: https://nlftp.mlit.go.jp/ksj/ (Japan Public Data License 1.0)
"""

import os

import geopandas as gpd

from src import config, geo_utils

HAZARD_DIR = os.path.join(config.RAW_DIR, "hazards")
A31_SHP = os.path.join(HAZARD_DIR, "A31_43_GML", "A31-12_43.shp")
A33_SHP = os.path.join(HAZARD_DIR, "A33_43_GML", "A33-18_43Polygon.shp")
N03_GEOJSON = os.path.join(HAZARD_DIR, "N03_43_GML", "N03-23_43_230101.geojson")

# A31_001 flood-depth-rank codes (NLNI spec): 11=0-0.5m ... 15=10m+.
# Mapped to a simple ordinal 1-5 (higher = deeper / more severe).
_FLOOD_RANK_MAP = {11: 1, 12: 2, 13: 3, 14: 4, 15: 5}


def _require(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Hazard data missing: {path}\n"
            "Run `python scripts/download_data.py` to fetch the NLNI archives."
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


def load_admin_boundaries() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(_require(N03_GEOJSON))
    gdf = gdf.rename(columns={"N03_007": "muni_code", "N03_004": "muni_name_ja"})
    gdf["muni_code"] = gdf["muni_code"].astype(str)
    return geo_utils.project(gdf[["muni_code", "muni_name_ja", "geometry"]])


def join_hazards_to_points(points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add flood_risk_class and in_landslide_zone to a GeoDataFrame of benchmark
    points (any CRS in, PROJECTED_CRS used internally). Returns the input
    GeoDataFrame (original CRS/columns preserved) with the two columns appended.

    Zoning is not joined from polygons: MLIT records each plot's zoning
    (用途地域) directly, which is more accurate than a point-in-polygon lookup.
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

    result = points_gdf.reset_index(drop=True).copy()
    result["flood_risk_class"] = pts["flood_risk_class"]
    result["in_landslide_zone"] = pts["in_landslide_zone"]
    return result.set_crs(original_crs, allow_override=True)

