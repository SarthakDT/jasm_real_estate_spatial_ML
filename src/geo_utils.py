"""Shared geopandas helpers: DataFrame <-> GeoDataFrame conversion and CRS projection."""

import geopandas as gpd
from shapely.geometry import Point

from src import config


def to_gdf(df, lat_col: str = "latitude", lon_col: str = "longitude", crs: str = config.WGS84) -> gpd.GeoDataFrame:
    """Wrap a plain DataFrame with lat/lon columns into a GeoDataFrame of Points."""
    geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)


def project(gdf: gpd.GeoDataFrame, to_crs: str = config.PROJECTED_CRS) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to the Kumamoto-appropriate projected CRS (meters)."""
    return gdf.to_crs(to_crs)


def distance_to_jasm_km(df, lat_col: str = "latitude", lon_col: str = "longitude"):
    """Straight-line distance (km) from each row's coordinates to the JASM fab, in the projected CRS."""
    points = project(to_gdf(df, lat_col=lat_col, lon_col=lon_col))
    fab = gpd.GeoSeries([jasm_point()], crs=config.WGS84).to_crs(config.PROJECTED_CRS).iloc[0]
    return (points.geometry.distance(fab) / 1000.0).to_numpy()


def jasm_point(crs: str = config.WGS84) -> Point:
    """The JASM fab centroid as a shapely Point in the given CRS's coordinate order."""
    lat, lon = config.JASM_COORDINATES
    return Point(lon, lat)
