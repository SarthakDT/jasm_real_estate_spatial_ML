"""
Spatial autocorrelation diagnostics (Moran's I).

Run on observed price changes to show why a random train/test split leaks
(nearby plots move together), and on forecast residuals to check whether the
model has captured that spatial structure (residual Moran's I near zero) or
left it on the table.

Moran's I is computed separately for each valuation year. Pooling years
would make every plot its own nearest neighbour (same coordinates in every
year) and overstate the spatial correlation.
"""

import json

import geopandas as gpd
import libpysal
import numpy as np
from esda.moran import Moran

from src import config, geo_utils
from src.logging_utils import get_logger

log = get_logger(__name__)

K_NEIGHBORS = 8


def compute_morans_i(gdf: gpd.GeoDataFrame, value_col: str, k: int = K_NEIGHBORS) -> dict:
    """Global Moran's I for `value_col` using a k-nearest-neighbours spatial weights matrix."""
    projected = geo_utils.project(gdf) if gdf.crs != config.PROJECTED_CRS else gdf
    w = libpysal.weights.KNN.from_dataframe(projected, k=k)
    w.transform = "r"
    mi = Moran(projected[value_col].to_numpy(), w, permutations=999)
    return {"n": int(len(projected)), "morans_i": float(mi.I), "p_value": float(mi.p_sim)}


def by_year(gdf: gpd.GeoDataFrame, value_col: str) -> dict:
    years = {int(y): compute_morans_i(g, value_col) for y, g in gdf.groupby("valuation_year")}
    values = [r["morans_i"] for r in years.values()]
    return {
        "value_col": value_col,
        "k_neighbors": K_NEIGHBORS,
        "by_year": years,
        "mean_morans_i": float(np.mean(values)),
        "max_p_value": float(max(r["p_value"] for r in years.values())),
    }


def run(residual_df=None) -> dict:
    """Moran's I of observed post-announcement changes, and of forecast residuals if given
    (a DataFrame with latitude, longitude, valuation_year and residual columns)."""
    gdf = gpd.read_parquet(config.STAGE2_GEOPARQUET_PATH)
    gdf = gdf[gdf["post_announcement"]].reset_index(drop=True)

    results = {"observed_change": by_year(gdf, config.TARGET_COL)}
    log.info("Observed change Moran's I (mean over years): %.3f", results["observed_change"]["mean_morans_i"])

    if residual_df is not None and len(residual_df) > 0:
        results["forecast_residuals"] = by_year(geo_utils.to_gdf(residual_df), "residual")
        log.info("Forecast residual Moran's I (mean over years): %.3f",
                 results["forecast_residuals"]["mean_morans_i"])

    with open(config.SPATIAL_DIAGNOSTICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return results


if __name__ == "__main__":
    run()
