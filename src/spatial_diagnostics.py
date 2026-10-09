"""
Spatial autocorrelation diagnostics (Moran's I).

Run on raw prices to establish *why* a plain random train/test split leaks
information (nearby plots share unobserved local shocks), and on model
residuals to check whether a trained model has actually captured that
spatial structure (residual Moran's I near zero = good; still significantly
positive = the model is missing spatial pattern a spatial-aware model could
capture).
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
    """Global Moran's I for `value_col` using a k-nearest-neighbors spatial weights matrix."""
    projected = geo_utils.project(gdf) if gdf.crs != config.PROJECTED_CRS else gdf
    w = libpysal.weights.KNN.from_dataframe(projected, k=k)
    w.transform = "r"
    y = projected[value_col].to_numpy()
    mi = Moran(y, w)
    return {
        "value_col": value_col,
        "k_neighbors": k,
        "n": int(len(y)),
        "morans_i": float(mi.I),
        "expected_i": float(mi.EI),
        "p_value": float(mi.p_sim),
        "z_score": float(mi.z_sim),
        "interpretation": _interpret(mi.I, mi.p_sim),
    }


def _interpret(i_stat: float, p_value: float) -> str:
    if p_value >= 0.05:
        return "No significant spatial autocorrelation detected (p >= 0.05)."
    direction = "positive" if i_stat > 0 else "negative"
    strength = "strong" if abs(i_stat) > 0.3 else "moderate" if abs(i_stat) > 0.1 else "weak"
    return (
        f"Statistically significant {strength} {direction} spatial autocorrelation "
        f"(I={i_stat:.3f}, p={p_value:.4f}). Nearby points have {'similar' if i_stat > 0 else 'dissimilar'} values."
    )


def run(residual_df: gpd.GeoDataFrame | None = None) -> dict:
    """Compute Moran's I on raw target_price_pct_change, and on model residuals if provided.

    residual_df: optional GeoDataFrame with a 'residual' column (test-set
    predictions minus actuals), indexed to line up with geometry — passed in
    by whichever training/evaluation step has just produced predictions.
    """
    gdf = gpd.read_parquet(config.STAGE2_GEOPARQUET_PATH)
    gdf = gdf[gdf["post_announcement"]].reset_index(drop=True)

    results = {"raw_price_change": compute_morans_i(gdf, config.TARGET_COL)}
    log.info("Raw target Moran's I: %s", results["raw_price_change"]["interpretation"])

    if residual_df is not None and len(residual_df) > 0:
        results["model_residuals"] = compute_morans_i(residual_df, "residual")
        log.info("Residual Moran's I: %s", results["model_residuals"]["interpretation"])

    with open(config.SPATIAL_DIAGNOSTICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    log.info("Spatial diagnostics saved to %s", config.SPATIAL_DIAGNOSTICS_PATH)

    return results


if __name__ == "__main__":
    run()
