"""Two interpretable spatial-econometrics baselines, reported alongside the
ML models — their value is a transparent, coefficient-level answer a
government reader can audit, not out-competing RF/XGBoost/GP on held-out
error.

`fit_spatial_lag` (GM_Lag) is fit in-sample only, on purpose: its spatial
dependence term is the *endogenous* lag of y itself (rho * W*y), which
creates a real simultaneity problem for out-of-sample prediction — a held-out
test point's own lag term would need its (unknown) neighbors' y values, and
if two test points are each other's nearest neighbors, predicting one
requires the other. Doing this correctly needs a bipartite train-only
weights matrix, deferred as future work rather than done partially.

`fit_spatial_error` (GM_Error) puts the spatial dependence in the disturbance
term instead, so `y_pred = X @ beta` is well-defined for genuinely new
points — no simultaneity problem — and IS evaluated out-of-sample on the
same spatial-block test split as every other model here, closing the gap
GM_Lag leaves open.
"""

import numpy as np
import libpysal
from spreg import GM_Error, GM_Lag

from src import config, geo_utils


def _build_weights(gdf, k: int = 8):
    projected = geo_utils.project(gdf) if gdf.crs != config.PROJECTED_CRS else gdf
    w = libpysal.weights.KNN.from_dataframe(projected, k=k)
    w.transform = "r"
    return w


def fit_spatial_lag(X, y, gdf, k: int = 8):
    """Fit a spatial-lag model on the full (in-sample) dataset.

    X: DataFrame of feature columns (config.FEATURE_COLS, minus lat/lon which
       are already implicitly captured by the spatial lag term).
    y: Series, the target.
    gdf: GeoDataFrame with matching row order, used to build KNN spatial weights.
    """
    w = _build_weights(gdf, k=k)
    y_arr = y.to_numpy().reshape(-1, 1)
    x_arr = X.to_numpy()

    return GM_Lag(
        y_arr, x_arr, w=w,
        name_y=y.name, name_x=list(X.columns),
        spat_diag=False,
    )


def fit_spatial_error(X_train, y_train, gdf_train, k: int = 8):
    """Fit a spatial-error model on the training portion of the spatial-block
    split. Spatial weights are built from the training points only.
    """
    w = _build_weights(gdf_train, k=k)
    y_arr = y_train.to_numpy().reshape(-1, 1)
    x_arr = X_train.to_numpy()

    return GM_Error(
        y_arr, x_arr, w=w,
        name_y=y_train.name, name_x=list(X_train.columns),
    )


def predict_out_of_sample(model, X_test) -> np.ndarray:
    """y_pred = X @ beta for a GM_Error model — valid out-of-sample because
    GM_Error's spatial dependence lives in the disturbance term, not in an
    endogenous lag of y (see module docstring). Deliberately excludes any
    spatial adjustment (a BLUP-style correction using the training
    residuals' spatial filter would need each test point's relationship to
    the *training* weights matrix, left as future work) — this is the plain
    Xb prediction only.

    model.betas has one entry per name_x column, prefixed by CONSTANT, and
    (for GM_Error specifically) suffixed by the estimated spatial
    autoregressive parameter (lambda) for the disturbance term — sliced off
    here since it multiplies the spatially-filtered residual, not a plain X
    column.
    """
    n_coefs = X_test.shape[1] + 1  # CONSTANT + one per feature column
    betas = model.betas.flatten()[:n_coefs]
    x_with_const = np.column_stack([np.ones(len(X_test)), X_test.to_numpy()])
    return x_with_const @ betas


def _pseudo_r2(model) -> float:
    if hasattr(model, "pr2") and model.pr2 is not None:
        return float(model.pr2)
    # Fallback: GM_Error doesn't always populate pr2 depending on spreg
    # version -- compute it directly from residuals vs. total variance.
    y = model.u.flatten() + model.predy.flatten()
    ss_res = float(np.sum(model.u.flatten() ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def summarize(model, kind: str = "lag") -> dict:
    """Extract a JSON-safe summary: pseudo R^2 and per-coefficient (beta, z, p).

    kind: "lag" (GM_Lag, appends the endogenous W_y coefficient name) or
          "error" (GM_Error, no trailing spatial term in the beta vector).
    """
    names = list(model.name_x)  # spreg auto-prepends "CONSTANT" to name_x
    if kind == "lag":
        names = names + ["W_" + model.name_y]  # spreg appends the endogenous spatial-lag coefficient last
    elif kind == "error":
        names = names + ["lambda (spatial-error autocorrelation)"]  # spreg appends this last

    coefs = []
    for name, beta, (z, p) in zip(names, model.betas.flatten(), model.z_stat):
        coefs.append({"name": name, "beta": float(beta), "z_score": float(z), "p_value": float(p)})

    note = (
        "In-sample spatial-lag model (spreg.GM_Lag) — pseudo R^2 is not "
        "directly comparable to the RF/XGBoost/GP held-out spatial-block Test R^2."
        if kind == "lag" else
        "Spatial-error model (spreg.GM_Error), fit on the spatial-block training "
        "split and evaluated out-of-sample on the same held-out test blocks as "
        "every other model here — pseudo_r2 below is in-sample training fit; see "
        "model_comparison.csv for its genuine held-out Test R^2/RMSE/MAE."
    )
    return {
        "pseudo_r2": _pseudo_r2(model),
        "n": int(model.n),
        "coefficients": coefs,
        "note": note,
    }
