"""
Refits a fresh copy of whichever model won Stage 3 on a different train
slice (used by src/backtest.py for the out-of-time train-through-2024 slice).
Not every candidate has a uniform sklearn `clone(); fit(X, y)` interface:
MixedEffects needs the municipality grouping column, and GM_Error is a
spreg object (constructed fresh with y/x/w at fit time — not a
clone-and-fit sklearn estimator, no get_params/set_params).
"""

import pandas as pd
from sklearn.base import clone

from src import geo_utils
from src.models import mixed_effects, spatial_baseline


def refit(model_name: str, winner_model, X_train: pd.DataFrame, y_train: pd.Series, df_train: pd.DataFrame | None = None):
    """df_train (full-column row slice matching X_train's rows) is required
    for MixedEffects and SpatialError; ignored for the sklearn-style models."""
    if model_name == "MixedEffects":
        if df_train is None:
            raise ValueError("MixedEffects refit requires df_train (needs the municipality column).")
        return mixed_effects.MixedLMModel(formula=winner_model.formula).fit(df_train, y_train)

    if model_name.startswith("SpatialError"):
        if df_train is None:
            raise ValueError("SpatialError refit requires df_train (needs geometry).")
        gdf_train = geo_utils.to_gdf(df_train)
        return spatial_baseline.fit_spatial_error(X_train, y_train, gdf_train)

    # RandomForest / XGBoost / GaussianProcess: uniform sklearn clone-and-fit API.
    fresh_model = clone(winner_model)
    fresh_model.fit(X_train, y_train)
    return fresh_model
