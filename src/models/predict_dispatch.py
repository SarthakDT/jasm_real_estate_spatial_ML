"""
Every model in the comparison (src/train.py) supports genuine out-of-sample
prediction, but not all through the same sklearn-style `model.predict(X)`
call: MixedLMModel needs the `municipality` grouping column alongside the
feature matrix, and GM_Error's out-of-sample prediction is a module-level
function (spreg models don't implement `.predict()` at all). Any code that
treats "the winning model" generically — evaluate.py, backtest.py,
dashboard.py — should go through `predict()` here rather than assuming
`model.predict(X)` works for every candidate.
"""

import numpy as np
import pandas as pd

from src.models import spatial_baseline


def predict(model, model_name: str, X: pd.DataFrame, df_slice: pd.DataFrame | None = None) -> np.ndarray:
    """
    model_name: one of the keys src/train.py's `candidates` dict can produce
                ("RandomForest", "XGBoost", "GaussianProcess", "MixedEffects",
                "SpatialError (GM_Error)").
    df_slice:   required only for "MixedEffects" — the full-column row slice
                (same rows/order as X) including the `municipality` column.
    """
    if model_name == "MixedEffects":
        if df_slice is None:
            raise ValueError("MixedEffects prediction requires df_slice (needs the municipality column).")
        return model.predict(df_slice)
    if model_name.startswith("SpatialError"):
        return spatial_baseline.predict_out_of_sample(model, X)
    return model.predict(X)
