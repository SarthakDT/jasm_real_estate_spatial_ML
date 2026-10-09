"""
Gaussian Process regression with an ARD Matern kernel — attempted as a
direct answer to the diagnosed problem that the RF winner's residuals carry
MORE spatial autocorrelation (Moran's I=0.575) than the raw target (0.49).

HONEST RESULT, kept as a documented negative finding rather than hidden:
this does NOT fix the problem. Three configurations were tried (see git
history / commit messages for exact numbers) — progressively tighter
regularization (length_scale/noise_level floors) and removing
distance_to_jasm_km/spatial_decay (redundant with the kernel's own x_m/y_m
dimensions) — and every one still produced negative held-out Test R^2 AND
residual Moran's I *higher* than RF's 0.575 (measured range across the three
attempts: 0.68-0.87). The mechanism, once diagnosed: a stationary spatial
kernel has essentially zero support for a held-out block that sits far from
every training point (that's the whole point of the spatial-block split —
genuinely novel geography). Far from any training point, a GP's prediction
reverts toward the (post normalize_y=True) global mean — and it does so
*uniformly* for every point inside that same held-out block, which is
precisely the mechanism that produces high spatial autocorrelation in the
residuals: nearby points end up with similar, systematically-biased errors.
This is a different, and here worse, failure mode than RF's plain
overfitting — RF's axis-aligned tree splits can still exploit non-spatial
features (past_year_price, valuation_year) for an unseen region;
a short-length-scale GP effectively can't once it's outside its kernel's
support. The configuration below (moderate regularization, redundant
columns dropped) is kept as the "best attempted" version and as a genuinely
informative comparison point in models/model_comparison.csv, not because it
solved the acceptance criterion this module was built to satisfy.

Uses a single automatic-relevance-determination (ARD) Matern kernel across
standardized features — including projected spatial coordinates — rather
than hand-composing separate per-block kernels via a custom column-subset
sklearn Kernel class (a lower-risk implementation choice; unrelated to why
this underperforms).

Exact GP training is a single O(n^3) Cholesky factorization — at n~1,400
this is sub-second-to-low-seconds per evaluation, even with multiple
optimizer restarts. GPyTorch/sparse-GP machinery is unnecessary at this
scale and would be pure over-engineering; the problem here isn't compute.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config, geo_utils

SPATIAL_COLS = ("latitude", "longitude")

# distance_to_jasm_km and spatial_decay are *deterministic functions* of the
# same (latitude, longitude) already entering the kernel as projected x_m/y_m
# -- carrying them alongside x_m/y_m gave the ARD optimizer redundant degrees
# of freedom it used to drive individual length-scales arbitrarily short and
# memorize training points (see build_model()'s docstring for the measured
# consequence). Dropped here for the GP specifically; RF/XGBoost/MixedLM
# still use all of config.FEATURE_COLS unchanged.
GP_DROPPED_COLS = ("distance_to_jasm_km", "spatial_decay")
GP_N_FEATURES = len(config.FEATURE_COLS) - len(GP_DROPPED_COLS)  # x_m, y_m replace lat/lon 1-for-1


class ProjectedCoordsTransformer(BaseEstimator, TransformerMixin):
    """Replaces (latitude, longitude) degree columns with projected
    (x_m, y_m) meter columns (config.PROJECTED_CRS) so the GP's spatial
    length-scale operates on a proper metric distance, not degrees. Also
    drops GP_DROPPED_COLS — see the module-level comment for why."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        gdf = geo_utils.project(geo_utils.to_gdf(X, lat_col="latitude", lon_col="longitude"))
        out = X.drop(columns=list(SPATIAL_COLS) + list(GP_DROPPED_COLS)).copy()
        out.insert(0, "y_m", gdf.geometry.y.to_numpy())
        out.insert(0, "x_m", gdf.geometry.x.to_numpy())
        return out.to_numpy(dtype="float64")


def build_model(n_restarts: int = 5) -> Pipeline:
    # length_scale_bounds/noise_level_bounds floors are a regularization
    # attempt (see module docstring for why it wasn't sufficient on its own).
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(GP_N_FEATURES), length_scale_bounds=(3e-1, 1e3), nu=1.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-1, 1e2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=config.RANDOM_STATE,
    )
    return Pipeline([
        ("project", ProjectedCoordsTransformer()),
        ("scale", StandardScaler()),
        ("gp", gp),
    ])
