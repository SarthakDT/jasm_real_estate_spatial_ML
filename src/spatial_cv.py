"""
Spatial block cross-validation.

Plain random train/test splitting leaks information for spatially
autocorrelated data: nearby benchmark plots share unobserved local shocks
(a new road, a school district, a rezoning), so a model can partly
"memorize" a neighborhood rather than generalize to unseen geography. This
module replaces `train_test_split` with block-based splitting, where whole
spatial clusters are held out together.

Blocks are assigned via KMeans on projected (meters) coordinates rather than
by municipality: Kumamoto-shi's 5 wards are all close to each other and all
fab-adjacent, so municipality-level grouping would not actually separate
near/far geography the way this dataset needs.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import BaseCrossValidator

from src import config, geo_utils

N_BLOCKS = 8


def assign_spatial_blocks(df, lat_col: str = "latitude", lon_col: str = "longitude", n_blocks: int = N_BLOCKS) -> np.ndarray:
    """Cluster points into n_blocks spatial groups using KMeans on projected coordinates."""
    gdf = geo_utils.project(geo_utils.to_gdf(df, lat_col=lat_col, lon_col=lon_col))
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    km = KMeans(n_clusters=n_blocks, random_state=config.RANDOM_STATE, n_init=10)
    return km.fit_predict(coords)


class SpatialBlockKFold(BaseCrossValidator):
    """K-fold splitter where each fold holds out one whole spatial block.

    Usage mirrors sklearn's GroupKFold: pass `groups=block_ids` to `split()`.
    Compatible with cross_val_score / GridSearchCV / RandomizedSearchCV.
    """

    def __init__(self, n_splits: int = N_BLOCKS):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("SpatialBlockKFold requires `groups` (block ids from assign_spatial_blocks).")
        groups = np.asarray(groups)
        unique_blocks = np.unique(groups)
        n = min(self.n_splits, len(unique_blocks))
        block_folds = np.array_split(np.random.RandomState(config.RANDOM_STATE).permutation(unique_blocks), n)
        indices = np.arange(len(groups))
        for held_out_blocks in block_folds:
            test_mask = np.isin(groups, held_out_blocks)
            yield indices[~test_mask], indices[test_mask]


def spatial_train_test_split(df, test_blocks: int = 2, n_blocks: int = N_BLOCKS):
    """One-shot spatial holdout: assign blocks, hold out `test_blocks` of them as the test set.

    Returns (train_idx, test_idx, block_ids) — positional indices into df.
    """
    block_ids = assign_spatial_blocks(df, n_blocks=n_blocks)
    rng = np.random.RandomState(config.RANDOM_STATE)
    unique_blocks = np.unique(block_ids)
    held_out = rng.choice(unique_blocks, size=test_blocks, replace=False)
    test_mask = np.isin(block_ids, held_out)
    indices = np.arange(len(df))
    return indices[~test_mask], indices[test_mask], block_ids
