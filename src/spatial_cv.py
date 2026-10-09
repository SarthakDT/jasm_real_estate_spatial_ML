"""
Spatial block cross-validation.

Nearby benchmark plots share local shocks (a new road, a rezoning, the fab),
so a model can do well on a random split by recognising a neighbourhood it
has already seen. Holding out whole spatial blocks tests whether it can
predict areas it has not seen at all.

Blocks are KMeans clusters of projected (metre) coordinates rather than
municipalities: Kumamoto-shi's five wards are small, adjacent and all close
to the fab, so municipality groups would not separate near from far.
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
    """K-fold splitter where each fold holds out whole spatial blocks.

    Usage mirrors sklearn's GroupKFold: pass `groups=block_ids` to `split()`.
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
