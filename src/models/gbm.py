"""Gradient-boosted model (XGBoost), hyperparameter-tuned via RandomizedSearchCV
using the spatial block CV splitter — so the search itself never leaks across
nearby plots either."""

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

from src import config
from src.spatial_cv import SpatialBlockKFold

PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}
N_ITER = 20


def build_and_tune(X_train, y_train, block_ids_train, n_splits: int = 5) -> XGBRegressor:
    """Randomized hyperparameter search over XGBRegressor, scored with spatial-block CV
    on the training portion only, then refit on the full training set with the best params.
    """
    search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=-1),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        cv=SpatialBlockKFold(n_splits=n_splits),
        scoring="r2",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train, groups=block_ids_train)
    return search.best_estimator_, search.best_params_
