"""Gradient-boosted trees (XGBoost), tuned by randomized search with
cross-validation grouped by plot, so a plot's other years never sit on both
sides of an inner split."""

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

from src import config

PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 5, 10],
}
N_ITER = 12
INNER_FOLDS = 3


def build_and_tune(X_train, y_train, groups):
    """Randomized hyperparameter search scored on grouped inner CV, then refit on all of X_train."""
    search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=1),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        cv=GroupKFold(n_splits=INNER_FOLDS),
        scoring="r2",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train, groups=groups)
    return search.best_estimator_, search.best_params_
