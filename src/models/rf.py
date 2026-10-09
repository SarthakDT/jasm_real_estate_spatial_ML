"""Random Forest — the original model, unchanged hyperparameters. Kept as the
first entry in the model comparison table; no tuning (matches the project's
original baseline exactly so the comparison table shows the effect of the
spatial split in isolation from any hyperparameter change)."""

from sklearn.ensemble import RandomForestRegressor

from src import config


def build_model() -> RandomForestRegressor:
    return RandomForestRegressor(**config.RF_PARAMS)
