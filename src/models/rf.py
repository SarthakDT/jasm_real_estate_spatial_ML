"""Random Forest with fixed, untuned hyperparameters (config.RF_PARAMS)."""

from sklearn.ensemble import RandomForestRegressor

from src import config


def build_model() -> RandomForestRegressor:
    return RandomForestRegressor(**config.RF_PARAMS)
