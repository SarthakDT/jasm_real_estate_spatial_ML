"""
Naive baselines that any model has to beat to be worth using.

Each baseline follows a small fit(df, y) / predict(df) interface on the full
stage-2 DataFrame, since they need columns (municipality, plot lag) that are
not model features.
"""

import numpy as np
import pandas as pd


class Persistence:
    """This year's change for a plot equals its change last year."""

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "Persistence":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["prev_year_pct_change"].to_numpy()


class GroupMean:
    """
    Mean training change for rows sharing the same `keys` (for example
    municipality and year). Unseen groups fall back to the overall training
    mean.
    """

    def __init__(self, keys: list[str]):
        self.keys = keys

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "GroupMean":
        self.means_ = pd.Series(y.to_numpy(), index=pd.MultiIndex.from_frame(df[self.keys])).groupby(level=self.keys).mean()
        self.fallback_ = float(y.mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(df[self.keys])
        return self.means_.reindex(idx).fillna(self.fallback_).to_numpy()


class TrainingMean:
    """Every plot gets the mean change of the training rows."""

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "TrainingMean":
        self.mean_ = float(y.mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mean_)


def build_baselines() -> dict:
    """Name -> unfitted baseline. Names are what appears in model_comparison.csv."""
    return {
        "Persistence (last year's change)": Persistence(),
        "Municipality-year mean": GroupMean(["municipality", "valuation_year"]),
        "Training mean": TrainingMean(),
    }
