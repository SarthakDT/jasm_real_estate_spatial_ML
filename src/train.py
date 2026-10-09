"""
Stage 3 — Model training and comparison

Compares Random Forest and XGBoost against three naive baselines on
post-announcement valuations (2022-2026) that have a previous-year change.
Each candidate is scored on the same evaluation splits:

  unseen_plots   5-fold CV grouped by plot: predict plots never seen in training
  unseen_areas   8 KMeans spatial blocks, one held out per fold: predict whole
                 areas never seen in training
  forecast       train on 2022-2024, predict 2025 and 2026 (each one year ahead
                 of the latest change the model is given)
  random_rows    plain 5-fold CV over rows. Reported for reference only: each
                 plot appears once per year, so a random split puts the same
                 plot's other years in training and flatters every model.

Writes data/analysis/model_comparison.csv (one row per split and candidate)
and the forecast-split predictions of every candidate. The better of RF and
XGBoost on the forecast split is saved twice (as the plain sklearn/xgboost
estimator): fitted on 2022-2024 only
(models/forecast_model.pkl, used by Stage 4 to explain forecast errors) and
fitted on all rows (models/best_model.pkl).
"""

import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from src import config
from src.logging_utils import get_logger, status
from src.models import baselines, gbm, rf
from src.spatial_cv import N_BLOCKS, SpatialBlockKFold, assign_spatial_blocks

log = get_logger(__name__)


def _status(msg, *args): status(log, msg, *args)


N_FOLDS = 5

SPLITS = {
    "unseen_plots": "Unseen plots (5-fold, grouped by plot)",
    "unseen_areas": f"Unseen areas ({N_BLOCKS} spatial blocks)",
    "forecast": f"Forecast (train 2022-{config.FORECAST_TRAIN_MAX_YEAR}, predict later years)",
    "random_rows": "Random rows (leaky, reference only)",
}

CANDIDATES = {
    "persistence": "Persistence (last year's change)",
    "muni_year_mean": "Municipality-year mean",
    "training_mean": "Training mean",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
ML_MODELS = ("random_forest", "xgboost")


class FeatureModel:
    """Adapts an sklearn regressor to the fit(df, y) / predict(df) interface of the baselines."""

    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "FeatureModel":
        X = df[config.FEATURE_COLS]
        if self.kind == "xgboost":
            self.estimator_, self.params_ = gbm.build_and_tune(X, y, groups=df["plot_id"])
        else:
            self.estimator_ = rf.build_model().fit(X, y)
            self.params_ = config.RF_PARAMS
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict(df[config.FEATURE_COLS])

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


def _make_candidate(key: str):
    if key in ML_MODELS:
        return FeatureModel(key)
    return baselines.build_baselines()[CANDIDATES[key]]


def load_model_rows() -> pd.DataFrame:
    if not os.path.isfile(config.STAGE2_PATH):
        raise FileNotFoundError(f"Stage 2 output missing: {config.STAGE2_PATH}. Run stage 2 first.")
    df = pd.read_csv(config.STAGE2_PATH, encoding="utf-8-sig")
    missing = (set(config.FEATURE_COLS) | {config.TARGET_COL, "plot_id"}) - set(df.columns)
    if missing:
        raise ValueError(f"stage2.csv is missing columns: {missing}. Re-run stage 2.")
    # Model on post-announcement valuations only; pre-announcement years are
    # the baseline for the fab-impact analysis, not training data.
    df = df[df["post_announcement"] & df["prev_year_pct_change"].notna()].reset_index(drop=True)
    log.info("Model rows: %d (valuations %d-%d with a previous-year change).",
             len(df), df["valuation_year"].min(), df["valuation_year"].max())
    return df


def make_splits(df: pd.DataFrame) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    idx = np.arange(len(df))
    blocks = assign_spatial_blocks(df, n_blocks=N_BLOCKS)
    in_train_years = (df["valuation_year"] <= config.FORECAST_TRAIN_MAX_YEAR).to_numpy()
    return {
        "unseen_plots": list(GroupKFold(n_splits=N_FOLDS).split(idx, groups=df["plot_id"])),
        "unseen_areas": list(SpatialBlockKFold(n_splits=N_BLOCKS).split(idx, groups=blocks)),
        "forecast": [(idx[in_train_years], idx[~in_train_years])],
        "random_rows": list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE).split(idx)),
    }


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "n": len(y_true),
    }


def evaluate_split(df: pd.DataFrame, folds, key: str) -> np.ndarray:
    """Out-of-fold predictions of one candidate (NaN for rows never in a test fold)."""
    y = df[config.TARGET_COL]
    pred = np.full(len(df), np.nan)
    for train_idx, test_idx in folds:
        model = _make_candidate(key).fit(df.iloc[train_idx], y.iloc[train_idx])
        pred[test_idx] = model.predict(df.iloc[test_idx])
    return pred


def run() -> pd.DataFrame:
    log.info("Training and comparing models.")
    df = load_model_rows()
    y = df[config.TARGET_COL]
    splits = make_splits(df)

    rows = []
    forecast_preds = {}
    for split, folds in splits.items():
        for key, name in CANDIDATES.items():
            # Same-year municipality means do not exist yet when forecasting.
            if split == "forecast" and key == "muni_year_mean":
                continue
            pred = evaluate_split(df, folds, key)
            scored = ~np.isnan(pred)
            rows.append({"split": split, "split_label": SPLITS[split], "model_key": key, "model": name,
                         **_metrics(y[scored], pred[scored])})
            if split == "forecast":
                forecast_preds[key] = pred
        log.info("  %s done.", SPLITS[split])

    comparison = pd.DataFrame(rows)
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
    comparison.to_csv(config.MODEL_COMPARISON_PATH, index=False)
    _status("Wrote %s.", config.MODEL_COMPARISON_PATH)

    # Pick the better ML model on the forecast split and save it.
    forecast_r2 = comparison[comparison["split"] == "forecast"].set_index("model_key")["r2"]
    best = max(ML_MODELS, key=lambda k: forecast_r2[k])
    train_idx, test_idx = splits["forecast"][0]
    forecast_model = FeatureModel(best).fit(df.iloc[train_idx], y.iloc[train_idx])
    final_model = FeatureModel(best).fit(df, y)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(forecast_model.estimator_, config.FORECAST_MODEL_PATH)
    joblib.dump(final_model.estimator_, config.BEST_MODEL_PATH)
    with open(config.BEST_MODEL_NAME_PATH, "w", encoding="utf-8") as fh:
        fh.write(CANDIDATES[best])

    test = df.iloc[test_idx][["point_id", "plot_id", "municipality", "valuation_year", "latitude",
                              "longitude", "distance_to_jasm_km", config.TARGET_COL]].copy()
    test = test.rename(columns={config.TARGET_COL: "actual"})
    for key, pred in forecast_preds.items():
        test[f"pred_{key}"] = pred[test_idx]
    test["predicted"] = test[f"pred_{best}"]
    test["residual"] = test["actual"] - test["predicted"]
    test.to_csv(config.FORECAST_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    _status("Saved %s (best on forecast split) and its forecast-split predictions.", CANDIDATES[best])

    table = comparison.pivot(index="model", columns="split", values="r2")[list(SPLITS)]
    table = table.reindex(list(CANDIDATES.values()))
    print()
    print("  R² by evaluation split")
    print(table.round(3).to_string(na_rep="n/a"))
    print()
    persistence = forecast_r2["persistence"]
    verdict = "beats" if forecast_r2[best] > persistence else "does NOT beat"
    print(f"  Forecast split: {CANDIDATES[best]} R²={forecast_r2[best]:.3f} {verdict} "
          f"persistence R²={persistence:.3f}.")
    return comparison


if __name__ == "__main__":
    run()
