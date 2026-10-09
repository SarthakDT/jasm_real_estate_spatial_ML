"""
Out-of-time backtest: train on years_since_announcement <= 3 (calendar year
<= 2024) and forecast years_since_announcement in {4, 5} (2025-2026) — both
already sitting in the raw MLIT data. This is what actually answers "should a
planner trust a 2027+ forecast from this model," which in-sample or
spatial-holdout Test R^2 alone cannot: it's a genuine test against years the
model has never seen, not just plots it hasn't seen.

Re-fits a fresh copy of whichever model won Stage 3 (src/train.py) — same
hyperparameters, retrained on the temporal-train-only slice — since the
Stage 3 model itself was fit on a spatial (not temporal) split.
"""

import math
import os

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.logging_utils import get_logger
from src.models.predict_dispatch import predict as model_predict
from src.models.refit_dispatch import refit as model_refit

log = get_logger(__name__)

BEST_MODEL_PATH = os.path.join(config.MODEL_DIR, "best_model.pkl")
BEST_MODEL_NAME_PATH = os.path.join(config.MODEL_DIR, "best_model_name.txt")
TRAIN_MAX_YSA = 3   # calendar year 2024
HOLDOUT_YSAS = (4, 5)  # calendar years 2025, 2026


def run() -> pd.DataFrame:
    if not os.path.isfile(BEST_MODEL_PATH):
        raise FileNotFoundError(f"{BEST_MODEL_PATH} missing — run src/train.py first.")
    if not os.path.isfile(config.STAGE2_PATH):
        raise FileNotFoundError(f"{config.STAGE2_PATH} missing — run src/features.py first.")

    winner_model = joblib.load(BEST_MODEL_PATH)
    model_name = "RandomForest"
    if os.path.isfile(BEST_MODEL_NAME_PATH):
        model_name = open(BEST_MODEL_NAME_PATH, encoding="utf-8").read().strip()

    df = pd.read_csv(config.STAGE2_PATH, encoding="utf-8-sig")
    df = df[df["years_since_announcement"] >= 0].reset_index(drop=True)
    df["calendar_year"] = config.ANNOUNCEMENT_YEAR + df["years_since_announcement"]

    train_mask = df["years_since_announcement"] <= TRAIN_MAX_YSA
    holdout_mask = df["years_since_announcement"].isin(HOLDOUT_YSAS)

    X, y = df[config.FEATURE_COLS], df[config.TARGET_COL]
    X_train, y_train = X[train_mask], y[train_mask]
    X_holdout, y_holdout = X[holdout_mask], y[holdout_mask]

    log.info(
        "Backtest: retraining %s on %d rows (<=%d, i.e. through calendar year %d), "
        "forecasting %d holdout rows (calendar years %s).",
        model_name, len(X_train), TRAIN_MAX_YSA, config.ANNOUNCEMENT_YEAR + TRAIN_MAX_YSA,
        len(X_holdout), sorted(config.ANNOUNCEMENT_YEAR + ysa for ysa in HOLDOUT_YSAS),
    )

    df_train = df.loc[X_train.index]
    df_holdout = df.loc[X_holdout.index]
    backtest_model = model_refit(model_name, winner_model, X_train, y_train, df_train=df_train)

    preds = model_predict(backtest_model, model_name, X_holdout, df_slice=df_holdout)
    result = df.loc[holdout_mask, ["point_id", "municipality", "calendar_year", "years_since_announcement"]].copy()
    result["actual"] = y_holdout.values
    result["predicted"] = preds
    result["residual"] = result["actual"] - result["predicted"]
    result["abs_error"] = result["residual"].abs()
    result.to_csv(config.BACKTEST_PATH, index=False, encoding="utf-8-sig")

    overall = {
        "r2": r2_score(y_holdout, preds),
        "rmse": math.sqrt(mean_squared_error(y_holdout, preds)),
        "mae": mean_absolute_error(y_holdout, preds),
    }
    by_year = result.groupby("calendar_year").apply(
        lambda g: pd.Series({
            "r2": r2_score(g["actual"], g["predicted"]) if len(g) > 1 else float("nan"),
            "rmse": math.sqrt(mean_squared_error(g["actual"], g["predicted"])),
            "mae": mean_absolute_error(g["actual"], g["predicted"]),
            "n": len(g),
        }),
        include_groups=False,
    )

    print()
    print(f"  Backtest model         : {model_name}")
    print(f"  Train rows (<=2024)    : {len(X_train)}")
    print(f"  Holdout rows (2025-26) : {len(X_holdout)}")
    print(f"  Holdout R^2            : {overall['r2']:.4f}")
    print(f"  Holdout RMSE           : {overall['rmse']:.4f} pp")
    print(f"  Holdout MAE            : {overall['mae']:.4f} pp")
    print()
    print("  By calendar year:")
    print(by_year.to_string())
    print()
    print(f"  Saved to {config.BACKTEST_PATH}")

    return result


if __name__ == "__main__":
    run()
