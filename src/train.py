"""
Stage 3 — Model Architecture & Training
JASM Spatial ML  |  Kumamoto MLIT Chika-Kouji land-price dataset

Reads data/processed_stage2.csv, trains a Random Forest Regressor on six
geospatial/temporal features, serializes the model to models/jasm_rf_model.pkl,
and prints evaluation diagnostics.
"""

import logging
import math
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Logging: custom SUCCESS / STATUS levels to match expected health report ───
_SUCCESS = 25
_STATUS  = 26
logging.addLevelName(_SUCCESS, "SUCCESS")
logging.addLevelName(_STATUS,  "STATUS")
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

def _success(msg, *args): log.log(_SUCCESS, msg, *args)
def _status(msg, *args):  log.log(_STATUS,  msg, *args)


# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
INPUT_PATH = os.path.join(DATA_DIR, "processed_stage2.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "jasm_rf_model.pkl")

FEATURE_COLS = [
    "latitude",
    "longitude",
    "past_year_price",
    "years_since_announcement",
    "distance_to_jasm_km",
    "spatial_decay",
]
TARGET_COL = "target_price_pct_change"

RF_PARAMS    = dict(n_estimators=100, max_depth=7, min_samples_split=5, random_state=42)
TEST_SIZE    = 0.25
RANDOM_STATE = 42


# ── Step A: Load & validate Stage 2 input ────────────────────────────────────
def _load_stage2() -> pd.DataFrame:
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(
            f"Stage 2 output missing: {INPUT_PATH}\n"
            "Run src/features.py first to generate the feature matrix."
        )
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    required = set(FEATURE_COLS) | {TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Stage 2 CSV is missing required columns: {missing}\n"
            "Re-run src/features.py to regenerate a valid processed_stage2.csv."
        )
    log.info("Loaded %d rows from %s.", len(df), INPUT_PATH)
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run() -> RandomForestRegressor:
    log.info("Initiating Model Architecture & Training Pipeline...")

    df = _load_stage2()

    # Step B: Feature-target separation
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    _success("Feature Matrix (X) and Target Vector (Y) separated cleanly.")

    # Step C: Deterministic train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    _success(
        "Train/Test partitioning completed: %d training rows | %d testing rows.",
        len(X_train), len(X_test),
    )

    # Step D: Model construction
    model = RandomForestRegressor(**RF_PARAMS)
    _success("RandomForestRegressor ensemble initialized with structural constraints.")

    # Step E: Training
    model.fit(X_train, y_train)
    _success("Model training execution completed.")

    # Step F: Serialization
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    _status(
        "Serialized model brain successfully exported to %s. Ready for Stage 4 Evaluation.",
        MODEL_PATH,
    )

    # ── Step G: Summary diagnostics ───────────────────────────────────────────
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    train_r2 = r2_score(y_train, train_preds)
    test_r2  = r2_score(y_test,  test_preds)
    test_rmse = math.sqrt(mean_squared_error(y_test, test_preds))
    test_mae  = mean_absolute_error(y_test, test_preds)

    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    print()
    print(f"  Train R2   : {train_r2:.4f}")
    print(f"  Test  R2   : {test_r2:.4f}")
    print(f"  Test  RMSE : {test_rmse:.4f} pp")
    print(f"  Test  MAE  : {test_mae:.4f} pp")
    print()
    print("  Feature Importances (descending):")
    for feat, imp in importances:
        print(f"    {feat:<28} {imp:.4f}")
    print()
    print("  Hyperparameters used:")
    for k, v in RF_PARAMS.items():
        print(f"    {k:<20} {v}")

    return model


if __name__ == "__main__":
    run()
