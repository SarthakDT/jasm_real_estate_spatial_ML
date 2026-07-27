"""
Stage 4 — Validation & Interpretability
JASM Spatial ML  |  Kumamoto MLIT Chika-Kouji land-price dataset

Loads models/jasm_rf_model.pkl and data/processed_stage2.csv, reconstructs
the identical test split from Stage 3, computes residuals, exports the top 5%
worst predictions to data/evaluation_anomalies.csv, and saves a feature
importance chart to models/feature_importance.png.
"""

import logging
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# ── Configuration (mirrors src/train.py exactly) ──────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
INPUT_PATH   = os.path.join(DATA_DIR,  "processed_stage2.csv")
MODEL_PATH   = os.path.join(MODEL_DIR, "jasm_rf_model.pkl")
ANOMALY_PATH = os.path.join(DATA_DIR,  "evaluation_anomalies.csv")
CHART_PATH   = os.path.join(MODEL_DIR, "feature_importance.png")

FEATURE_COLS = [
    "latitude",
    "longitude",
    "past_year_price",
    "years_since_announcement",
    "distance_to_jasm_km",
    "spatial_decay",
]
TARGET_COL   = "target_price_pct_change"
TEST_SIZE    = 0.25
RANDOM_STATE = 42
ANOMALY_PCT  = 0.95


# ── Step A: Load model & data, reconstruct identical test split ───────────────
def _load_inputs():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model missing: {MODEL_PATH}\n"
            "Run src/train.py first to generate the serialized model."
        )
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(
            f"Feature matrix missing: {INPUT_PATH}\n"
            "Run src/features.py first to generate processed_stage2.csv."
        )

    model = joblib.load(MODEL_PATH)
    df    = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    required = set(FEATURE_COLS) | {TARGET_COL, "municipality"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"processed_stage2.csv is missing columns: {missing}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log.info(
        "Loaded model and %d rows. Reconstructed test split: %d rows.",
        len(df), len(X_test),
    )
    return model, df, X_test, y_test


# ── Step B: Build evaluation matrix ──────────────────────────────────────────
def _build_eval_matrix(model, df, X_test, y_test) -> pd.DataFrame:
    eval_df = X_test.copy()
    eval_df["municipality"] = df.loc[X_test.index, "municipality"]
    eval_df["actual"]       = y_test
    eval_df["predicted"]    = model.predict(X_test)
    eval_df["residual"]     = eval_df["actual"] - eval_df["predicted"]
    eval_df["abs_error"]    = eval_df["residual"].abs()
    return eval_df


# ── Step D: Anomaly extraction ────────────────────────────────────────────────
def _extract_anomalies(eval_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    threshold = eval_df["abs_error"].quantile(ANOMALY_PCT)
    anomalies = (
        eval_df[eval_df["abs_error"] >= threshold]
        .sort_values("abs_error", ascending=False)
    )
    anomalies.to_csv(ANOMALY_PATH, index=False, encoding="utf-8-sig")
    return anomalies, threshold


# ── Step E: Feature importance chart ─────────────────────────────────────────
def _plot_feature_importance(model) -> None:
    importances  = model.feature_importances_
    indices      = np.argsort(importances)           # ascending → highest bar at top
    sorted_feats = [FEATURE_COLS[i] for i in indices]
    sorted_imps  = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_feats, sorted_imps, color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title("JASM Random Forest — Feature Importances")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run() -> pd.DataFrame:
    log.info("Initiating Validation & Interpretability Pipeline...")

    model, df, X_test, y_test = _load_inputs()

    # Step B
    eval_df = _build_eval_matrix(model, df, X_test, y_test)
    _success("Comprehensive residual calculations mapped across test matrices.")

    # Step C: spatial & temporal slices (computed now, printed after DoD)
    spatial_slice  = (
        eval_df.groupby("municipality")["abs_error"]
        .mean()
        .sort_values(ascending=False)
    )
    temporal_slice = (
        eval_df.groupby("years_since_announcement")["abs_error"]
        .mean()
        .sort_values()
    )

    # Step D
    anomalies, threshold = _extract_anomalies(eval_df)
    _success(
        "Spatial blind spots evaluated. Anomaly threshold locked at >= %.2f percentage points.",
        threshold,
    )
    _success("Top 5%% high-variance predictions exported to %s.", ANOMALY_PATH)

    # Step E
    os.makedirs(MODEL_DIR, exist_ok=True)
    _plot_feature_importance(model)
    _success("Feature importance graphics generated and saved to %s.", CHART_PATH)

    # Step F: DoD status line
    _status(
        "Model validation complete. Ready for Stage 5 Production Prep & Repository Packaging."
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print()
    print(f"  Test rows evaluated : {len(eval_df)}")
    print(f"  Anomaly threshold   : {threshold:.4f} pp")
    print(f"  Anomaly rows (5%)   : {len(anomalies)}")
    print()
    print("  Mean Abs Error by Municipality (top 8):")
    for muni, err in spatial_slice.head(8).items():
        print(f"    {muni:<30} {err:.4f} pp")
    print()
    print("  Mean Abs Error by Years Since Announcement:")
    for ysa, err in temporal_slice.items():
        print(f"    YSA={int(ysa)}  {err:.4f} pp")

    return eval_df


if __name__ == "__main__":
    run()
