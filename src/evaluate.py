"""
Stage 4 — Validation & Interpretability
JASM Spatial ML  |  Kumamoto MLIT Chika-Kouji land-price dataset

Loads the winning model from Stage 3 (models/best_model.pkl) and
data/processed/stage2.csv, reconstructs the identical spatial-block test
split, computes residuals, exports the top 5% worst predictions to
data/processed/evaluation_anomalies.csv, saves a feature-importance chart,
and runs Moran's I on the residuals (src/spatial_diagnostics.py) to check
whether the winning model still leaves spatial structure on the table.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, geo_utils, spatial_diagnostics
from src.logging_utils import get_logger, success, status
from src.models.predict_dispatch import predict as model_predict
from src.spatial_cv import spatial_train_test_split
from src.train import N_SPATIAL_BLOCKS, N_TEST_BLOCKS

log = get_logger(__name__)

def _success(msg, *args): success(log, msg, *args)
def _status(msg, *args):  status(log, msg, *args)

ANOMALY_PCT = 0.95
BEST_MODEL_PATH = os.path.join(config.MODEL_DIR, "best_model.pkl")
BEST_MODEL_NAME_PATH = os.path.join(config.MODEL_DIR, "best_model_name.txt")


# ── Step A: Load model & data, reconstruct identical spatial test split ─────
def _load_inputs():
    if not os.path.isfile(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model missing: {BEST_MODEL_PATH}\n"
            "Run src/train.py first to generate the serialized model."
        )
    if not os.path.isfile(config.STAGE2_PATH):
        raise FileNotFoundError(
            f"Feature matrix missing: {config.STAGE2_PATH}\n"
            "Run src/features.py first to generate stage2.csv."
        )

    model = joblib.load(BEST_MODEL_PATH)
    model_name = "RandomForest"
    if os.path.isfile(BEST_MODEL_NAME_PATH):
        model_name = open(BEST_MODEL_NAME_PATH, encoding="utf-8").read().strip()

    df = pd.read_csv(config.STAGE2_PATH, encoding="utf-8-sig")

    required = set(config.FEATURE_COLS) | {config.TARGET_COL, "municipality"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"stage2.csv is missing columns: {missing}")

    # Mirror src/train.py's filter: stage2.csv includes pre-announcement rows
    # (for the fab event-study analysis) but the model was trained only on
    # post-announcement valuations (2022+).
    df = df[df["post_announcement"]].reset_index(drop=True)

    X = df[config.FEATURE_COLS]
    y = df[config.TARGET_COL]
    # Identical spatial-block split logic/seeds as src/train.py -> same test set.
    _, test_idx, _ = spatial_train_test_split(df, test_blocks=N_TEST_BLOCKS, n_blocks=N_SPATIAL_BLOCKS)
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    log.info(
        "Loaded %s model and %d rows. Reconstructed spatial test split: %d rows.",
        model_name, len(df), len(X_test),
    )
    return model, model_name, df, X_test, y_test


# ── Step B: Build evaluation matrix ──────────────────────────────────────────
def _build_eval_matrix(model, model_name, df, X_test, y_test) -> pd.DataFrame:
    eval_df = X_test.copy()
    eval_df["municipality"] = df.loc[X_test.index, "municipality"]
    eval_df["actual"]       = y_test
    eval_df["predicted"]    = model_predict(model, model_name, X_test, df_slice=df.loc[X_test.index])
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
    anomalies.to_csv(config.ANOMALY_PATH, index=False, encoding="utf-8-sig")
    return anomalies, threshold


# ── Step E: Feature importance chart ─────────────────────────────────────────
def _plot_feature_importance(model, model_name: str) -> None:
    if not hasattr(model, "feature_importances_"):
        log.warning("Model %s has no feature_importances_ — skipping chart.", model_name)
        return
    importances  = model.feature_importances_
    indices      = np.argsort(importances)           # ascending → highest bar at top
    sorted_feats = [config.FEATURE_COLS[i] for i in indices]
    sorted_imps  = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_feats, sorted_imps, color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"JASM {model_name} — Feature Importances")
    plt.tight_layout()
    plt.savefig(config.CHART_PATH, dpi=150)
    plt.close()


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run() -> pd.DataFrame:
    log.info("Evaluating the selected model.")

    model, model_name, df, X_test, y_test = _load_inputs()

    # Step B
    eval_df = _build_eval_matrix(model, model_name, df, X_test, y_test)

    # Step C: error by municipality and by year
    spatial_slice  = (
        eval_df.groupby("municipality")["abs_error"]
        .mean()
        .sort_values(ascending=False)
    )
    temporal_slice = (
        eval_df.groupby("valuation_year")["abs_error"]
        .mean()
        .sort_values()
    )

    # Step D
    anomalies, threshold = _extract_anomalies(eval_df)
    _success(
        "Anomaly threshold: >= %.2f percentage points.",
        threshold,
    )
    _success("Top 5%% high-variance predictions exported to %s.", config.ANOMALY_PATH)

    # Step E
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    _plot_feature_importance(model, model_name)
    _success("Feature importance graphics generated and saved to %s.", config.CHART_PATH)

    # Step F: Residual spatial autocorrelation — does the winning model leave
    # spatial structure on the table?
    residual_gdf = geo_utils.to_gdf(eval_df)
    spatial_diagnostics.run(residual_df=residual_gdf)
    _success("Residual Moran's I computed — see %s.", config.SPATIAL_DIAGNOSTICS_PATH)


    # ── Diagnostics ───────────────────────────────────────────────────────────
    print()
    print(f"  Winning model        : {model_name}")
    print(f"  Test rows evaluated  : {len(eval_df)}")
    print(f"  Anomaly threshold    : {threshold:.4f} pp")
    print(f"  Anomaly rows (5%)    : {len(anomalies)}")
    print()
    print("  Mean Abs Error by Municipality (top 8):")
    for muni, err in spatial_slice.head(8).items():
        print(f"    {muni:<30} {err:.4f} pp")
    print()
    print("  Mean Abs Error by valuation year:")
    for ysa, err in temporal_slice.items():
        print(f"    {int(ysa)}  {err:.4f} pp")

    return eval_df


if __name__ == "__main__":
    run()
