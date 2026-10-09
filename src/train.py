"""
Stage 3 — Model training and comparison
JASM Spatial ML  |  Kumamoto MLIT Chika-Kouji land-price dataset

Reads data/processed/stage2.csv, trains and compares five models on an
identical spatial-block train/test split — not a plain random split, which
would leak information given how spatially autocorrelated these prices are
(see src/spatial_diagnostics.py):

  - RandomForest        original baseline hyperparameters
  - XGBoost             tuned via spatial-block RandomizedSearchCV
  - GaussianProcess     ARD Matern kernel over projected coordinates + other
                         features — a direct answer to the diagnosed problem
                         that RF's residuals carry MORE spatial
                         autocorrelation than the raw target (see the
                         gp_residual_diagnostics.json check below)
  - MixedEffects        random intercept per municipality (partial pooling),
                         targets the small-sample-per-municipality imbalance
  - SpatialError (GM_Error)  interpretable spatial-econometrics baseline,
                         genuinely evaluated out-of-sample (unlike GM_Lag,
                         see src/models/spatial_baseline.py)

Plus GM_Lag, fit in-sample only (its endogenous spatial-lag term makes
honest out-of-sample prediction a real simultaneity problem — see
src/models/spatial_baseline.py) — reported for its coefficient table, not
competing for "winner."

Writes models/model_comparison.csv, always serializes the Random Forest to
models/jasm_rf_model.pkl (backward-compatible with the existing dashboard
tab), and separately serializes whichever model wins on spatial-held-out
Test R^2 (among the five genuinely out-of-sample candidates) to
models/best_model.pkl, plus a MAPIE conformal-interval wrapper (method swept
across plus/base/minmax, see src/models/uncertainty.py) around the winner to
models/mapie_model.pkl.
"""

import json
import math
import os

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config, geo_utils, spatial_diagnostics
from src.logging_utils import get_logger, success, status
from src.models import gbm, gp, mixed_effects, rf, spatial_baseline, uncertainty
from src.spatial_cv import spatial_train_test_split

log = get_logger(__name__)

def _success(msg, *args): success(log, msg, *args)
def _status(msg, *args):  status(log, msg, *args)


BEST_MODEL_NAME_PATH = os.path.join(config.MODEL_DIR, "best_model_name.txt")
MAPIE_MODEL_PATH = os.path.join(config.MODEL_DIR, "mapie_model.pkl")
GP_RESIDUAL_DIAGNOSTICS_PATH = os.path.join(config.DATA_DIR, "gp_residual_diagnostics.json")
N_SPATIAL_BLOCKS = 8
N_TEST_BLOCKS = 2


# ── Step A: Load & validate Stage 2 input ────────────────────────────────────
def _load_stage2() -> pd.DataFrame:
    if not os.path.isfile(config.STAGE2_PATH):
        raise FileNotFoundError(
            f"Stage 2 output missing: {config.STAGE2_PATH}\n"
            "Run src/features.py first to generate the feature matrix."
        )
    df = pd.read_csv(config.STAGE2_PATH, encoding="utf-8-sig")
    required = set(config.FEATURE_COLS) | {config.TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Stage 2 CSV is missing required columns: {missing}\n"
            "Re-run src/features.py to regenerate a valid stage2.csv."
        )
    log.info("Loaded %d rows from %s.", len(df), config.STAGE2_PATH)

    # stage2.csv now includes pre-announcement transition pairs (added so the
    # fab event-study analysis has a pre-2021 baseline — see
    # src/data_sources/chika_kouji.py). Model training is restricted to the
    # post-announcement window so this refactor doesn't change training data.
    df = df[df["post_announcement"]].reset_index(drop=True)
    log.info("Filtered to post-announcement valuations (2022+): %d rows.", len(df))
    return df


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def _row(model_name, split, train_metrics=None, test_metrics=None, notes="") -> dict:
    return {
        "model": model_name,
        "split": split,
        "train_r2": train_metrics["r2"] if train_metrics else None,
        "test_r2": test_metrics["r2"] if test_metrics else None,
        "test_rmse": test_metrics["rmse"] if test_metrics else None,
        "test_mae": test_metrics["mae"] if test_metrics else None,
        "notes": notes,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run():
    log.info("Training models.")

    df = _load_stage2()
    X = df[config.FEATURE_COLS]
    y = df[config.TARGET_COL]
    _success("Feature Matrix (X) and Target Vector (Y) separated cleanly.")

    # Step B: Spatial-block train/test split (replaces plain train_test_split —
    # see src/spatial_cv.py for why a random split leaks for this dataset).
    train_idx, test_idx, block_ids = spatial_train_test_split(
        df, test_blocks=N_TEST_BLOCKS, n_blocks=N_SPATIAL_BLOCKS
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    block_ids_train = block_ids[train_idx]
    _success(
        "Spatial-block partitioning completed: %d training rows | %d testing rows "
        "(%d of %d spatial blocks held out).",
        len(X_train), len(X_test), N_TEST_BLOCKS, N_SPATIAL_BLOCKS,
    )

    comparison_rows = []
    candidates = {}  # name -> (model, test_metrics) for genuinely out-of-sample models only

    # ── Random Forest (original baseline, unchanged hyperparameters) ─────────
    rf_model = rf.build_model()
    rf_model.fit(X_train, y_train)
    rf_train_metrics = _metrics(y_train, rf_model.predict(X_train))
    rf_test_metrics = _metrics(y_test, rf_model.predict(X_test))
    _success("RandomForestRegressor trained on the spatial-block training split.")
    comparison_rows.append(_row(
        "RandomForest", "spatial-block", rf_train_metrics, rf_test_metrics,
        "Original baseline hyperparameters (n_estimators=100, max_depth=7).",
    ))
    candidates["RandomForest"] = (rf_model, rf_test_metrics)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(rf_model, config.MODEL_PATH)
    _status("Random Forest serialized to %s (backward-compatible path).", config.MODEL_PATH)

    # ── XGBoost (tuned via spatial-block RandomizedSearchCV) ─────────────────
    xgb_model, best_params = gbm.build_and_tune(X_train, y_train, block_ids_train)
    xgb_train_metrics = _metrics(y_train, xgb_model.predict(X_train))
    xgb_test_metrics = _metrics(y_test, xgb_model.predict(X_test))
    _success("XGBRegressor tuned via spatial-block RandomizedSearchCV: best_params=%s", best_params)
    comparison_rows.append(_row(
        "XGBoost", "spatial-block", xgb_train_metrics, xgb_test_metrics,
        f"Tuned params: {best_params}",
    ))
    candidates["XGBoost"] = (xgb_model, xgb_test_metrics)

    # ── Gaussian Process (ARD Matern kernel, spatially aware by construction) ─
    gp_model = gp.build_model()
    gp_model.fit(X_train, y_train)
    gp_train_metrics = _metrics(y_train, gp_model.predict(X_train))
    gp_test_metrics = _metrics(y_test, gp_model.predict(X_test))
    _success("GaussianProcessRegressor (ARD Matern kernel) trained on the spatial-block training split.")
    comparison_rows.append(_row(
        "GaussianProcess", "spatial-block", gp_train_metrics, gp_test_metrics,
        "ARD Matern(nu=1.5) kernel over standardized [x_m, y_m, past_year_price, "
        "valuation_year, distance_to_jasm_km, spatial_decay] + WhiteKernel.",
    ))
    candidates["GaussianProcess"] = (gp_model, gp_test_metrics)

    # Acceptance criterion for this model (independent of whether it wins
    # outright on Test R^2): does its spatial kernel actually reduce residual
    # spatial autocorrelation relative to RF's 0.575?
    gp_test_residual_gdf = geo_utils.to_gdf(df.loc[X_test.index].copy())
    gp_test_residual_gdf["residual"] = y_test.to_numpy() - gp_model.predict(X_test)
    gp_residual_diagnostics = spatial_diagnostics.compute_morans_i(gp_test_residual_gdf, "residual")
    with open(GP_RESIDUAL_DIAGNOSTICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(gp_residual_diagnostics, fh, indent=2, ensure_ascii=False)
    _success("GP residual Moran's I: %s", gp_residual_diagnostics["interpretation"])

    # ── Mixed-effects (random intercept per municipality, partial pooling) ───
    mixed_model = mixed_effects.MixedLMModel()
    mixed_model.fit(df.loc[X_train.index], y_train)
    mixed_train_metrics = _metrics(y_train, mixed_model.predict(df.loc[X_train.index]))
    mixed_test_metrics = _metrics(y_test, mixed_model.predict(df.loc[X_test.index]))
    _success("MixedLM (random intercept per municipality) fit on the spatial-block training split.")
    comparison_rows.append(_row(
        "MixedEffects", "spatial-block", mixed_train_metrics, mixed_test_metrics,
        f"Formula: {mixed_effects.FORMULA}",
    ))
    candidates["MixedEffects"] = (mixed_model, mixed_test_metrics)

    # ── Spatial-lag baseline (interpretable, in-sample only — see module docstring) ─
    try:
        gdf_full = geo_utils.to_gdf(df)
        lag_model = spatial_baseline.fit_spatial_lag(X, y, gdf_full)
        lag_summary = spatial_baseline.summarize(lag_model, kind="lag")
        comparison_rows.append(_row(
            "SpatialLag (GM_Lag)", "in-sample (full data)", None, None,
            f"Pseudo R^2={lag_summary['pseudo_r2']:.4f} (in-sample; not comparable "
            "to held-out Test R^2 above). Full coefficient table logged separately.",
        ))
        _success("Spatial-lag baseline fit: pseudo R^2=%.4f", lag_summary["pseudo_r2"])
        with open(os.path.join(config.MODEL_DIR, "spatial_lag_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(lag_summary, fh, indent=2)
    except Exception as exc:  # spreg is occasionally finicky about near-collinear features
        log.warning("Spatial-lag baseline failed (%s) — continuing without it.", exc)
        comparison_rows.append(_row("SpatialLag (GM_Lag)", "in-sample (full data)", None, None, f"Fit failed: {exc}"))

    # ── Spatial-error baseline (interpretable, genuinely out-of-sample) ──────
    try:
        gdf_train = geo_utils.to_gdf(df.loc[X_train.index])
        error_model = spatial_baseline.fit_spatial_error(X_train, y_train, gdf_train)
        error_summary = spatial_baseline.summarize(error_model, kind="error")
        error_test_pred = spatial_baseline.predict_out_of_sample(error_model, X_test)
        error_test_metrics = _metrics(y_test, error_test_pred)
        error_train_pred = spatial_baseline.predict_out_of_sample(error_model, X_train)
        error_train_metrics = _metrics(y_train, error_train_pred)
        comparison_rows.append(_row(
            "SpatialError (GM_Error)", "spatial-block", error_train_metrics, error_test_metrics,
            f"In-sample pseudo R^2={error_summary['pseudo_r2']:.4f}. Coefficient table logged separately.",
        ))
        candidates["SpatialError (GM_Error)"] = (error_model, error_test_metrics)
        _success("Spatial-error baseline: held-out Test R^2=%.4f (in-sample pseudo R^2=%.4f)",
                 error_test_metrics["r2"], error_summary["pseudo_r2"])
        with open(os.path.join(config.MODEL_DIR, "spatial_error_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(error_summary, fh, indent=2)
    except Exception as exc:
        log.warning("Spatial-error baseline failed (%s) — continuing without it.", exc)
        comparison_rows.append(_row("SpatialError (GM_Error)", "spatial-block", None, None, f"Fit failed: {exc}"))

    # ── Pick the winner among all genuinely out-of-sample candidates ─────────
    winner_name = max(candidates, key=lambda name: candidates[name][1]["r2"])
    winner_model, winner_test_metrics = candidates[winner_name]
    best_model_path = os.path.join(config.MODEL_DIR, "best_model.pkl")
    joblib.dump(winner_model, best_model_path)
    with open(BEST_MODEL_NAME_PATH, "w", encoding="utf-8") as fh:
        fh.write(winner_name)
    _success("Winner: %s (Test R^2=%.4f) among %s. Serialized to %s.",
             winner_name, winner_test_metrics["r2"], list(candidates), best_model_path)

    # ── Conformal prediction intervals around the winner (method swept) ──────
    mapie_model, mapie_sweep = uncertainty.sweep_and_fit(
        winner_model, X_train, y_train, block_ids_train, X_test, y_test
    )
    selected_method = mapie_sweep["selected"]
    cov = mapie_sweep[selected_method]["coverage"]
    joblib.dump(mapie_model, MAPIE_MODEL_PATH)
    with open(os.path.join(config.MODEL_DIR, "mapie_sweep.json"), "w", encoding="utf-8") as fh:
        json.dump(mapie_sweep, fh, indent=2)
    _success(
        "MAPIE conformal intervals fit around %s: method=%s selected from %s, "
        "%.0f%% target coverage, %.1f%% observed on test set.",
        winner_name, selected_method, list(uncertainty.SWEEP_METHODS),
        uncertainty.CONFIDENCE_LEVEL * 100, cov * 100,
    )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["winner"] = comparison_df["model"] == winner_name
    comparison_df.to_csv(config.MODEL_COMPARISON_PATH, index=False)
    _status("Model comparison table saved to %s.", config.MODEL_COMPARISON_PATH)

    # ── Summary diagnostics ───────────────────────────────────────────────────
    print()
    print(comparison_df.to_string(index=False))
    print()
    print(f"  Winner                       : {winner_name}")
    print(f"  MAPIE method (swept)         : {selected_method}")
    print(f"  MAPIE target/actual coverage : {uncertainty.CONFIDENCE_LEVEL:.0%} / {cov:.1%}")
    print()
    print(f"  GP residual Moran's I        : {gp_residual_diagnostics['morans_i']:.4f} "
          f"(p={gp_residual_diagnostics['p_value']:.4f}) — compare against RF's last-measured 0.575")

    return {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "gp_model": gp_model,
        "mixed_model": mixed_model,
        "winner_name": winner_name,
        "winner_model": winner_model,
        "comparison": comparison_df,
        "gp_residual_diagnostics": gp_residual_diagnostics,
    }


if __name__ == "__main__":
    run()
