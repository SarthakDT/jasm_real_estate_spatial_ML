"""
End-to-end smoke test for the 4-stage pipeline.

Not a substitute for unit tests on individual transforms — this is a refactor
safety net: run the real pipeline against the tracked raw data and assert the
outputs have the shape/contract every stage expects from its predecessor.
"""

import json
import os

import joblib
import pandas as pd

from src import backtest, config, evaluate, features, load_data, train


def test_pipeline_end_to_end():
    load_data.run()
    assert os.path.isfile(config.STAGE1_PATH)
    stage1 = pd.read_csv(config.STAGE1_PATH, encoding="utf-8-sig")
    assert len(stage1) > 0
    assert stage1["years_since_announcement"].min() >= -2

    features.run()
    assert os.path.isfile(config.STAGE2_PATH)
    assert os.path.isfile(config.STAGE2_GEOPARQUET_PATH)
    stage2 = pd.read_csv(config.STAGE2_PATH, encoding="utf-8-sig")
    for col in config.FEATURE_COLS + [config.TARGET_COL]:
        assert col in stage2.columns
    for col in ("flood_risk_class", "in_landslide_zone", "urban_planning_zone_type"):
        assert col in stage2.columns
    assert stage2.loc[stage2["municipality"] == "Kikuyo-cho"].shape[0] > 0, (
        "MUNI_NAMES regression: Kikuyo-cho (the JASM fab host municipality, code 43404) "
        "should be present in the data."
    )

    result = train.run()
    assert os.path.isfile(config.MODEL_PATH)  # RF, always saved (backward-compatible)
    model = joblib.load(config.MODEL_PATH)
    assert hasattr(model, "predict")
    assert os.path.isfile(config.MODEL_COMPARISON_PATH)
    comparison = pd.read_csv(config.MODEL_COMPARISON_PATH)
    expected_models = {
        "RandomForest", "XGBoost", "GaussianProcess", "MixedEffects",
        "SpatialLag (GM_Lag)", "SpatialError (GM_Error)",
    }
    assert expected_models <= set(comparison["model"])
    assert comparison["winner"].sum() == 1
    best_model_path = os.path.join(config.MODEL_DIR, "best_model.pkl")
    assert os.path.isfile(best_model_path)
    assert result["winner_name"] in (
        "RandomForest", "XGBoost", "GaussianProcess", "MixedEffects", "SpatialError (GM_Error)",
    )

    assert os.path.isfile(train.GP_RESIDUAL_DIAGNOSTICS_PATH)
    gp_diag = json.load(open(train.GP_RESIDUAL_DIAGNOSTICS_PATH, encoding="utf-8"))
    assert "morans_i" in gp_diag

    eval_df = evaluate.run()
    assert os.path.isfile(config.ANOMALY_PATH)
    assert {"actual", "predicted", "residual", "abs_error"}.issubset(eval_df.columns)
    assert os.path.isfile(config.CHART_PATH)
    assert os.path.isfile(config.SPATIAL_DIAGNOSTICS_PATH)
    diagnostics = json.load(open(config.SPATIAL_DIAGNOSTICS_PATH, encoding="utf-8"))
    assert "raw_price_change" in diagnostics
    assert "model_residuals" in diagnostics
    # Raw prices must show significant positive spatial autocorrelation — this is
    # the whole justification for using a spatial-block split instead of a random one.
    assert diagnostics["raw_price_change"]["morans_i"] > 0
    assert diagnostics["raw_price_change"]["p_value"] < 0.05

    backtest_result = backtest.run()
    assert os.path.isfile(config.BACKTEST_PATH)
    assert len(backtest_result) > 0
    assert set(backtest_result["calendar_year"].unique()) <= {2025, 2026}
