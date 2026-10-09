"""
End-to-end smoke test for the pipeline.

Not a substitute for unit tests on individual transforms: this runs the real
pipeline on the raw data and checks that each stage produces what the next
one expects.
"""

import json
import os

import joblib
import pandas as pd

from src import config, evaluate, features, load_data, train


def test_pipeline_end_to_end():
    load_data.run()
    stage1 = pd.read_csv(config.STAGE1_PATH, encoding="utf-8-sig")
    assert len(stage1) > 0
    assert stage1["valuation_year"].min() == config.YEARS[0] + 1
    assert stage1["point_id"].is_unique

    features.run()
    stage2 = pd.read_csv(config.STAGE2_PATH, encoding="utf-8-sig")
    for col in config.FEATURE_COLS + [config.TARGET_COL]:
        assert col in stage2.columns
    assert "spatial_decay" not in stage2.columns
    assert (stage2["municipality"] == "Kikuyo-cho").any(), (
        "Kikuyo-cho (the JASM fab host municipality, code 43404) should be present."
    )

    comparison = train.run()
    expected = {(split, key) for split in train.SPLITS for key in train.CANDIDATES} - {("forecast", "muni_year_mean")}
    assert set(zip(comparison["split"], comparison["model_key"])) == expected
    assert comparison["r2"].notna().all()
    model = joblib.load(config.BEST_MODEL_PATH)
    assert hasattr(model, "predict")
    preds = pd.read_csv(config.FORECAST_PREDICTIONS_PATH, encoding="utf-8-sig")
    assert set(preds["valuation_year"]) == {y for y in config.YEARS if y > config.FORECAST_TRAIN_MAX_YEAR}

    evaluate.run()
    assert os.path.isfile(config.ANOMALY_PATH)
    assert os.path.isfile(config.CHART_PATH)
    diagnostics = json.load(open(config.SPATIAL_DIAGNOSTICS_PATH, encoding="utf-8"))
    # Observed changes must be spatially autocorrelated: that is why the
    # comparison includes a spatial-block split.
    assert diagnostics["observed_change"]["mean_morans_i"] > 0
    assert diagnostics["observed_change"]["max_p_value"] < 0.05
