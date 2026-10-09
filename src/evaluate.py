"""
Stage 4 — Forecast error analysis

Uses the forecast split from Stage 3 (trained on 2022-2024, predicting 2025
and 2026) to explain where the selected model goes wrong:

  - error by municipality and by year, next to the persistence baseline
  - the worst 5% of forecasts, exported for inspection
  - permutation importance on the forecast years (how much R² drops when a
    feature is shuffled), which unlike impurity importance is measured on
    data the model did not train on
  - Moran's I of the forecast residuals, to see whether errors cluster in space
"""

import os

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config, plot_style, spatial_diagnostics
from src.logging_utils import get_logger, status
from src.train import load_model_rows

log = get_logger(__name__)


def _status(msg, *args): status(log, msg, *args)


ANOMALY_QUANTILE = 0.95

FEATURE_LABELS = {
    "prev_year_pct_change": "Last year's % change",
    "distance_to_jasm_km": "Distance to fab (km)",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "past_year_price": "Last year's price (¥/m²)",
    "valuation_year": "Valuation year",
    "lot_area_m2": "Lot area (m²)",
    "station_distance_m": "Distance to station (m)",
    "road_width_m": "Front road width (m)",
    "far": "Floor-area ratio (%)",
    "land_use_code": "Land use",
    "zoning_group": "Zoning",
    "planning_area_code": "Planning area",
    "flood_risk_class": "Flood risk class",
    "in_landslide_zone": "In landslide zone",
}


def _load_inputs():
    for path in (config.FORECAST_PREDICTIONS_PATH, config.FORECAST_MODEL_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} missing. Run stage 3 first.")
    preds = pd.read_csv(config.FORECAST_PREDICTIONS_PATH, encoding="utf-8-sig")
    model = joblib.load(config.FORECAST_MODEL_PATH)
    model_name = open(config.BEST_MODEL_NAME_PATH, encoding="utf-8").read().strip()
    return preds, model, model_name


def _error_tables(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    errors = preds.assign(
        model_abs_error=(preds["actual"] - preds["predicted"]).abs(),
        persistence_abs_error=(preds["actual"] - preds["pred_persistence"]).abs(),
    )
    cols = ["model_abs_error", "persistence_abs_error"]
    by_muni = errors.groupby("municipality")[cols].mean().assign(n=errors.groupby("municipality").size())
    by_year = errors.groupby("valuation_year")[cols].mean().assign(n=errors.groupby("valuation_year").size())
    return by_muni.sort_values("model_abs_error", ascending=False), by_year


def _plot_permutation_importance(model, model_name: str) -> pd.Series:
    df = load_model_rows()
    test = df[df["valuation_year"] > config.FORECAST_TRAIN_MAX_YEAR]
    result = permutation_importance(
        model, test[config.FEATURE_COLS], test[config.TARGET_COL],
        n_repeats=20, random_state=config.RANDOM_STATE, scoring="r2",
    )
    imp = pd.DataFrame({"mean": result.importances_mean, "std": result.importances_std},
                       index=[FEATURE_LABELS.get(c, c) for c in config.FEATURE_COLS]).sort_values("mean")

    fig, ax = plot_style.figure(8, 5.5)
    ax.barh(imp.index, imp["mean"], xerr=imp["std"], color=plot_style.SERIES, height=0.6,
            error_kw={"ecolor": plot_style.TEXT_SECONDARY, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=plot_style.TEXT_SECONDARY, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=plot_style.GRID, linewidth=0.8)
    ax.set_xlabel("Drop in forecast R² when the feature is shuffled")
    plot_style.title(ax, f"What drives the {model_name} forecast",
                     f"Permutation importance on the {test['valuation_year'].min()}–{test['valuation_year'].max()} "
                     "forecasts, ±1 sd over 20 shuffles")
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plot_style.save(fig, config.CHART_PATH)
    return imp["mean"].sort_values(ascending=False)


def run() -> pd.DataFrame:
    log.info("Analysing forecast errors.")
    preds, model, model_name = _load_inputs()

    preds["abs_error"] = preds["residual"].abs()
    threshold = preds["abs_error"].quantile(ANOMALY_QUANTILE)
    anomalies = preds[preds["abs_error"] >= threshold].sort_values("abs_error", ascending=False)
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
    anomalies.to_csv(config.ANOMALY_PATH, index=False, encoding="utf-8-sig")
    _status("Wrote the worst %d forecasts (|error| >= %.2f pp) to %s.", len(anomalies), threshold, config.ANOMALY_PATH)

    by_muni, by_year = _error_tables(preds)
    importance = _plot_permutation_importance(model, model_name)
    _status("Wrote %s.", config.CHART_PATH)

    diagnostics = spatial_diagnostics.run(residual_df=preds)

    print()
    print(f"  Model: {model_name}, forecast rows: {len(preds)}")
    print()
    print("  Mean absolute error by year (pp)")
    print(by_year.round(2).to_string())
    print()
    print("  Municipalities with the largest model errors (pp)")
    print(by_muni.head(8).round(2).to_string())
    print()
    print("  Permutation importance (drop in R²)")
    print(importance.round(3).to_string())
    print()
    obs, res = diagnostics["observed_change"], diagnostics["forecast_residuals"]
    print(f"  Moran's I, observed change (mean over years): {obs['mean_morans_i']:.3f}")
    print(f"  Moran's I, forecast residuals (mean over years): {res['mean_morans_i']:.3f} "
          f"(max p={res['max_p_value']:.3f})")
    return preds


if __name__ == "__main__":
    run()
