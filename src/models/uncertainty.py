"""Conformal prediction intervals (MAPIE) for the winning point-estimate model,
calibrated using the same spatial-block CV splitter so the intervals are
spatially honest too — a prediction should ship with a range a planner can
trust, not a false-precision point estimate.

`method="plus"` (jackknife+ / CV+) is conservative by construction in
small-N regimes: it guarantees coverage but often overshoots it, which is
exactly what was observed (98.6% actual vs. 90% target). `sweep_and_fit`
below compares "plus" against "base" and "minmax" and keeps whichever meets
the target coverage with the tightest (smallest mean-width) interval —
tightening the interval this way only makes sense once the underlying
point-estimate model is already a good spatial fit (see src/models/gp.py);
retuning intervals before fixing spatial underfit just optimizes around the
symptom, not the cause.

Not yet swept: `conformity_score` alternatives (e.g. `ResidualNormalisedScore`,
which would let interval width scale with local residual variance instead of
a single flat global width) — that needs an auxiliary residual-magnitude
model and is left as documented future work rather than implemented partially.
"""

import numpy as np
from mapie.regression import CrossConformalRegressor
from sklearn.base import clone

from src import config
from src.spatial_cv import SpatialBlockKFold

CONFIDENCE_LEVEL = 0.9
SWEEP_METHODS = ("plus", "base", "minmax")


def fit_conformal(base_model, X_train, y_train, block_ids_train, n_splits: int = 5, method: str = "plus") -> CrossConformalRegressor:
    mapie_model = CrossConformalRegressor(
        estimator=clone(base_model),
        confidence_level=CONFIDENCE_LEVEL,
        method=method,
        cv=SpatialBlockKFold(n_splits=n_splits),
        random_state=config.RANDOM_STATE,
    )
    mapie_model.fit_conformalize(X_train, y_train, groups=block_ids_train)
    return mapie_model


def sweep_and_fit(base_model, X_train, y_train, block_ids_train, X_test, y_test, n_splits: int = 5):
    """Fit a conformal wrapper for each method in SWEEP_METHODS, evaluate
    coverage + mean interval width on X_test/y_test, and return the tightest
    one that still meets CONFIDENCE_LEVEL (falling back to the best available
    if none do). Returns (best_mapie_model, results_by_method: dict).
    """
    results = {}
    candidates = []
    for method in SWEEP_METHODS:
        try:
            mapie_model = fit_conformal(base_model, X_train, y_train, block_ids_train, n_splits=n_splits, method=method)
        except Exception as exc:
            results[method] = {"error": str(exc)}
            continue
        _, lower, upper = predict_with_interval(mapie_model, X_test)
        cov = coverage(y_test, lower, upper)
        mean_width = float(np.mean(upper - lower))
        results[method] = {"coverage": cov, "mean_width": mean_width}
        candidates.append((method, mapie_model, cov, mean_width))

    meeting_target = [c for c in candidates if c[2] >= CONFIDENCE_LEVEL]
    pool = meeting_target if meeting_target else candidates
    best_method, best_model, best_cov, best_width = min(pool, key=lambda c: c[3])
    results["selected"] = best_method
    return best_model, results


def predict_with_interval(mapie_model, X):
    """Returns (point_predictions, lower_bounds, upper_bounds)."""
    y_pred, y_pis = mapie_model.predict_interval(X)
    lower = y_pis[:, 0, 0]
    upper = y_pis[:, 1, 0]
    return y_pred, lower, upper


def coverage(y_true, lower, upper) -> float:
    """Fraction of true values falling within [lower, upper] — should be ~= CONFIDENCE_LEVEL."""
    y_true = y_true.to_numpy() if hasattr(y_true, "to_numpy") else y_true
    return float(((y_true >= lower) & (y_true <= upper)).mean())
