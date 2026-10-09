"""
Hierarchical/partial-pooling model: a random intercept per municipality via
statsmodels MixedLM. Complementary to the GP (src/models/gp.py, which models
point-to-point spatial autocorrelation) — this instead directly targets the
small-sample-per-municipality imbalance confirmed in the actual data
(Ashikita-machi=12 rows, several more under 30, vs. Kumamoto-shi
Chuo-ku=275): thin municipalities' estimates get shrunk toward the overall
mean instead of overfitting on a handful of points.

Explicitly not PyMC/bambi: full Bayesian MCMC is a materially heavier
dependency/maintenance burden (convergence diagnostics, R-hat, divergences
a small team now owns forever) than this project's scale justifies.
MixedLM's REML point-estimate fit gets most of the practical shrinkage
benefit at effectively zero added cost — statsmodels is already pinned.

Not a generic sklearn-style fit(X, y)/predict(X) model, because it needs the
grouping column alongside the feature columns — see src/train.py for how
it's wired into the comparison loop (same pattern already used for the
spatial-lag baseline).
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src import config

FORMULA = (
    "target_price_pct_change ~ past_year_price + years_since_announcement "
    "+ distance_to_jasm_km + spatial_decay"
)
GROUP_COL = "municipality"


class MixedLMModel:
    def __init__(self, formula: str = FORMULA, group_col: str = GROUP_COL):
        self.formula = formula
        self.group_col = group_col
        self.result_ = None
        self._group_effects: dict[str, float] = {}

    def fit(self, df_slice: pd.DataFrame, y: pd.Series) -> "MixedLMModel":
        data = df_slice[list(config.FEATURE_COLS) + [self.group_col]].copy()
        data[config.TARGET_COL] = y.to_numpy()

        model = smf.mixedlm(self.formula, data, groups=data[self.group_col])
        self.result_ = model.fit(reml=True)

        # Single random intercept per group (no re_formula given) -> each
        # group's random_effects Series has exactly one entry; read it
        # positionally since statsmodels' default label for it isn't
        # guaranteed across versions.
        self._group_effects = {
            group: float(effects.iloc[0])
            for group, effects in self.result_.random_effects.items()
        }
        return self

    def predict(self, df_slice: pd.DataFrame) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("MixedLMModel must be fit() before predict().")
        X = df_slice[list(config.FEATURE_COLS)]
        fixed_effects_pred = self.result_.predict(X).to_numpy()
        # Unseen groups (e.g. a municipality entirely held out in the
        # spatial-block test split) fall back to 0.0 -- i.e. the pooled
        # fixed-effects-only prediction, which is exactly the point of
        # partial pooling for thin/unseen groups.
        random_intercepts = df_slice[self.group_col].map(
            lambda g: self._group_effects.get(g, 0.0)
        ).to_numpy()
        return fixed_effects_pred + random_intercepts
