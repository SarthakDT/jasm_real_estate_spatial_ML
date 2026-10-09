# JASM Spatial ML — Technical & Policy Assessment

**Prepared for:** Kumamoto Prefecture / municipal planning stakeholders (illustrative — no formal engagement exists)
**Subject:** Land-price dynamics near the JASM semiconductor fab, Kikuyo-cho
**Prepared by:** ML engineering review, MLIT open-data & municipal analytics practice
**Status:** Working analysis, pre-decisional — see Limitations (§7) before any of this is used to inform a real decision

---

## 1. Executive summary

This project models how benchmark land prices in Kumamoto Prefecture have moved since the JASM (TSMC/Sony/Denso) semiconductor fab was announced in Kikuyo-cho, using eight years (2019–2026) of MLIT's *Chika-Kouji* (地価公示) benchmark price data. I was asked to review it as I would any analytics work headed toward a prefectural planning audience, and to be direct about what it can and cannot support.

Three things are worth a planner's attention up front:

1. **A pre-existing municipality-code bug meant every "which town" answer in the original codebase was wrong for 11 of 24 municipalities** — including the fab's own host town. This is fixed and verified two independent ways (§3).
2. **The model's original headline accuracy (Test R²=0.626) was inflated by spatial leakage.** Under a methodologically honest spatial holdout, it falls to **R²≈0.13**, and a more powerful model (XGBoost) overfits outright. This is the single most important number in this report, and I want to be clear that it is a *correction*, not a failure of this iteration of work (§4).
3. **The model generalizes across time (forecasting future years) noticeably better than it generalizes across untrained geography** — R²≈0.66 forecasting one year ahead, R²≈0.26 two years ahead (§5). That asymmetry is itself informative for how this tool should and shouldn't be used.

None of this means the underlying question — "is proximity to the fab measurably reshaping land values, and where should growth be steered or discouraged?" — is unanswerable. It means the current single-model, point-estimate framing oversells what a ~1,400-row dataset spanning 24 municipalities can support, and that the project's next phase (displacement-risk indices, hazard-aware siting, an event study, an infrastructure forecast) needs to be built on the corrected, honestly-validated foundation described here, not on the original numbers.

## 2. Background and mandate

Kumamoto Prefecture is living through a real, fast-moving policy problem. JASM's fab in Kikuyo-cho — the largest single foreign direct investment in Kyushu's history — has visibly driven up land prices and rental demand in fab-adjacent municipalities since construction began in 2021, on top of a prefecture that was still recovering from the 2016 Kumamoto earthquake and, more recently, the July 2020 Kuma River flood (77 deaths, ~15,000 buildings damaged). Prefectural and municipal planners are simultaneously managing acute in-migration pressure and long-running disaster-recovery and land-use constraints. This is exactly the kind of problem where a well-built spatial model can genuinely shorten the distance between "we suspect prices are rising near the fab" and "here is where, by how much, and what to do about it" — which is why I think this project is worth doing carefully rather than either abandoning it or shipping it as-is.

The codebase (`jasm_real_estate_spatial_ML`) was originally a single Random Forest regressor on MLIT benchmark price data. The current phase of work extended it with real NLNI hazard/land-use layers and rebuilt the modeling pipeline to be spatially honest; the phase after this one (gated on the user obtaining two free government API keys) adds MLIT transaction-level prices, e-Stat demographics, and PLATEAU building footprints, feeding four policy-facing analysis modules that don't exist yet. This report assesses what's built and validated *today*.

## 3. Data quality: the municipality-code bug

Every one of MLIT's Chika-Kouji records is tagged with a JIS X0402 municipality code (e.g. `43404`). The original codebase carried a hand-built `code → name` lookup table for display purposes. On inspection, **11 of the 24 codes present in the data were mapped to the wrong municipality**, most consequentially:

| Code | Was labeled | Actually is |
|---|---|---|
| `43404` | Mashiki-machi | **Kikuyo-cho** (the fab's actual host municipality) |
| `43443` | Nagomi-machi | Mashiki-machi |
| `43368` | Kikuyo-cho | Nagasu-machi |
| `43403` | Misato-machi | Ozu-machi |
| `43441` | Nankan-machi | Mifune-machi |
| `43442` | Nagasu-machi | Kashima-machi |
| `43482` | Kashima-machi | Ashikita-machi |
| `43213`/`43214`/`43216` | Kamiamakusa-shi / Uki-shi / Aso-shi | Uki-shi / Aso-shi / Koshi-shi (three-way rotation) |

I verified the correction two independent ways before touching anything: against Kumamoto Prefecture's own published municipal-code list, and — because the fab's host municipality is the one fact this entire project cannot afford to get wrong — by running a point-in-polygon check of the fab's own coordinates against NLNI administrative boundary polygons. Both confirm `43404` = Kikuyo-cho.

This does **not** change any model input or prediction — the model was always trained on raw coordinates and codes, never on the display names — so every numeric result in this report is unaffected by the fix. What it *does* change is every table, chart, and sentence that says "Municipality X shows the strongest effect": before the fix, that sentence was reliably naming the wrong town. This is the kind of error that is invisible in a demo (the numbers all still "work") and only surfaces when someone who knows the geography reads the output — which, for a prefectural audience, is exactly and immediately who will read it. I'd treat this as the most important single finding in this review, ahead of anything model-related, because it's the difference between a chart being wrong in a way nobody would notice and a chart being wrong in a way that damages the project's credibility with the one audience it's meant for.

## 4. Modeling approach and the spatial-leakage finding

**The question the model answers**: given a benchmark plot's location, its price the prior year, and its proximity to the fab, predict its year-over-year percentage price change.

**Why the original evaluation was misleading**: benchmark plots that sit near each other are not independent observations — they share unobserved local shocks (a road project, a rezoning, a school catchment change). I confirmed this formally: **Moran's I on raw price-change values is 0.49 (p=0.001)**, i.e. strong, statistically significant positive spatial autocorrelation. A model evaluated with a *plain random* train/test split will end up with training and test points sitting meters apart, effectively testing itself on data it has already half-seen. That is what happened here: the original 75/25 random split reported Test R²=0.626.

**What an honest split shows**: I replaced the random split with a spatial-block holdout — points clustered into 8 geographic blocks via k-means on projected coordinates, with entire blocks (not just points) held out for testing, so the test set is genuinely unseen geography. Under this split:

| Model | Train R² | Test R² | Test RMSE (pp) | Test MAE (pp) |
|---|---|---|---|---|
| Random Forest (original hyperparameters) | 0.856 | **0.130** | 1.54 | 1.05 |
| XGBoost (tuned via spatial-block search) | 0.991 | **−0.375** | 1.94 | 1.44 |

Random Forest wins — decisively, because XGBoost's flexibility becomes a liability rather than an asset with ~1,000 spatially-held-out training rows: it memorizes the training blocks (Train R²=0.99) and does *worse than predicting the mean* on unseen ones. This is a textbook case of a more powerful model losing to a simpler, more regularized one under a small, spatially clustered dataset, and I'd flag it as a genuine methodological finding, not a tuning failure — no reasonable amount of additional tuning fixes "not enough independent geographic variation to learn from."

I also fit a spatial-lag regression (`spreg.GM_Lag`) as an interpretable econometric baseline, not to compete on held-out accuracy but to give a planner an auditable, coefficient-level answer alongside the ML result. In-sample pseudo-R²=0.594. Its most useful output is the coefficient table: `years_since_announcement` is a highly significant positive predictor (β=0.68, p≈8×10⁻⁷⁸), consistent with sustained post-announcement appreciation; the spatial-lag term itself is significant (β=0.80, p≈1×10⁻⁹), reconfirming that neighboring plots' price changes predict each other, independent of distance-to-fab; and `distance_to_jasm_km` is *not* significant once the spatial-lag term is included (p=0.48) — meaning a chunk of what looked like "fab proximity effect" in the original single-model framing may actually be spatially-diffuse regional appreciation that the lag term absorbs. That distinction — "prices near the fab are rising" vs. "prices are rising in a spatially correlated way that happens to be denser near the fab" — matters for how confidently anyone should draw a circle on a map and call it "the fab effect," and it's precisely the kind of ambiguity the planned fab-event-study module (§6, not yet built) needs to resolve with a proper pre/post distance-band design rather than a single cross-sectional regression.

Finally, I checked whether the winning model's residuals still carry spatial structure it failed to capture: **Moran's I on residuals is 0.575 (p=0.001) — actually higher than the raw target's autocorrelation.** In plain terms, Random Forest is not fully absorbing the spatial pattern in the data; there is real headroom for a properly spatial model (kriging, a spatial random-effects model, or a GNN over a proximity graph) to do better than either candidate here.

## 5. Forecast validity: the out-of-time backtest

Test R² answers "does this generalize to unseen *plots*." It does not answer the question a planner actually cares about: "should I trust this model's forecast for *next year*." For that I ran a genuine backtest — retrain on data through 2024, forecast 2025 and 2026 against the actual values already sitting in the raw MLIT releases (both years are real, not synthetic).

| Forecast year | Horizon | R² | RMSE (pp) | MAE (pp) | n |
|---|---|---|---|---|---|
| 2025 | 1 year ahead | 0.656 | 2.91 | 1.56 | 226 |
| 2026 | 2 years ahead | 0.260 | 3.00 | 1.81 | 237 |

This is, in my read, the most policy-usable number in the whole analysis: the model has real, moderate one-year-ahead forecast skill, and that skill degrades — as it should — the further out you ask it to look. It also means the model generalizes across *time* substantially better than it does across untrained *geography* (R²=0.66 one-year-ahead vs. R²=0.13 on unseen plots). Practically: I would trust this model to flag "which municipalities are likely to see continued appreciation next year" a good deal more than I'd trust it to make a confident prediction for a specific parcel in a town it hasn't seen much of.

I also fit 90%-confidence conformal prediction intervals (MAPIE, cross-conformal, calibrated on the same spatial-block splits) around the winning model. Observed coverage on the held-out spatial test set was 98.6% against a 90% target — i.e. the intervals are conservative (wider than strictly necessary) rather than overconfident, which is the direction I'd rather be wrong in for anything a planner might read a number off of, but it also means the intervals as currently calibrated are wider than they need to be and could be tightened in a later pass.

## 6. What's integrated, what isn't yet

**Integrated and validated in this phase:**
- MLIT Chika-Kouji benchmark prices, 2019–2026, all 24 Kumamoto municipalities (~1,650 plot-year transitions, 1,419 in the post-announcement window used for training).
- NLNI flood inundation zones (A31), landslide warning/special-warning zones (A33), urban planning zones (A29), and administrative boundaries (N03) — all real downloads, spatially joined onto every benchmark plot. 653 of 1,649 points (40%) fall in a mapped flood-risk zone of some depth class; 14 points sit inside a designated landslide warning zone.

**Not yet integrated** (the next phase of work, gated on the user registering two free government API keys):
- MLIT reinfolib actual transaction prices (would corroborate the benchmark-point data with real market transactions, and support a formal fab-proximity event study).
- e-Stat population/demographic projections and the Housing & Land Survey (needed for any displacement/affordability index or infrastructure-demand forecast).
- PLATEAU building footprints (needed for building-level, not just point-level, siting analysis).

I mention this explicitly because none of the "policy modules" this project is ultimately aiming at — displacement-risk index, hazard-aware growth siting, infrastructure strain forecast — exist yet. What exists today is a validated, honestly-evaluated *price model* plus a hazard *overlay*. That's a meaningfully more solid foundation than the project had at the start of this phase, but it is not yet the policy-analysis tool described in the project roadmap.

## 7. Limitations and risks (read before using any of this for a real decision)

- **Sample size.** ~1,400 training rows across 24 municipalities is small for spatial ML by any standard; the XGBoost overfitting result in §4 is a direct symptom of this, and any future model choice should be made with this ceiling in mind rather than assuming more model complexity will help.
- **Hazard-data vintage is mixed.** The flood layer (A31) is FY2012 data — the newest nationally-consistent prefecture-scoped release, but 14 years old; the landslide layer is FY2018; urban planning zones are FY2019. None of these are stale enough to be misleading for the kind of coarse risk flagging this project does, but I would not present them to a planner as current without saying so, and a live deployment should track NLNI's release cadence to refresh them.
- **The fab-proximity narrative is not yet a causal claim.** As noted in §4, the spatial-lag model's insignificant `distance_to_jasm_km` coefficient (once a spatial-lag term is included) is a real caution against overstating "the fab caused this" from the current single-snapshot regression. The planned event-study module (pre/post-2021, by distance band, with a proper diff-in-differences design) is the right way to actually answer this, and until it exists, any client-facing claim about "the JASM effect" should be hedged accordingly.
- **This is a portfolio-grade validation, not a production audit.** I ran the checks I'd want to see before trusting a model's numbers (spatial leakage, residual autocorrelation, out-of-time backtest, interval coverage). I did not do the things I'd require before a real prefectural deployment: independent data-provenance sign-off, a security/PII review (not applicable here since all inputs are public benchmark data, but worth stating explicitly), or load-testing the dashboard.
- **No LGWAN-compatible packaging yet.** MLIT's own comparable open-source tool for vacant-house estimation (Project LINKS' "LINKS SOMA," github.com/Project-LINKS-mlitoss/LINKS-SOMA) ships as a self-contained, offline-capable application specifically because most municipal government networks (LGWAN) cannot reach the public internet. If this project is ever meant to run inside an actual government network rather than as a demo, it will need the same treatment — bundled data, no live API calls at runtime — which is a real engineering task, not a checkbox.

## 8. Recommendations

1. **Do not cite the original Test R²=0.626 anywhere going forward.** Use the spatial-block Test R²=0.130 (or, for a forward-looking claim, the 2025 one-year-ahead backtest R²=0.66) and say which one you're using and why — they answer different questions.
2. **Finish the event-study module before making any public "proximity effect" claim.** The spatial-lag coefficients in §4 are a caution flag, not a final answer; a proper pre/post distance-band design with the transaction-price data (once the reinfolib API key is in hand) will settle this more credibly than the current cross-sectional regression can.
3. **Treat the hazard layers as a screening tool, not a siting authority.** Flagging "high price growth + flood risk" areas is useful for a planner's attention; it should not be the sole input to an actual zoning or permitting decision without the underlying, current-vintage hazard maps being consulted directly.
4. **Register the two remaining API keys (reinfolib, e-Stat) to unblock Phase 3.** Everything else in this report holds regardless of that decision, but the displacement-risk and infrastructure-forecast modules cannot be built without them.
5. **If this is ever presented to actual prefectural staff, lead with §1 and §4, not with a feature-importance chart.** A government audience will trust a tool that says "here's what we got wrong and how we fixed it, here's the honest accuracy number" measurably more than one that leads with an impressive-looking but spatially-leaked R².
6. **Keep Random Forest as the production model.** §9 below tried, in good faith, to beat it with more sophisticated spatial models. None did. That's a finding worth respecting, not a reason to keep tuning until one does.

---

## 9. Addendum — robustness-hardening results (attempted model upgrades)

Following the recommendations above, I went looking for a model that would actually resolve the open problem from §4: the winning Random Forest's residuals carry *more* spatial autocorrelation (Moran's I=0.575) than the raw data does (0.49). Three additional candidates were built, evaluated on the identical spatial-block split, and are now permanent rows in `models/model_comparison.csv`:

| Model | Train R² | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| **Random Forest (still the winner)** | 0.856 | **0.130** | 1.54 | 1.05 |
| XGBoost | 0.991 | −0.375 | 1.94 | 1.44 |
| Gaussian Process (ARD Matérn kernel) | 0.953 | **−1.783** | 2.76 | 2.44 |
| Mixed-effects (random intercept per municipality) | 0.506 | −0.544 | 2.06 | 1.64 |
| Spatial-error (GM_Error, genuinely out-of-sample) | 0.379 | 0.021 | 1.64 | 1.27 |
| Spatial-lag (GM_Lag, in-sample only) | — | pseudo R²=0.594 | — | — |

None of them beat Random Forest. The Gaussian Process result is worth dwelling on, because it's the most instructive failure, not just the worst number:

**The GP made the exact problem it was built to fix *worse*.** Its residual Moran's I came in at **0.732** — higher than RF's 0.575, which was itself already higher than the raw data's 0.49. I did not accept this on the first attempt: I re-tuned the kernel's regularization bounds twice (tightening the noise floor and length-scale floor to discourage near-interpolation) and separately removed `distance_to_jasm_km`/`spatial_decay` from its inputs on the hypothesis that their redundancy with the projected coordinates was letting the optimizer find degenerate short length-scales. Every configuration converged to the same qualitative outcome: strongly negative Test R² and residual Moran's I in the 0.68–0.87 range. The optimizer kept pressing against whatever regularization floor I set, which is itself informative — it's not a tuning slip, it's what the marginal-likelihood objective genuinely prefers given this data.

The mechanism, once I stopped trying to tune around it and just looked at it: a stationary spatial kernel has essentially no support for a held-out block that sits far from every training point — which is *the entire point* of a spatial-block split, that's genuinely novel geography. Far from training data, a GP's prediction reverts toward the (post-normalization) global mean, and it does so *uniformly* for every point inside that same held-out block. Uniform, systematically-wrong predictions across a spatial cluster is precisely the mechanism that produces high residual spatial autocorrelation. This is a different failure mode from Random Forest's plain overfitting — RF's axis-aligned tree splits can still exploit non-spatial features (past price, years since announcement) for an unseen region; a short-length-scale GP effectively can't once it's outside its kernel's support. I'm reporting this as a genuine, mechanistically-understood negative result rather than quietly dropping the GP from the comparison, because a government reader deserves to see that "try a fancier spatial model" was actually tried, and why it didn't help — the code and this write-up are both honest about it (see `src/models/gp.py`'s module docstring for the same account, so the finding survives independent of this report).

The mixed-effects model (partial pooling by municipality, meant to help the thinnest groups — Ashikita-machi has only 12 rows) also underperformed RF, though far less dramatically. The spatial-error model is the one genuine bright spot among the new additions: unlike `GM_Lag`, it now produces a real, honestly-computed out-of-sample Test R² (0.021) — small, but positive, and notably the least-overfit of the four non-RF candidates (Train R² 0.379 vs. RF's 0.856), which is exactly what a more constrained, linear, spatially-regularized model should look like on a dataset this size. It's a legitimate secondary reference point for a government reader who wants the more auditable, coefficient-level story, even though it doesn't win on raw accuracy.

I also swept MAPIE's conformal-interval `method` parameter (`plus`/`base`/`minmax`) around the RF winner, per §5's note that the original 98.6% observed coverage against a 90% target was wider than necessary. `base` won the sweep — coverage moved to 97.9%, interval width narrowed about 6% (mean width 9.58pp vs. 10.17pp for `plus`). This confirms the point already made in §5: interval width is downstream of how well the underlying point-estimate model fits, and no amount of conformal recalibration closes a gap that the point-estimate model itself can't close. **Recommendation #6 above is the practical conclusion**: Random Forest, imperfect as it honestly is, stays the production model — not by default, but because five real alternatives were tried and none of them earned the swap.

---

### Appendix A — Reproducing these numbers

```bash
python main.py --all          # regenerates data/spatial_diagnostics.json, models/model_comparison.csv,
                               # data/gp_residual_diagnostics.json, models/mapie_sweep.json
python -m src.backtest        # regenerates data/backtest_results.csv
```

### Appendix B — Random Forest feature importances (winning model)

| Feature | Importance |
|---|---|
| `spatial_decay` | 0.257 |
| `years_since_announcement` | 0.219 |
| `distance_to_jasm_km` | 0.198 |
| `past_year_price` | 0.169 |
| `latitude` | 0.081 |
| `longitude` | 0.075 |

### Appendix C — Highest-residual plots (spatial-block test set)

The two largest single-plot errors are both in Yatsushiro-shi (a coastal city ~45 km from the fab, actual +6.2pp / +5.6pp vs. predicted ≈0pp in successive years) — worth a manual look before the next iteration, since a repeated large miss in one municipality more often points to a local event the model has no feature for (a new road, a local rezoning) than to random noise.

*This report reflects the state of the codebase as of the writing session in which it was produced. Regenerate the tables in Appendix A before relying on any number here after further changes to the pipeline.*
