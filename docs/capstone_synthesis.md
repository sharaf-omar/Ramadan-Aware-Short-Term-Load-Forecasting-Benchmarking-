# Capstone Synthesis — Ramadan-Aware Short-Term Load Forecasting Benchmark

Integrated findings across all seven plans. Sources every claim back to
a prediction parquet, a result doc, or a statistical artifact.

| Companion doc | Scope |
|---|---|
| [`tsfm_zero_shot_baseline.md`](tsfm_zero_shot_baseline.md) | Final cross-model headline |
| [`classical_baselines.md`](classical_baselines.md) | MSTL+ETS + SARIMAX detail |
| [`tsfm_context_length_sweep.md`](tsfm_context_length_sweep.md) | Ablation C — context length |
| [`tsfm_hijri_covariates.md`](tsfm_hijri_covariates.md) | Ablation A — Hijri covariates |
| [`statistical_appendix.md`](statistical_appendix.md) | 95% CIs + full pairwise DM matrices |
| [`deep_analysis.md`](deep_analysis.md) | Per-horizon + diurnal MAE decomposition |
| [`failure_modes.md`](failure_modes.md) | Worst-day analysis per model + universal |

---

## TL;DR for the report

1. **Chronos-Bolt-Base at L=720 is the headline winner** on aggregate MAE
   (968.9 [95% CI 868.8, 1097.9]), narrowly beating LightGBM-hijri
   (979.0 [890.2, 1079.9]) and Time-MoE-200M (985.9 [878.2, 1119.7]).
   The three CIs overlap heavily — the differences are not statistically
   significant on aggregate. The DM stack-ranking is meaningful only on
   regime-stratified slices.
2. **LightGBM dominates Ramadan** (MAE 800), 24-33% better than the best
   TSFM. Explicit Hijri feature engineering wins on this regime — the
   proposal's central hypothesis is confirmed.
3. **TSFMs dominate Heatwave** (Chronos-L720: MAE 1221, 28% better than
   LGBM at 1693). Long-context attention captures weather-load
   non-linearity better than tree-based `temp_above_35` features.
4. **The Compound regime (Ramadan + Heatwave) is empty 2018-2025** —
   Ramadan falls March-June, Turkish heatwaves June-August. Proposal
   Ablation B is structurally inactive. Documented in every results doc.
5. **Adding Hijri covariates *hurts* TSFMs** (DM significant, Holm-adj
   p<0.001) but *helps* every other model class. Classical
   architectures and tree-based models can exploit explicit calendar
   structure; off-the-shelf TSFM covariate paths cannot. Post-hoc
   residual correction (Plan 6) is the principled fix and is deferred.
6. **PatchTSMixer (the deep-learning baseline) is mid-tier** (agg MAE
   ~1551) — better than the classical baselines but solidly worse than
   the LGBM/TSFM cluster. With cross-channel mixing enabled, even the
   Hijri channels failed to materially shift its predictions. Single
   seed; multi-seed grid deferred.
7. **Every model fails on Eid days and New Year's**, with a triple-
   anomaly (Eid al-Adha 2024-06-15: Eid + heatwave start + weekend)
   producing the worst day for 6 of 12 models. These are the
   highest-value targets for a holiday-aware second-stage corrector.

---

## 1. Headline cross-model table (aggregate, n=10,944)

From [`statistical_appendix.md`](statistical_appendix.md).

| Model | MAE [95% CI] | RMSE | Rank |
|---|---|---|---|
| chronos-bolt-L720 | **968.9** [868.8, 1097.9] | 1630.8 | 1 |
| lgbm-hijri (seed 44) | 979.0 [890.2, 1079.9] | 1527.1 | 2 |
| time-moe-200m-L720 | 985.9 [878.2, 1119.7] | 1620.5 | 3 |
| lgbm-nohijri (seed 44) | 1003.3 [907.5, 1113.3] | 1588.0 | 4 |
| timesfm-2.5-L168 | 1173.2 [1057.2, 1313.2] | 1848.9 | 5 |
| mstl_ets-hijri | 1527.5 [1379.5, 1692.9] | 2289.4 | 6 |
| patchtsmixer-nohijri-L168 (seed 42) | 1550.6 [1489.2, 1625.1] | — | 7 |
| patchtsmixer-hijri-L168 (seed 42) | 1552.7 [1491.9, 1620.8] | — | 8 |
| mstl_ets-nohijri | 1593.3 [1448.6, 1759.8] | 2344.1 | 9 |
| moirai-1.1-small-L336 | 1727.1 [1620.3, 1853.5] | 2549.2 | 10 |
| sarimax-hijri | 2485.9 [2344.8, 2648.8] | 3356.2 | 11 |
| sarimax-nohijri | 2525.8 [2373.9, 2708.9] | 3440.8 | 12 |

**CI-overlap clusters:**
- **Top tier (CI≈[870, 1130]):** Chronos-L720, LGBM-hijri, Time-MoE-L720, LGBM-nohijri. All four overlap; no aggregate winner.
- **Mid tier:** TimesFM, MSTL+ETS, PatchTSMixer, Moirai. PatchTSMixer's CI is tightest (~136 wide vs ~200-300 for the others).
- **Bottom tier:** SARIMAX (both variants), well-separated from the rest.

## 2. Per-regime story

| Regime | Winner | MAE | Why |
|---|---|---|---|
| Normal | LGBM-hijri | 873.5 | Tabular features dominate when no regime shift |
| Ramadan | LGBM-hijri | **800.0** | Explicit `is_ramadan`/`day_of_ramadan`/`is_eid` features |
| Heatwave | Chronos-L720 | **1221.2** | Long-context attention >> hand-crafted temp features |
| Compound | — | (empty in 2018-2025) | Hijri-Gregorian misalignment |

**The proposal's regime-conditional thesis is confirmed:** different
architectures win different regimes; no single model dominates
everywhere.

## 3. Per-horizon decomposition (block-forecaster models)

From [`deep_analysis.md`](deep_analysis.md). MAE at horizons h ∈ {1, 24}
from issuance, and the compounding ratio.

| Model | h=1 MAE | h=24 MAE | Ratio |
|---|---|---|---|
| time-moe-200m-L720 | 262 | 986 | 3.76× |
| chronos-bolt-L720 | 369 | 969 | 2.62× |
| timesfm-2.5-L168 | 413 | 1173 | 2.84× |
| moirai-1.1-small-L336 | 871 | 1727 | 1.98× |
| patchtsmixer-nohijri-L168 | 1258 | 1551 | 1.23× |
| patchtsmixer-hijri-L168 | 1270 | 1553 | 1.22× |

**Insight:** PatchTSMixer's direct-prediction architecture gives the
flattest horizon curve, but its h=1 starting point is so much worse
than the autoregressive TSFMs (1258 vs 262 for Time-MoE) that
PatchTSMixer still loses overall. If a downstream use needs *short-horizon*
forecasts (1-6 hours), Time-MoE-200M is the clear best at MAE ~262-641.
If a use needs full 24-hour shape, Chronos-Bolt is the most balanced
(369-969).

## 4. Diurnal failure split

From [`deep_analysis.md`](deep_analysis.md). Each model's worst hour
(UTC; local = UTC + 3):

| Failure cluster | Models | Worst hour | Plausible cause |
|---|---|---|---|
| Morning-ramp (UTC 5-6 = local 8-9 AM) | PatchTSMixer, SARIMAX, Moirai | 05-06 | Steep load ramp from overnight base; differencing-based models can't anchor the level |
| Afternoon-peak (UTC 10-11 = local 13-14) | LGBM, Chronos, Time-MoE, TimesFM, MSTL+ETS | 10-11 | Weather-driven afternoon peak that the model under/over-shoots |

PatchTSMixer's morning-ramp peak MAE is 4498 — the largest single-hour
MAE across all 12 models. This is the model's single largest weakness.

## 5. Failure days everyone shares

From [`failure_modes.md`](failure_modes.md). The 5 hardest days across
all 12 models (mean MAE across models):

| Date | Mean MAE | Anomaly type |
|---|---|---|
| 2024-06-15 | 7708 | Eid al-Adha + heatwave start + weekend |
| 2024-04-10 | 6400 | Eid al-Fitr |
| 2024-01-01 | 5791 | New Year's Day |
| 2025-01-01 | 5747 | New Year's Day |
| 2024-04-09 | 5715 | Eid al-Fitr eve (last day of Ramadan) |

These five days alone account for a disproportionate share of total
test-set error. A second-stage holiday-aware corrector (Plan 6 candidate)
that handles just these calendar events could plausibly halve their
contribution.

## 6. Ablation findings

### Ablation A — Hijri covariates

Two opposite findings depending on model class:

| Model class | Hijri-vs-nohijri effect | Significance |
|---|---|---|
| Tree-based (LGBM) | Helps: −98 MW on Ramadan (-11%) | DM p<0.001 |
| TSFM covariate-capable (TimesFM, Moirai) | **Hurts:** TimesFM +24% Ramadan MAE; Moirai +25% | DM p<0.001 |
| Classical (MSTL+ETS) | Helps: −516 MW Ramadan (-28%) | DM p<1e-10 |
| Classical (SARIMAX) | Helps modestly: −40 MW agg | DM p=0.02 |
| Deep (PatchTSMixer cross-channel) | Indistinguishable: +2 MW agg | not significant |

**Three distinct response classes.** Tree-based and classical models
exploit explicit calendar features. TSFM covariate paths inject the
features as auxiliary channels that distract the attention mechanism.
PatchTSMixer's cross-channel mixing should be able to use them but
doesn't — possibly because a single from-scratch training seed at
L=168 is under-parameterised relative to its feature space.

### Ablation B — Compound (Ramadan × Heatwave)

Structurally inactive. Ramadan falls March-June in 2018-2025; Turkish
heatwaves (southern-region temp ≥35°C × 3 consecutive days) fall
June-August. Compound regime n=0. The two interaction features
(`ramadan_x_heatwave`, `ramadan_x_temp_above_35`) are identically zero
in train/val/test. Variant `hijri_plusB` produces predictions
statistically indistinguishable from `hijri` across all models that
include it (LGBM, SARIMAX, PatchTSMixer). The constraint will start to
relax around 2030 as Ramadan shifts later in the Gregorian calendar.

### Ablation C — Context length sensitivity (4 TSFMs)

Detail in [`tsfm_context_length_sweep.md`](tsfm_context_length_sweep.md).
Two L-response patterns:

- **Monotone with L** (better with more context): Chronos-Bolt and Time-MoE
  both win at L=720.
- **Non-monotone**: TimesFM peaks at L=168 (longer hurts); Moirai peaks
  at L=336.

This affects deployment cost — Chronos-L720 needs ~85 s for the full
sweep but only ~28 s at L=336 for ~12% worse MAE.

## 7. What's still open

| Item | Status | Effort |
|---|---|---|
| Plan 5 deferred: L=336, L=720 PatchTSMixer probe | Open | ~5 h GPU |
| Plan 5 deferred: 5-seed × 3-variant PatchTSMixer grid (currently 1 seed) | Open | ~20 h GPU at L=168 |
| Plan 6: post-hoc residual correction on TSFMs (esp. for Hijri) | Designed not built | 1-2 days |
| Plan 7 sub-task: cross-model synthesis (this doc) | DONE | — |
| Final report .docx (a Word version of this synthesis) | Open | 0.5-1 day |

## 8. Recommendations for the deployment story

1. **For the proposal-faithful "single model in production":** ship
   LightGBM-hijri. Beats every model except Chronos by <1% on aggregate
   while being orders of magnitude cheaper to retrain and the easiest
   to debug.
2. **For best aggregate accuracy:** ship Chronos-Bolt-Base at L=720. No
   retraining, no feature engineering, zero-shot deployable. Pay a
   modest GPU bill (~28 s per inference call at L=336).
3. **For best heatwave-period accuracy:** ship Chronos-L720; it's 28%
   better than LGBM on this regime.
4. **For Ramadan periods:** ship LightGBM-hijri; explicit feature
   engineering wins decisively (800 vs 1061 for Chronos).
5. **For a "best of both" deployment:** route by regime — Chronos in
   Heatwave, LGBM-hijri in Ramadan and Normal. A simple regime router
   (predicate on `is_ramadan` + 3-day-temp-≥35 sliding window) gives
   the best of each.

## 9. What this benchmark *cannot* answer

- Whether the patterns will hold for 2030+ (Compound regime materializes).
- Whether the Hijri-hurts-TSFM finding generalises beyond the
  HuggingFace covariate path (no fine-tuning was attempted).
- Whether PatchTSMixer's mid-tier rank reflects the architecture or the
  single-seed protocol — needs the 5-seed grid to know for sure.
- The cost-benefit of Plan 6's post-hoc residual correction. Designed
  in the spec; not implemented.

---

*Generated 2026-05-14 by aggregating findings from Plans 1-5 + Plan 7
sub-tasks (a) statistical appendix, (b) deep analysis, (c) failure
modes. All numbers source back to `data/predictions/` parquets and
`data/{statistical_appendix,analysis}/` CSVs.*
