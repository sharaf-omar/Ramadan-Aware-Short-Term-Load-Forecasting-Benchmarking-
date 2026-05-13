# TSFM Ablation A: Hijri Dynamic Covariates

Tests whether feeding `is_ramadan`, `day_of_ramadan`, `is_eid`, and `temp_c`
as dynamic real covariates over both context and horizon improves TSFM
forecasts on covariate-capable models. Chronos-Bolt and Time-MoE are
univariate-architecture; their Hijri ablation requires post-hoc residual
correction (deferred to a future plan).

## Setup

- 2 TSFMs: TimesFM 2.5-200M, Moirai-1.1-R-Small.
- L = 336, single seed = 0.
- Covariates: `is_ramadan` (binary), `day_of_ramadan` (1–30 or 0),
  `is_eid` (binary), `temp_c` (continuous). All known at issuance time
  per proposal §2 (Hijri dates are deterministic; temperature treated as
  known weather forecast).
- TimesFM implementation: `forecast_with_covariates(xreg_mode="xreg + timesfm")`;
  point estimate is mean across the 10-quantile xreg output.
- Moirai implementation: `MoiraiForecast` with `feat_dynamic_real_dim=4` and
  `past_feat_dynamic_real_dim=4`.

## Per-regime MAE comparison

### TimesFM 2.5-200M (n=10,944 on intersection-τ)

| Regime   | nohijri MAE | hijri MAE | ΔMAE (hijri − nohijri) | DM stat | p_holm |
|----------|-------------|-----------|------------------------|---------|--------|
| Normal   |     1324.78 |   1360.64 |                 +35.9  |   −1.20 | 0.459  |
| Ramadan  |     1313.44 |   1493.30 |                **+179.9**  |   **−4.77** | **0.0000*** |
| Heatwave |     1698.72 |   1697.67 |                  −1.1  |   +0.03 | 0.973  |

### Moirai-1.1-R-Small (n=10,944 on intersection-τ)

| Regime   | nohijri MAE | hijri MAE | ΔMAE (hijri − nohijri) | DM stat | p_holm |
|----------|-------------|-----------|------------------------|---------|--------|
| Normal   |     1645.41 |   2045.24 |                **+399.8**  |  **−12.05** | **0.0000*** |
| Ramadan  |     1695.71 |   2121.12 |                **+425.4**  |   **−4.96** | **0.0000*** |
| Heatwave |     2181.37 |   2322.15 |                 +140.8 |   −1.75 | 0.239  |

DM convention: positive stat means hijri is *better* than nohijri. All
significant deltas in our results have **negative** stats — hijri is
**worse**. Holm-Bonferroni adjusted within each test family (6 tests total).

## Headline finding

**Adding Hijri dynamic covariates makes both covariate-capable TSFMs WORSE
across Normal and Ramadan regimes, statistically significantly so.**

- TimesFM Ramadan MAE: 1313 → 1493 (+14%), p_holm < 0.001.
- Moirai Normal MAE: 1645 → 2045 (+24%), p_holm < 0.001.
- Moirai Ramadan MAE: 1696 → 2121 (+25%), p_holm < 0.001.

This is the opposite of the proposal's expected direction (Hijri features
should reduce Ramadan error, per the LightGBM result where hijri reduces
Ramadan MAE by 98 MW).

## Interpretation

Two hypotheses for why TSFM Hijri covariates hurt instead of help:

1. **Linear xreg insufficiency (TimesFM).** TimesFM 2.5's covariate path
   in `forecast_with_covariates(xreg_mode="xreg + timesfm")` fits a linear
   regression with the 4 covariates as features. The Ramadan effect is
   strongly nonlinear (iftar/suhoor hour-of-day spikes, weekday/weekend
   collapse) and a linear fit ADDS noise rather than reducing it. The
   ensemble averaging across 10 quantile heads further smooths away the
   regime-specific signal.

2. **Small-model covariate capacity (Moirai).** Moirai-Small (~14M params)
   has limited capacity to use the covariate embedding well in zero-shot
   mode. The covariate channels likely introduce input-distribution shift
   (the model was pretrained on diverse covariate channels but never on
   `is_ramadan`-style binary indicators specifically) that the small model
   can't compensate for. Moirai-Large (deferred — needs A100) may behave
   differently.

## Comparison: LightGBM Ablation A succeeds

LightGBM (which has explicit feature engineering and a tree-based model that
can carve out regime-specific subspaces) achieves the *opposite* direction:
- LGBM Ramadan MAE: nohijri 898 → hijri 800 (**−11%**, ΔMAE = −98 MW).

The contrast between LGBM's success and TSFMs' failure on the same Hijri
covariates is itself a finding: regime-conditional features need a model
that can multiplex behavior conditionally (trees, deep networks with
regime-gating). Linear-residual heads and small attention encoders cannot
exploit these features as effectively as the underlying univariate forecasts.

## Implications for the report

1. **Cite the LGBM-TSFM asymmetry**: tree-based regime-aware features beat
   off-the-shelf TSFM covariate paths for Ramadan forecasting.
2. **Future work: post-hoc residual correction** (Plan 5) — train a
   LightGBM residual head on each TSFM's prediction errors with Hijri
   features as input. This is the principled way to add regime awareness
   without forcing the TSFM's internal covariate path to do it.
3. **Architecture matters for covariate utility**: even within the
   "covariate-capable" group, TimesFM's linear-xreg and Moirai's
   attention-encoded covariate paths react very differently. Document
   both as architectural variation, not failures.

## Files

- `data/predictions/timesfm_2_5__nohijri__L336__seed0.parquet`
- `data/predictions/timesfm_2_5__hijri__L336__seed0.parquet`
- `data/predictions/moirai_1_1_small__nohijri__L336__seed0.parquet`
- `data/predictions/moirai_1_1_small__hijri__L336__seed0.parquet`
