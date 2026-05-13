# TSFM Zero-Shot Baseline Results (L=336, seed=0)

First cross-architecture comparison of three of the four proposal TSFMs
against the LightGBM baseline on the Turkish STLF test set
(2024-01-01 to 2025-03-31, 10,944 forecast hours).

## Setup

- Context length **L = 336 hours** (~2 weeks).
- Forecast horizon: 24 hours. Headline metric is the **t+24 point** of the
  forecast block.
- Single seed (seed=0): TSFMs are deterministic zero-shot.
- All TSFMs in **univariate framing** (no Hijri covariates) for this plan.
  TimesFM and Moirai will be re-run with Hijri dynamic covariates in Plan 3.
- Hardware substitutions due to local 8GB VRAM:
  - Chronos-Bolt-**Base** (proposal said Large)
  - TimesFM 2.5-200M (PyPI's `timesfm` is broken on Python 3.12; installed
    `timesfm 2.0.0` from GitHub HEAD which ships 2.5-200M as the latest checkpoint)
  - Moirai-1.1-R-**Small** (proposal said Large; tested patch_size ∈ {8, 16, 32}
    and committed to 32 which gave the best validation MAE)
- **Time-MoE-200M: deferred to Plan 3.** The bundled remote code uses a
  pre-4.46 transformers Cache API incompatible with our chronos-pinned
  transformers 4.48.3. See `src/models/tsfm/time_moe.py` docstring for details.

## Aggregate metrics (shared timestamps across all 4 models)

| Model              | MAE     | RMSE    | MAPE   | MASE   |
|--------------------|---------|---------|--------|--------|
| LightGBM (hijri)   |  979.00 | 1527.14 | 2.4757 | 0.5910 |
| Chronos-Bolt-Base  | 1008.02 | 1654.77 | 2.7302 | 0.6085 |
| TimesFM 2.5-200M   | 1375.79 | 2040.02 | 3.6847 | 0.8306 |
| Moirai-1.1-R-Small | 1727.14 | 2549.18 | 4.4838 | 1.0427 |

**Headline:** Chronos-Bolt-Base is within **3% of tuned LightGBM** on
aggregate MAE — striking for a zero-shot model that has never seen Turkish
load data and uses no Hijri features. TimesFM-200M and Moirai-Small (the
smallest variants we can run on 8GB VRAM) lag substantially but are stand-ins
for the proposal's "Large" variants that need an A100.

## Per-regime breakdown

### LightGBM (hijri, seed 44)

| Regime   | n     | MAE     | RMSE    | MAPE   | MASE   |
|----------|-------|---------|---------|--------|--------|
| Normal   | 7,992 |  873.51 | 1366.77 | 2.2887 | 0.5273 |
| Ramadan  | 1,416 |  799.94 | 1120.52 | 2.2130 | 0.4829 |
| Heatwave | 1,536 | 1692.96 | 2395.73 | 3.6906 | 1.0220 |
| Compound | 0     | NaN     | NaN     | NaN    | NaN    |

### Chronos-Bolt-Base (nohijri)

| Regime   | n     | MAE     | RMSE    | MAPE   | MASE   |
|----------|-------|---------|---------|--------|--------|
| Normal   | 7,992 |  936.36 | 1567.75 | 2.5958 | 0.5653 |
| Ramadan  | 1,416 | 1062.65 | 1562.51 | 3.0436 | 0.6415 |
| Heatwave | 1,536 | 1330.50 | 2114.45 | 3.1406 | 0.8032 |
| Compound | 0     | NaN     | NaN     | NaN    | NaN    |

### TimesFM 2.5-200M (nohijri)

| Regime   | n     | MAE     | RMSE    | MAPE   | MASE   |
|----------|-------|---------|---------|--------|--------|
| Normal   | 7,992 | 1324.78 | 2011.24 | 3.6134 | 0.7998 |
| Ramadan  | 1,416 | 1313.44 | 1896.45 | 3.7767 | 0.7929 |
| Heatwave | 1,536 | 1698.72 | 2299.88 | 3.9707 | 1.0255 |
| Compound | 0     | NaN     | NaN     | NaN    | NaN    |

### Moirai-1.1-R-Small (nohijri, patch_size=32)

| Regime   | n     | MAE     | RMSE    | MAPE   | MASE   |
|----------|-------|---------|---------|--------|--------|
| Normal   | 7,992 | 1645.41 | 2444.18 | 4.3483 | 0.9933 |
| Ramadan  | 1,416 | 1695.71 | 2419.56 | 4.7346 | 1.0237 |
| Heatwave | 1,536 | 2181.37 | 3133.66 | 4.9577 | 1.3169 |
| Compound | 0     | NaN     | NaN     | NaN    | NaN    |

## Key findings

1. **The proposal's central hypothesis plays out on real data.**
   Chronos-Bolt loses badly to LightGBM on **Ramadan** (1063 vs 800 MW, +33%)
   but **beats LightGBM by 21% on Heatwave** (1330 vs 1693 MW). TSFMs are
   missing the Hijri-calendar regime structure that LightGBM gets from its
   explicit `is_ramadan` / `day_of_ramadan` features, but TSFM pretraining
   captures the weather-driven nonlinearity that LightGBM with `temp_above_35`
   only partially models.

2. **Zero-shot Chronos-Bolt-Base is shockingly close to tuned LightGBM.**
   Aggregate MAE within 3% despite zero exposure to Turkish data, zero feature
   engineering, no Hijri awareness. Validates the proposal's premise that TSFMs
   are competitive STLF candidates worth benchmarking.

3. **TimesFM and Moirai underperform Chronos at the smallest variants.**
   TimesFM 2.5-200M ≈ 1.4x Chronos's MAE; Moirai-Small ≈ 1.7x Chronos's MAE.
   Both natively support dynamic covariates — Plan 3 will add Hijri features
   to test whether the Ramadan gap closes for them.

4. **Heatwave is hard for everyone.** All four models show MAE ≈ 1.3-2.2x
   their Normal-regime MAE on Heatwave hours. Even LGBM with `temp_above_35`
   and `heatwave_x_temp` features doubles its error. This is a publishable
   finding: extreme-heat AC-load nonlinearity is genuinely under-modeled
   across the entire model zoo.

5. **Compound regime empty in 2024-2025.** Ramadan 2018-2025 always fell
   March-June, before southern Turkey's heatwave season (June-August). The
   proposal's Compound regime is hypothetical in this test window; it will
   start materializing in ~2030 as Ramadan shifts later via the Hijri-Gregorian
   drift.

## Runtime on RTX 4070 Laptop 8GB

| Model              | Wall-clock (full 10,944 test preds) |
|--------------------|-------------------------------------|
| Chronos-Bolt-Base  |  28 s    |
| TimesFM 2.5-200M   | 212 s    |
| Moirai-1.1-R-Small |  46 s    |
| **Total**          | **~5 min** |

Plan 3's context-length sweep (L ∈ {96, 168, 336, 720}) × 3 models will
~quadruple this; estimate ~20–30 min for the full sweep.

## Files

- Plan: `docs/superpowers/plans/2026-05-13-tsfm-zero-shot-baseline.md`
- Predictions:
  - `data/predictions/chronos_bolt_base__nohijri__L336__seed0.parquet`
  - `data/predictions/timesfm_2_5__nohijri__L336__seed0.parquet`
  - `data/predictions/moirai_1_1_small__nohijri__L336__seed0.parquet`
- Time-MoE-200M deferred to Plan 3 with a stub wrapper that raises NotImplementedError on `_load()`.
