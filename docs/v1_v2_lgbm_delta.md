# LightGBM v1 -> v2 Data Quality Delta

Comparison of LightGBM-Hijri-Tuned test metrics before (v1) and after (v2)
fixing the rolling-feature leakage. v1 rolling features used
`y.shift(1).rolling(24)` which peeked 23 hours past issuance time;
v2 uses `y.shift(24).rolling(24)` which ends exactly at issuance.

v2 also adds the southern-region heatwave detection (`temp_c_south`) to make
the 35 C threshold meaningful (pop-weighted national temp rarely crosses 35 C).

Test set: 2024-01-01 .. 2025-03-31 (10,944 hours).
Reported below: median seed (44), Hijri-Optuna-tuned variant.

## Headline: Test-set aggregate metrics

| Metric | v1 (leaky)    | v2 (clean)    | Δ (v2 − v1) | %Δ      |
|--------|---------------|---------------|-------------|---------|
| MAE    | 790.241       | 979.00        | +188.76     | +23.9%  |
| RMSE   | 1165.975      | 1527.14       | +361.17     | +31.0%  |
| MAPE   | 1.980         | 2.476         | +0.496      | +25.0%  |
| MASE   | 0.476         | 0.591         | +0.115      | +24.2%  |

v1 numbers from `notebooks/lgbm_training.ipynb` cell 20 output (LightGBM —
Hijri (Optuna tuned), pre-refactor). v2 numbers from
`data/predictions/lgbm__hijri__seed44.parquet` after `notebooks/02_lgbm.ipynb`.

**Interpretation.** v2 is ~24% worse than v1. This is the *real* expected
direction once the rolling-feature leakage is removed. v1's rolling-mean and
rolling-std features peeked 23 hours past the issuance time (using `shift(1)`
then `rolling(24)` produces a window covering `[τ-23, τ]`, but the t+24
forecast issued at `t = τ-24` should only see data up to `τ-24`). Those 23
hours of forbidden recent load history were doing significant work — removing
them shifts MAE from 790 -> 979 MW.

This direction is correct. The v1 numbers were inflated by leakage; v2 is
the honest baseline. The final report should cite v2 throughout and document
this fix in the Methods section.

## Per-regime breakdown (v2, seed 44)

### Variant `nohijri` (BASE features only)

| Regime   | n     | MAE     | RMSE    | MAPE    | MASE    |
|----------|-------|---------|---------|---------|---------|
| Normal   | 7,992 |  889.21 | 1430.10 |  2.3795 | 0.5368  |
| Ramadan  | 1,416 |  897.74 | 1320.82 |  2.5253 | 0.5420  |
| Heatwave | 1,536 | 1693.87 | 2391.14 |  3.6830 | 1.0226  |
| Compound | 0     | NaN     | NaN     | NaN     | NaN     |

### Variant `hijri` (BASE + Hijri features)

| Regime   | n     | MAE     | RMSE    | MAPE    | MASE    |
|----------|-------|---------|---------|---------|---------|
| Normal   | 7,992 |  873.51 | 1366.77 |  2.2887 | 0.5273  |
| Ramadan  | 1,416 |  799.94 | 1120.52 |  2.2130 | 0.4829  |
| Heatwave | 1,536 | 1692.96 | 2395.73 |  3.6906 | 1.0220  |
| Compound | 0     | NaN     | NaN     | NaN     | NaN     |

### Hijri-feature delta (nohijri − hijri)

| Regime   | ΔMAE  | Interpretation |
|----------|-------|----------------|
| Normal   |  +15.70 | Hijri features modestly help on Normal hours (some signal carries through) |
| Ramadan  |  +97.79 | **Hijri features substantially help during Ramadan (~11% MAE reduction)** |
| Heatwave |  +0.91  | Hijri features have ~no effect outside Ramadan (expected) |
| Compound | N/A     | Empty regime |

**Headline finding (Ablation A, LGBM only).** Hijri features cut Ramadan MAE by
~98 MW (~11%), while leaving Normal/Heatwave essentially unchanged. This is
exactly the proposal's central hypothesis — Hijri-feature value is
regime-conditional and concentrates on Ramadan hours.

## Ablation B (Compound regime) — empty in this test window

`hijri` and `hijri_plusB` produce **identical metrics** because Compound regime
has n=0. The added `ramadan_x_heatwave` and `ramadan_x_temp_above_35` features
are always 0 in this data (Ramadan 2018–2025 never coincided with a heatwave
in southern Turkey — Ramadan was always in March–June, before heatwave
season). This is a structural fact about the 2018–2025 Hijri calendar window.

Ablation B will become testable when Ramadan shifts into summer (~2030 onward)
or on a longer historical dataset that captures the prior Ramadan-summer
coincidence (~2010–2014).

## Per-seed variance (5 seeds, full test MAE)

| Variant      | Mean MAE | Std MAE | Min     | Max     |
|--------------|----------|---------|---------|---------|
| nohijri      |  999.39  |  8.56   |  984.62 | 1010.14 |
| hijri        |  970.63  |  8.02   |  960.73 |  979.00 |
| hijri_plusB  |  970.63  |  8.02   |  960.73 |  979.00 |

Std/mean ≈ 0.8% — well within the per-regime deltas above (e.g., Ramadan
delta = ~98 MW vs seed std = ~8 MW), so the Ramadan finding is robust to
seed variation by an order of magnitude.

## Optuna result (50 trials on `hijri` variant, seed=42, val=2023)

```
learning_rate     : 0.0335
num_leaves        : 221
max_depth         : 11
min_child_samples : 51
feature_fraction  : 0.6412
bagging_fraction  : 0.8339
lambda_l1         : 8.4e-04
lambda_l2         : 1.0e-04
min_split_gain    : 0.1054
```

Reused across all 15 final-model runs.

## Files

- v1 predictions: not persisted (existed only as in-memory NumPy in the old notebook).
- v2 predictions: `data/predictions/lgbm__{nohijri,hijri,hijri_plusB}__seed{42..46}.parquet` (15 files).
- v2 dataset: `data/processed/final_training_set_v2.csv` + `final_training_set_v2.meta.json`.
