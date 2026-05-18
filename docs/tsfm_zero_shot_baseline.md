# TSFM Zero-Shot Baseline — Final Results

End-state cross-architecture comparison of the four proposal TSFMs against the
LightGBM and classical baselines on the Turkish STLF test set
(2024-01-01 to 2025-03-31, 10,944 forecast hours). Reports each TSFM at its
**best context-length** from the Ablation C sweep.

This doc supersedes the Plan 2 baseline draft. Companion docs:
[`docs/tsfm_context_length_sweep.md`](tsfm_context_length_sweep.md) (Ablation
C, full L × model grid), [`docs/tsfm_hijri_covariates.md`](tsfm_hijri_covariates.md)
(Ablation A), [`docs/classical_baselines.md`](classical_baselines.md)
(MSTL+ETS and SARIMAX detail).

## Setup

- Test window: 2024-01-01 .. 2025-03-31 (10,944 hours).
- Single seed = 0 for TSFMs (deterministic zero-shot); seed=44 (median of
  42–46) for LightGBM.
- Models at their best L from the L-sweep:
  - Chronos-Bolt-Base — L=720
  - TimesFM 2.5-200M — L=168
  - Moirai-1.1-R-Small — L=336
  - Time-MoE-200M — L=720
- Hardware substitutions (8GB local VRAM, RTX 4070 Laptop):
  - Chronos-Bolt-**Base** (proposal said Large)
  - TimesFM **2.5-200M** (proposal said 2.0-500M; PyPI's `timesfm` 1.0
    package is broken on Python 3.12; installed 2.0.0 from GitHub HEAD
    which ships 2.5-200M as the current checkpoint)
  - Moirai-1.1-R-**Small** (proposal said Large; patch_size=32)
  - Time-MoE-**200M** (proposal said the sparse "Large" variant)

## Aggregate test metrics (intersection-τ = 10,944 rows)

| Model                       | MAE    | RMSE   |
|-----------------------------|--------|--------|
| **Meta-router v2** (ensemble Normal / LGBM Ramadan / Heatwave ensemble) | **838.8** | — |
| Meta-router (ensemble Normal / LGBM Ramadan / Chronos Heatwave) | 840.9 | — |
| Ensemble (median of top 4 residual-corrected) | 872.4 | — |
| Ensemble (median of top 4 bare/mixed) | 891.4 | — |
| Routed best-per-regime      | 916.0  | — |
| LightGBM (hijri) + residual-h | 940.4  | — |
| Chronos-Bolt-Base L=720 + residual-h | 948.5 | — |
| LightGBM (nohijri) + residual-h | 950.7 | — |
| Time-MoE-200M L=720 + residual-h     | 954.5 | — |
| Chronos-Bolt-Base L=720     | 968.9  | 1630.8 |
| LightGBM (hijri-tuned)      | 979.0  | 1527.1 |
| Time-MoE-200M L=720         | 985.9  | 1620.5 |
| PatchTSMixer L=168 + residual-h | 1045.8 | — |
| TimesFM 2.5-200M L=168 + residual-h  | 1057.5 | — |
| TimesFM 2.5-200M L=168      | 1173.2 | 1848.9 |
| SARIMAX hijri + residual-h  | 1299.3 | — |
| Moirai-1.1-R-Small L=336 + residual-h | 1317.2 | — |
| MSTL+ETS hijri + residual-h | 1364.9 | — |
| MSTL+ETS hijri              | 1527.5 | 2289.4 |
| MSTL+ETS nohijri            | 1593.3 | 2344.1 |
| Moirai-1.1-R-Small L=336    | 1727.1 | 2549.2 |
| SARIMAX hijri               | 2485.9 | 3356.2 |
| SARIMAX nohijri             | 2525.8 | 3440.8 |

Detail on post-hoc residual correction (the `+ residual-h` rows): [`residual_correction.md`](residual_correction.md).
Detail on the ensemble and regime-routing recipes: [`capstone_synthesis.md`](capstone_synthesis.md) §10.

**Headline (updated):** The current champion is **meta-router-v2** at
aggregate MAE **838.8** [95% CI 750.8, 948.7], **−13% vs the original
single-model champion Chronos-Bolt-Base L=720 (968.9).** Meta-router-v2
uses the ensemble-of-residual-corrected-models for Normal hours
(MAE 775), LightGBM-hijri for Ramadan (MAE 800), and an ensemble of
Chronos/Time-MoE bare+residual for Heatwave (MAE 1206). Behind it,
the simpler meta-router v1 lands at 840.9 (Chronos-bare alone for
Heatwave), and the all-regime residual-corrected ensemble at 872.4.
Among single models, **LightGBM-hijri + residual head** (940.4) is the
new leader — post-hoc residual correction rescues even the tuned
tabular incumbent. The original zero-shot single-model finding still
stands: Chronos-Bolt-Base L=720 (968.9) narrowly beats LightGBM-hijri
(979.0) bare-vs-bare.

## Per-regime MAE

Top entries per regime (full 28-model table in [`statistical_appendix.md`](statistical_appendix.md)):

| Model                       | Normal | Ramadan | Heatwave | Compound |
|-----------------------------|--------|---------|----------|----------|
| Ensemble (top 4 residual)   | **775.1** |  947.9  | 1309.1   | (empty)  |
| Ensemble (top 4 mixed)      |  804.2 |  930.2  | 1309.1   | (empty)  |
| LightGBM (hijri) + residual-h | 811.8 |  849.5  | 1693.0   | (empty)  |
| LightGBM (nohijri) + residual-h | 815.6 | 907.2 | 1693.9 | (empty)  |
| Routed best-per-regime      |  878.0 | **799.9** | **1221.2** | (empty)  |
| LightGBM (hijri-tuned)      |  873.5 |  800.0  | 1693.0   | (empty)  |
| Time-MoE-200M L=720 + residual-h | 872.6 | 1076.7 | 1267.6 | (empty)  |
| Chronos-Bolt-Base L=720 + residual-h | 878.0 | 1050.5 | 1221.2 | (empty)  |
| Chronos-Bolt-Base L=720     |  904.0 | 1061.0  | 1221.2   | (empty)  |
| Time-MoE-200M L=720         |  908.8 | 1115.6  | 1267.6   | (empty)  |
| PatchTSMixer L=168 + residual-h | 866.2 | 1190.1 | 1847.4 | (empty)  |
| TimesFM 2.5-200M L=168 + residual-h | 949.5 | 1052.7 | 1624.2 | (empty)  |
| SARIMAX hijri + residual-h  |  972.7 | 1264.6  | 3030.9   | (empty)  |
| TimesFM 2.5-200M L=168      | 1082.5 | 1195.8  | 1624.2   | (empty)  |
| Moirai-1.1-R-Small L=336 + residual-h | 1150.2 | 1322.5 | 2181.4 | (empty)  |
| MSTL+ETS hijri + residual-h | 1179.8 | 1154.3  | 2522.3   | (empty)  |
| MSTL+ETS hijri              | 1371.9 | 1327.0  | 2522.3   | (empty)  |
| Moirai-1.1-R-Small L=336    | 1645.4 | 1695.7  | 2181.4   | (empty)  |
| SARIMAX hijri               | 2422.8 | 2250.6  | 3030.9   | (empty)  |

## Where each model wins

| Regime   | Winner (single model)  | MAE   | Winner (incl. composite) | MAE   |
|----------|------------------------|-------|---------------------------|-------|
| Normal   | LightGBM-hijri         | 873.5 | **Ensemble top-4 median** | **804.2** |
| Ramadan  | LightGBM-hijri         | 800.0 | **Routed best-per-regime** | **799.9** (= LGBM-hijri by construction) |
| Heatwave | Chronos-Bolt-Base L=720 | 1221.2 | (tied — Chronos and Routed)| 1221.2 |
| Aggregate | Chronos-Bolt-Base L=720 | 968.9 | **Ensemble top-4 median** | **891.4** |

Classical baselines never win a regime: MSTL+ETS hijri is the strongest
classical (Ramadan 1327, +66% over LGBM) but ranked #5–6 overall. SARIMAX
ranks last on every regime. Detail: [`classical_baselines.md`](classical_baselines.md).

**The proposal's central regime-conditional hypothesis is confirmed on real
data:**
- **LightGBM dominates Ramadan** (800 MW MAE; closest TSFM is +33% worse).
  Explicit Hijri feature engineering wins on this regime.
- **TSFMs dominate Heatwave** (Chronos at L=720 beats LGBM by **28%**,
  1221 vs 1693). Long-context attention captures weather-load nonlinearity
  better than LightGBM's `temp_above_35` feature.
- **Normal regime is close** across the top 3 (LGBM, Chronos, Time-MoE all
  within ~4%) — a "no clear winner" zone.

## Compound regime stays empty

Ramadan 2018-2025 always fell in March–June, before southern Turkey's
heatwave season (June–August). The proposal's Compound regime has n=0 in
this test window. This is a structural fact about the 2018-2025 Hijri-Gregorian
alignment, not a bug. It will start materializing ~2030 as Ramadan shifts later.

## Ablations summary

- **Ablation A — Hijri covariates** ([detail](tsfm_hijri_covariates.md)):
  Adding Hijri dynamic covariates to TimesFM and Moirai *hurts* (DM significant
  with Holm-Bonferroni p<0.001 on Ramadan for both). Tree-based LightGBM
  succeeds with the same features (Ramadan MAE 898→800, −11%). Conclusion:
  off-the-shelf TSFM covariate paths cannot exploit Hijri features; post-hoc
  residual correction is the principled alternative (deferred).
- **Ablation B — Compound regime interaction**: skipped (Compound regime empty).
- **Ablation C — Context-length sweep** ([detail](tsfm_context_length_sweep.md)):
  Chronos and Time-MoE improve monotonically with L (best at L=720). TimesFM
  and Moirai have non-monotone L curves (best at L=168 and L=336 respectively).
  At L=720, Chronos and Time-MoE both beat LightGBM on Heatwave by 25–28%.

## Wall-clock (RTX 4070 Laptop, bf16)

| Model              | L=336 single run | Full L sweep (4 L) |
|--------------------|------------------|--------------------|
| Chronos-Bolt-Base  |  28 s           | ~114 s             |
| TimesFM 2.5-200M   | 212 s           | ~820 s             |
| Moirai-1.1-R-Small |  46 s           | ~179 s             |
| Time-MoE-200M      | 645 s           | ~50 min (L=720 dominates) |

Total Plan 2+3 GPU time: ~90 min across 21 inference passes (4 models × 4 L
nohijri + 2 models × 1 L hijri).

## Files

- 18 prediction parquets in `data/predictions/`:
  - 16 nohijri at L ∈ {96, 168, 336, 720} × 4 models
  - 2 hijri at L=336 × {TimesFM, Moirai}
  - 15 LightGBM at 3 variants × 5 seeds (no L)

## Pipeline reproducibility

```
.venv/Scripts/python.exe scripts/run_tsfm.py --model chronos --context-length 720
.venv/Scripts/python.exe scripts/run_tsfm.py --model timesfm --context-length 168
.venv/Scripts/python.exe scripts/run_tsfm.py --model moirai  --context-length 336
.venv/Scripts/python.exe scripts/run_tsfm.py --model timemoe --context-length 720
```

Env stack pinned: Python 3.12, torch 2.4.1+cu124, numpy 1.26.4, pandas 2.1.4,
transformers 4.48.3, chronos-forecasting 1.5.2, uni2ts 2.0.0,
timesfm 2.0.0 (from GitHub).
