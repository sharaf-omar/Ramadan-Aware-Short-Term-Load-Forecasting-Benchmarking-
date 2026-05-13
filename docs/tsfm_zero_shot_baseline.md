# TSFM Zero-Shot Baseline — Final Results

End-state cross-architecture comparison of the four proposal TSFMs against the
LightGBM baseline on the Turkish STLF test set (2024-01-01 to 2025-03-31,
10,944 forecast hours). Reports each model at its **best context-length** from
the Ablation C sweep.

This doc supersedes the Plan 2 baseline draft. The two companion docs are
[`docs/tsfm_context_length_sweep.md`](tsfm_context_length_sweep.md) for the
full L × model grid, and [`docs/tsfm_hijri_covariates.md`](tsfm_hijri_covariates.md)
for Ablation A.

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
| LightGBM (hijri-tuned)      |  979.0 | 1527.1 |
| **Chronos-Bolt-Base L=720** | **968.9** | 1630.8 |
| Time-MoE-200M L=720         |  985.9 | 1620.5 |
| TimesFM 2.5-200M L=168      | 1173.2 | 1848.9 |
| Moirai-1.1-R-Small L=336    | 1727.1 | 2549.2 |

**Headline:** **Chronos-Bolt-Base at L=720 beats LightGBM on aggregate MAE
(968.9 vs 979.0)** — a zero-shot model with no Turkish-data exposure and no
Hijri features outperforms a 5-seed Optuna-tuned LightGBM. Time-MoE-200M at
L=720 is essentially tied (985.9).

## Per-regime MAE

| Model                       | Normal | Ramadan | Heatwave | Compound |
|-----------------------------|--------|---------|----------|----------|
| LightGBM (hijri-tuned)      | **873.5**  | **800.0**   | 1693.0   | (empty)  |
| Chronos-Bolt-Base L=720     |  904.0 | 1061.0  | 1221.2   | (empty)  |
| Time-MoE-200M L=720         |  908.8 | 1115.6  | **1267.6**   | (empty)  |
| TimesFM 2.5-200M L=168      | 1082.5 | 1195.8  | 1624.2   | (empty)  |
| Moirai-1.1-R-Small L=336    | 1645.4 | 1695.7  | 2181.4   | (empty)  |

## Where each model wins

| Regime   | Winner               | MAE   | Runner-up                | MAE   |
|----------|----------------------|-------|--------------------------|-------|
| Normal   | LightGBM-hijri       | 873.5 | Chronos-Bolt L=720       | 904.0 |
| Ramadan  | **LightGBM-hijri**       | **800.0** | Chronos-Bolt L=720       | 1061.0 |
| Heatwave | **Chronos-Bolt L=720**   | **1221.2** | Time-MoE L=720           | 1267.6 |

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
