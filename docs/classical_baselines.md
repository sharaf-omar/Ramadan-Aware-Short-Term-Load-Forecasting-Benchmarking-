# Classical Baselines — MSTL+ETS and SARIMAX

End-state results for the two classical baselines from proposal §4.1.
Both wrappers were optimized after their first-pass implementations
(commits `95aad8d`, `96787a5`) and the numbers below are from the
optimized versions.

Companion docs: [`tsfm_zero_shot_baseline.md`](tsfm_zero_shot_baseline.md)
(TSFM headline), [`v1_v2_lgbm_delta.md`](v1_v2_lgbm_delta.md) (LightGBM).

## Setup

- Test window: 2024-01-01 .. 2025-03-31 (10,944 forecast hours; 456 days).
- Day-ahead protocol: at issuance `t = first_τ(d) − 24h`, forecast 47
  steps and slice `[23:47]` to align with day `d` hours 0..23. Day d
  hour 0 is therefore a strict t+24 forecast; subsequent hours are
  progressively longer-horizon (up to t+47) extrapolations from the
  same issuance.
- Seed: 0 (classical models are deterministic given the same data).
- LightGBM reference: variant `hijri`, seed 44 (median of 42–46).

## Headline metrics

| Model                       | Rows  | Agg MAE | Agg RMSE | Normal | Ramadan | Heatwave |
|-----------------------------|-------|---------|----------|--------|---------|----------|
| LightGBM (hijri, seed 44)   | 10944 |   979.0 |   1527.1 |  873.5 |   799.9 |   1693.0 |
| Chronos-Bolt-Base L=720     | 10944 |   968.9 |   1630.8 |  904.0 |  1061.0 |   1221.2 |
| **MSTL+ETS hijri**          | 10944 | **1527.5** | **2289.4** | 1371.9 |  **1327.0** |   2522.3 |
| MSTL+ETS nohijri            | 10944 |  1593.3 |   2344.1 | 1370.6 |  1842.7 |   2522.3 |
| SARIMAX hijri               | 10944 |  2485.9 |   3356.2 | 2422.8 |  2250.6 |   3030.9 |
| SARIMAX hijri_plusB         | 10944 |  2485.9 |   3356.2 | 2422.8 |  2250.6 |   3030.9 |
| SARIMAX nohijri             | 10944 |  2525.8 |   3440.8 | 2477.9 |  2255.5 |   3024.1 |

**Headline:** MSTL+ETS (hijri) is the **strongest classical baseline**,
aggregate MAE 1528 vs SARIMAX's best 2486. Both are decisively beaten
by LightGBM and the best TSFM (Chronos-Bolt-Base) — but the two
classical methods bracket the "what a univariate seasonal baseline
can do" ceiling for this dataset. The Hijri exogs help SARIMAX too,
just modestly (−40 MW aggregate); the `hijri_plusB` variant adds the
two structurally-zero Compound features and produces predictions
identical to `hijri` up to 4-decimal aggregate MAE (max element-wise
difference 0.0004 MW).

## Diebold-Mariano tests vs LightGBM-hijri

All comparisons use the unified intersection set (n=10,944), MAE loss,
HAC variance with truncation lag h=24 (24h day-ahead horizon), and
Holm-Bonferroni adjustment across the four classical comparisons.
**DM sign convention:** the implementation returns DM > 0 when model B
(second argument) has lower loss. The LGBM-hijri tests put LGBM as
model A, so negative stats mean LGBM is significantly better.

| Comparison (A vs B)            | DM stat | Raw p     | Holm-adj p | Verdict                       |
|--------------------------------|---------|-----------|-----------|--------------------------------|
| LGBM-hijri vs MSTL+ETS nohijri | −12.72  | <1e-30    | <1e-30    | LGBM significantly better      |
| LGBM-hijri vs MSTL+ETS hijri   | −11.41  | <1e-30    | <1e-30    | LGBM significantly better      |
| LGBM-hijri vs SARIMAX nohijri  | −20.33  | <1e-30    | <1e-30    | LGBM significantly better      |
| LGBM-hijri vs SARIMAX hijri    | −21.02  | <1e-30    | <1e-30    | LGBM significantly better      |

Ramadan-only (n=1,416):

| Comparison (A vs B)            | DM stat | Raw p     |
|--------------------------------|---------|-----------|
| LGBM-hijri vs MSTL+ETS nohijri | −8.64   | <1e-15    |
| LGBM-hijri vs MSTL+ETS hijri   | −4.17   | 3.1e-05   |
| LGBM-hijri vs SARIMAX nohijri  | −6.93   | 4.1e-12   |
| LGBM-hijri vs SARIMAX hijri    | −7.20   | 6.1e-13   |

LGBM-hijri retains a statistically significant edge over the best
classical baseline (MSTL+ETS hijri) even on the Ramadan slice where
classical Hijri-aware models would in principle be most competitive.

### Within-classical: do the Hijri exogs actually help?

| Comparison (A vs B)              | DM stat | p        | Interpretation                  |
|----------------------------------|---------|----------|----------------------------------|
| MSTL+ETS nohijri vs MSTL+ETS hijri | +6.41 | 1.4e-10  | hijri significantly better       |
| SARIMAX  nohijri vs SARIMAX hijri  | +2.35 | 1.9e-02  | hijri better (marginal, p=0.02)  |

(DM > 0 here means the second model — *hijri* — has lower loss.)

**Result:** unlike the TSFMs where Hijri covariates *hurt*
(Ablation A), in classical baselines Hijri exogs help — strongly for
MSTL+ETS (driven almost entirely by the Ramadan-block hourly pattern,
not the regression exogs) and marginally for SARIMAX (a small
exog-regression nudge).

## Hijri variant comparison (MSTL+ETS)

| Variant   | Aggregate MAE | Normal MAE | Ramadan MAE | Heatwave MAE |
|-----------|---------------|------------|-------------|--------------|
| nohijri   |        1593.3 |     1370.6 |      1842.7 |       2522.3 |
| hijri     |        1527.5 |     1371.9 |   **1327.0** |       2522.3 |
| Δ (hijri − nohijri) | **−65.8** | +1.3   | **−515.7 (−28%)** | 0.0 |

The Ramadan-only hourly pattern (computed from the most-recent
contiguous Ramadan block in train+val and used to *replace* the
generic daily seasonal component when forecast issuance falls in
Ramadan) drops Ramadan MAE by 28% with no degradation on Normal or
Heatwave. This is the single most informative classical-baseline
result: it confirms the proposal's central thesis — explicit Hijri
feature engineering helps — works for a purely univariate seasonal
decomposition too, not just tree-based models with rich features.

## Why SARIMAX is the weakest classical

| Horizon (h) | SARIMAX nohijri MAE | MSTL+ETS nohijri MAE |
|-------------|---------------------|-----------------------|
| 24          |               1606 |                  1211 |
| 27          |               1946 |                  1751 |
| 28–30       |          ~3666 (worst) |              ~1685 |
| 47 (end)    |               2716 |                  1291 |

SARIMAX's error peaks at horizons 28–30 — UTC 4–6, which is local 7–9
AM in Turkey — i.e., the steep morning peak ramp. SARIMAX with one
seasonal difference at lag 24 still struggles to predict the *amplitude*
of the daily ramp at long horizons; MSTL+ETS forecasts the seasonal
shape directly from the per-hour lookup so it never accumulates ramp
error.

We initially auto-selected the order via `pmdarima.auto_arima` on
daily-resampled data with weekly seasonality (m=7). It picked
(0,1,3)(1,0,1,24) and gave aggregate MAE 3184 — 21% worse than the
hardcoded Hyndman default. The reason: `d=1` non-seasonal differencing
makes long-horizon forecasts random-walk-like and the 24-step
accumulation amplifies any drift during the morning ramp. We replaced
auto_arima with the Hyndman & Athanasopoulos (FPP3 §9.10) default
**(1,0,1)(0,1,1,24)** — one seasonal difference at lag 24 absorbs the
daily cycle while no non-seasonal differencing preserves level
anchoring. See commit `96787a5`.

## What the optimization passes bought

Both classical models were re-implemented after their naive first-pass
versions revealed pathologies. The before/after numbers (aggregate MAE
on the full test set):

| Model              | Naive | Optimized | Δ      |
|--------------------|-------|-----------|--------|
| MSTL+ETS nohijri   | ~2632 (Normal regime, prior naïve daily-fit version) | 1370.6 (Normal) | −48% |
| MSTL+ETS hijri Ramadan | ~2446 (prior naïve)        | 1327.0          | −46% |
| SARIMAX nohijri    |          3184 (auto_arima)  |          2525.8 | −21% |

MSTL+ETS optimization (commit `95aad8d`):
1. **Holt's linear ETS** (`trend='add'`) for trend trajectory across the
   24h horizon. The naive version returned `ets.forecast(1)[0]` as a
   single value applied to all 24 hours — a flat trend that lost
   within-day level evolution.
2. **Recent 12-week seasonal lookup** instead of last-365-day mean. The
   per-(hour, day-of-week) lookup table is now built from the most
   recent quarter of the MSTL decomposition, so it tracks current
   dynamics rather than being diluted by older patterns.
3. **Most-recent-Ramadan-block** for the `hijri` variant pattern,
   replacing the multi-year detrended mean. This is what drives the
   −28% Ramadan MAE.

SARIMAX optimization (commit `96787a5`):
1. **Day-ahead protocol**: forecast 47 steps from issuance and slice
   `[23:47]`. The naive version misaligned `block[0]` (a t+1 forecast)
   with day-d-hour-0 (which should be t+24).
2. **Hyndman default** (1,0,1)(0,1,1,24) replacing auto_arima-selected
   (0,1,3)(1,0,1,24) — see above.
3. `maxiter=200`, `method='lbfgs'` for more reliable convergence.

## Compute

| Model       | Variant     | Fit  | Predict | Total wall |
|-------------|-------------|------|---------|-----------:|
| MSTL+ETS    | nohijri     | 10.2s |   3.3s |     13.5s  |
| MSTL+ETS    | hijri       | 10.0s |   3.1s |     13.1s  |
| SARIMAX     | nohijri     | 622s  |  7295s |   ~132 min |
| SARIMAX     | hijri       | 686s  |  7073s |   ~129 min |
| SARIMAX     | hijri_plusB | 718s  |  7212s |   ~132 min |

MSTL+ETS is essentially free: one MSTL fit + per-day lookup table
applications. SARIMAX is dominated by per-day Kalman filtering — each
day extends the state to the day's issuance time and then forecasts 47
steps, which with the (0,1,1,24) seasonal state requires Kalman passes
over the full 60k+ hour history per day. We accepted this 2-hour-per-
variant cost for the −21% MAE improvement over the auto_arima default.

## Ablation B is structurally inactive

The `hijri_plusB` variant adds the two interaction features
`ramadan_x_heatwave` and `ramadan_x_temp_above_35`. Both are
**identically zero** across train, val, and test in the 2018–2025
window because the Compound regime (simultaneous Ramadan + heatwave)
never occurs:

```
ramadan_x_heatwave:      train nnz=0  val nnz=0  test nnz=0
ramadan_x_temp_above_35: train nnz=0  val nnz=0  test nnz=0
```

So SARIMAX's `hijri_plusB` fit produces virtually the same coefficient
estimates as `hijri` (the two extra columns are constant, no
information to add to the likelihood), and the aggregate MAE is
identical to 4 decimal places (2485.8735). Element-wise the
predictions differ by at most 0.0004 MW — pure numerical noise from
the two-extra-parameter optimizer path through `lbfgs`. This is a
structural fact about the 2018–2025 Hijri-Gregorian alignment
(Ramadan falls March–June, southern-Turkey heatwaves June–August) —
not a code bug. The same constraint already invalidated proposal
Ablation B for TSFMs and LightGBM; documenting it here for
completeness.

## Files

- `data/predictions/mstl_ets__nohijri__seed0.parquet`
- `data/predictions/mstl_ets__hijri__seed0.parquet`
- `data/predictions/sarimax__nohijri__seed0.parquet`
- `data/predictions/sarimax__hijri__seed0.parquet`
- `data/predictions/sarimax__hijri_plusB__seed0.parquet`

## Reproducibility

```bash
.venv/Scripts/python.exe scripts/run_classical.py --model mstl_ets --variant nohijri
.venv/Scripts/python.exe scripts/run_classical.py --model mstl_ets --variant hijri
.venv/Scripts/python.exe scripts/run_classical.py --model sarimax  --variant nohijri
.venv/Scripts/python.exe scripts/run_classical.py --model sarimax  --variant hijri
.venv/Scripts/python.exe scripts/run_classical.py --model sarimax  --variant hijri_plusB
```

Env stack pinned: Python 3.12, statsmodels (MSTL + SARIMAX from
`statsmodels.tsa.{seasonal, statespace.sarimax, holtwinters}`),
pandas 2.1.4, numpy 1.26.4.
