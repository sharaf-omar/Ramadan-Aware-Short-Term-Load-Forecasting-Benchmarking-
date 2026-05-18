# Plan 6 — Post-Hoc Residual Correction Design Spec

**Date:** 2026-05-14
**Branch:** new `plan-6-residual` off whichever branch holds the
finished Plan 5 + 7 work.

## Goal

Test whether a small **post-hoc residual head** (LightGBM fit on
`y_true - y_pred_TSFM` with weather + Hijri features) can rescue the
two TSFMs that cannot ingest Hijri covariates natively (Chronos and
Time-MoE), and quantify the effect on the two TSFMs that *can*
ingest them via the HuggingFace covariate path (TimesFM and Moirai —
where Plan 3 showed that path *hurts*).

The proposal §6 explicitly lists this as the principled alternative
to in-band covariate injection. We deferred it through Plans 2-5
because we wanted the bare TSFM headline first.

## Hypothesis

A LightGBM residual head trained on weather + Hijri features should:
1. **Rescue Chronos-L720 and Time-MoE-L720 on Ramadan** (the regime
   where bare TSFMs lose by 24-33% to LightGBM-hijri).
2. **Not materially hurt Heatwave** (the regime where bare TSFMs win).
3. **Outperform the in-band covariate path** on TimesFM and Moirai —
   i.e., post-hoc correction works *better* than the HuggingFace
   covariate ingestion that Plan 3 showed hurts.

If confirmed, the headline becomes: *"zero-shot TSFM + post-hoc
LightGBM residual head beats every alternative on aggregate and on
Ramadan."*

## Scope

| Item | Count |
|---|---|
| TSFMs to correct | 4 (Chronos-L720, Time-MoE-L720, TimesFM-L168, Moirai-L336) |
| Residual variants | 2 (`nohijri` features only, full `hijri` features) |
| Resulting test parquets | 8 (4 TSFMs × 2 residual variants) |
| Bare-TSFM baselines | already on disk (4 parquets, no rerun) |
| Seeds | 1 per residual variant (LGBM converges deterministically with `n_estimators` fixed; multi-seed deferred) |

Excluded: PatchTSMixer (deep model already trained, not the central
ablation; can be added later). LGBM and classical baselines (residual
correction is for models that can't take covariates natively).

## Architecture

### Stage 1 — TSFM train+val forecasts

For each of the 4 TSFMs, we need `y_pred` at every τ in train+val
(2018-2023, ~52,500 hours). This is **pure inference**, not training —
the TSFMs are zero-shot. We add a CLI flag to `scripts/run_tsfm.py`
to select the prediction window (default unchanged at test).

New parquets: `data/predictions/<tsfm>__nohijri__L<ctx>__seed0__window_trainval.parquet`
(4 files, ~150 MB total).

GPU time estimate (on RTX 4070 Laptop):

| TSFM | Test time (~11k rows) | Train+val (~52k rows) |
|---|---|---|
| Chronos-Bolt-Base L=720 | 28 s | ~150 s |
| Moirai-1.1-small L=336 | 46 s | ~250 s |
| TimesFM 2.5-200M L=168 | 212 s | ~1100 s (~18 min) |
| Time-MoE-200M L=720 | 645 s | ~3300 s (~55 min) |

Total: ~80 min GPU.

### Stage 2 — LightGBM residual heads

For each (TSFM, residual_variant) pair, fit a LightGBM regressor on:
- **Target:** `y_residual = y_true - y_pred_TSFM` over train+val
- **Features (nohijri variant):** weather (`temp_c`, `dewpoint_c`,
  `wind_speed`, `solar_rad`, `temp_sq`, `temp_above_35`) + calendar
  (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`) + lag
  features already in v2 (`y_lag_24h`, `y_lag_168h`, `y_lag_336h`,
  `y_roll168_mean`, `y_roll168_std`)
- **Features (hijri variant):** above + `is_ramadan`, `day_of_ramadan`,
  `is_eid`
- **Train/val split:** train on 2018-2022, early-stop on 2023, eval on
  test 2024-2025-03
- **Hyperparams:** reuse the existing LightGBM hyperparams from
  `notebooks/02_lgbm.ipynb` (the proposal-default tuned config). No
  re-tuning per residual head — that's a 200-run Optuna and YAGNI.

Output for each combo: a corrected prediction parquet
`data/predictions/<tsfm>__L<ctx>__residual_<variant>__seed0.parquet`
with columns `y_true, y_pred (= y_pred_TSFM + y_residual_hat),
regime, y_block` (block kept from the TSFM, not corrected per-step).

CPU time: ~30 s per LGBM fit × 8 combos = ~5 min.

### Stage 3 — Evaluation

Drop the 8 new parquets into the existing evaluation harness.
Regenerate:
- `docs/statistical_appendix.md` (now 20 models instead of 12)
- `docs/deep_analysis.md` (per-horizon stays the same since residual
  doesn't modify `y_block`; diurnal will reflect the residual fix)
- New: `docs/residual_correction.md` summarising the rescue effect
- Update `docs/tsfm_zero_shot_baseline.md` headline tables

## File layout

```
src/models/residual/                        (NEW package)
  __init__.py
  lgbm_residual.py                          LGBMResidualModel class

scripts/
  run_tsfm.py                               +--window {test,trainval,both} flag
  run_residual.py                           NEW: per-(TSFM, variant) fit+apply

tests/models/
  test_lgbm_residual.py                     NEW: unit tests

docs/
  residual_correction.md                    NEW: results doc
  superpowers/specs/2026-05-14-residual-correction-design.md   THIS FILE
  superpowers/plans/2026-05-14-residual-correction.md          plan to be written

data/predictions/
  chronos_bolt_base__nohijri__L720__seed0__window_trainval.parquet     NEW
  timesfm_2_5__nohijri__L168__seed0__window_trainval.parquet            NEW
  moirai_1_1_small__nohijri__L336__seed0__window_trainval.parquet       NEW
  time_moe_200m__nohijri__L720__seed0__window_trainval.parquet          NEW
  chronos_bolt_base__L720__residual_nohijri__seed0.parquet              NEW
  chronos_bolt_base__L720__residual_hijri__seed0.parquet                NEW
  timesfm_2_5__L168__residual_nohijri__seed0.parquet                    NEW
  timesfm_2_5__L168__residual_hijri__seed0.parquet                      NEW
  moirai_1_1_small__L336__residual_nohijri__seed0.parquet               NEW
  moirai_1_1_small__L336__residual_hijri__seed0.parquet                 NEW
  time_moe_200m__L720__residual_nohijri__seed0.parquet                  NEW
  time_moe_200m__L720__residual_hijri__seed0.parquet                    NEW
```

## Risk register

| Risk | Mitigation |
|---|---|
| `run_tsfm.py --window trainval` triggers OOM on long TSFM-context configs (L=720 over 52k rows) | Existing batched inference path already handles it; if needed, drop `batch_size` |
| Residual head overfits to train-period quirks and fails on 2024-25 distribution drift | Early-stop on 2023 val; LGBM has natural regularisation; matches the LightGBM-hijri protocol |
| Residual makes Heatwave *worse* by adding noise to an already-correct TSFM forecast | Will surface in regime-stratified table; if so, recommendation = route by regime (Heatwave: bare TSFM, Normal+Ramadan: TSFM+residual) |
| TSFM train+val parquets differ in structure from test parquets (e.g., missing `regime` column) | Add the test-set `regime`/feature columns to train+val parquets at write time; verified in Stage 1 |

## Out of scope (explicitly)

- **Multi-seed residual heads.** LightGBM with `random_state=42` is
  deterministic given the same data; the variance is mostly in Optuna
  trial-noise, not seed-noise.
- **Optuna tuning per residual head.** YAGNI; reuse existing LGBM-hijri
  hyperparams. A future Plan 7 sub-task could revisit.
- **PatchTSMixer correction.** PatchTSMixer can already take covariates
  through cross-channel mixing. Adding a residual head on top would be
  belt-and-suspenders.
- **Classical-baseline correction.** SARIMAX/MSTL+ETS already take
  exog regressors; no covariate-injection problem to solve.
- **Per-horizon residual.** A separate LGBM per horizon (24 heads)
  would be more expressive but 24× the fitting cost. A single LGBM
  trained on h=24 residuals (the canonical y_pred) is the minimum
  viable; per-horizon is future work.
- **The "orchestration" piece of Plan 6.** A single runner that loops
  over all (model, variant, seed) combos — convenient but YAGNI for
  this round. Each script (`run_tsfm.py`, `run_residual.py`) already
  parametrises cleanly.

## Definition of done

- 4 new TSFM train+val parquets exist (Stage 1).
- 8 new TSFM-residual test parquets exist (Stage 2).
- `docs/residual_correction.md` exists with: per-TSFM bare vs
  residual_nohijri vs residual_hijri MAE table (aggregate + 3 regimes),
  DM tests of residual_hijri vs bare for each TSFM, runtime table.
- `docs/statistical_appendix.md` regenerated with all 20 models.
- `docs/tsfm_zero_shot_baseline.md` headline table updated with the
  residual-corrected rows.
- `tests/test_smoke_pipeline.py` extended with 8 parquet existence checks.
- All pytest green.

## Decision deferred to plan

- Whether the train+val TSFM inference should reuse `scripts/run_tsfm.py`
  via a `--window` flag, or a thin separate `scripts/run_tsfm_trainval.py`
  wrapper. Plan defaults to the flag approach (less code duplication).
- Whether to share one LGBM hyperparam config across all 4 TSFMs, or
  to allow per-TSFM overrides. Plan defaults to shared.
