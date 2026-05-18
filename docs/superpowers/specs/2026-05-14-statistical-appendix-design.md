# Statistical Appendix — Design Spec

**Date:** 2026-05-14
**Plan:** 7 (sub-task: statistical rigor pass)
**Branch:** `plan-7-statistics` off `plan-5-patchtst` (so PatchTSMixer parquets are in tree)

## Goal

Produce `docs/statistical_appendix.md` as the canonical statistical-rigor
artifact for the benchmark: 95% block-bootstrap CIs around MAE for
every headline model × regime, plus full pairwise Diebold-Mariano
matrices (aggregate / Normal / Ramadan / Heatwave), all Holm-Bonferroni
adjusted within each regime. Existing result docs stay untouched and
link to the appendix when they want to cite a CI.

Driver: `scripts/build_statistical_appendix.py` is the single
re-generator. It produces the doc + dumps the underlying CSVs to
`data/statistical_appendix/` so the report can source the numbers
directly without re-running the whole pipeline.

## Models in scope (12)

The full headline cohort as of Plan 5. Each model contributes one
parquet (median seed where multiple seeds exist).

| # | Model                           | Parquet |
|---|---------------------------------|---------|
| 1 | LightGBM nohijri                | `lgbm__nohijri__seed44.parquet` |
| 2 | LightGBM hijri                  | `lgbm__hijri__seed44.parquet` |
| 3 | Chronos-Bolt-Base L=720         | `chronos_bolt_base__nohijri__L720__seed0.parquet` |
| 4 | TimesFM 2.5-200M L=168          | `timesfm_2_5__nohijri__L168__seed0.parquet` |
| 5 | Moirai-1.1-R-Small L=336        | `moirai_1_1_small__nohijri__L336__seed0.parquet` |
| 6 | Time-MoE-200M L=720             | `time_moe_200m__nohijri__L720__seed0.parquet` |
| 7 | MSTL+ETS nohijri                | `mstl_ets__nohijri__seed0.parquet` |
| 8 | MSTL+ETS hijri                  | `mstl_ets__hijri__seed0.parquet` |
| 9 | SARIMAX nohijri                 | `sarimax__nohijri__seed0.parquet` |
| 10 | SARIMAX hijri                  | `sarimax__hijri__seed0.parquet` |
| 11 | PatchTSMixer nohijri L=168 s42 | `patchtsmixer__nohijri__L168__seed42.parquet` |
| 12 | PatchTSMixer hijri L=168 s42   | `patchtsmixer__hijri__L168__seed42.parquet` |

**Excluded:**
- `*__hijri_plusB__*` variants for SARIMAX and (once it lands) PatchTSMixer
  — predictions are statistically indistinguishable from the `hijri`
  variant (Compound regime empty 2018-2025; documented finding from
  Plan 4). Including them in the DM matrix would burn 11 cells per
  regime on guaranteed-null comparisons and inflate the Holm denominator.

## Bootstrap CI table

For each (model, regime) cell, compute:
- MAE = `mean(|y_true - y_pred|)` on the rows where `df.regime == regime`
  (aggregate regime = all rows)
- 95% CI via `src.evaluation.bootstrap.block_bootstrap_ci(abs_err,
  block_size=24, n_resamples=1000, alpha=0.05, seed=0,
  statistic=np.mean)`. Block=24 = 1 day for hourly data.

Output a 12-row × 4-regime CI table where each cell shows
`MAE [CI_lo, CI_hi]`. Markdown-rendered as a wide table.

## DM matrices (4 regimes)

For each regime r ∈ {aggregate, Normal, Ramadan, Heatwave}:
- Form pairwise (i, j) for `i < j` (lower-triangular, 66 pairs from
  12 models — `12*11/2 = 66`).
- For each pair, compute `dm_test(y_true_r, y_pred_i_r, y_pred_j_r,
  h=24, loss='mae')` on the regime-filtered intersection set.
- Collect the 66 raw p-values; apply `holm_bonferroni` within the
  regime; emit p_adj.

Render each regime as a 12×12 lower-triangular matrix where cell (i, j)
shows either:
- `< 0.001` / `< 0.01` / `< 0.05` / `ns` for the adjusted p-value, OR
- the signed DM statistic with a stars marker

I'll use the **adjusted-p-with-direction** format: each cell shows
`±sign·|stat|` (sign indicates which model wins; positive = column model
better, negative = row model better) plus a stars marker (`***` p_adj<0.001,
`**` p_adj<0.01, `*` p_adj<0.05, no mark = ns). Cleanest single-table
read for the report.

## File layout

| File | Responsibility |
|------|----------------|
| `scripts/build_statistical_appendix.py` | One-shot generator: loads parquets, computes CIs + DM, writes doc + CSVs. Idempotent. |
| `docs/statistical_appendix.md` | The human-readable artifact. |
| `data/statistical_appendix/ci_table.csv` | 12×4 CI table (long format: model, regime, mae, ci_lo, ci_hi) |
| `data/statistical_appendix/dm_<regime>.csv` | Four CSVs (one per regime): model_i, model_j, dm_stat, p_raw, p_holm |
| `tests/test_smoke_pipeline.py` | New parametrized smoke check: appendix file + 5 CSVs exist |

No changes to `src/evaluation/*` — `bootstrap.py` and `dm_test.py`
already do everything needed.

## Wall-clock estimate

Per regime: 12 bootstrap CIs (~1000 resamples each at ~10ms = 10s each
= 2 min) + 66 DM tests (~50ms each = 3s). Per appendix: 4 regimes
× ~2 min = ~10 min. Doc generation + CSV dump: seconds. **Total: ~10-15
min.**

## Reproducibility

- All RNG seeds fixed (bootstrap seed=0).
- All comparisons use the **intersection set** per regime — the set of
  τ rows where every model has a non-NaN prediction. Drop policy is
  documented in the appendix header.
- Holm-Bonferroni adjustment is per-regime (so significance in
  Ramadan-only is not penalised by the n=66 comparison family in
  aggregate).

## Out of scope (explicitly)

- Per-horizon decomposition using `y_block` (deferred to a separate
  Plan 7 sub-task).
- Diurnal / per-weekday analysis (separate sub-task).
- Failure-mode analysis (separate sub-task).
- Multi-seed CIs (we report median-seed numbers; cross-seed std is in
  the per-model results docs already).
- Cross-model synthesis writing / final report .docx artifacts.

## Open questions (none)

Everything above is pinned. Bootstrap params are the existing
`bootstrap.py` defaults; DM params match the rest of the project; doc
location matches the existing `docs/*.md` convention.
