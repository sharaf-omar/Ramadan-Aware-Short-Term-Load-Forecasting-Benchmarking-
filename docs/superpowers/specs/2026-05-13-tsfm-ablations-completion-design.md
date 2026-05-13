# Ramadan-Aware STLF: TSFM, Ablations, and Polish Completion Design

**Project:** Ramadan-Aware Short-Term Electricity Load Forecasting
**Source proposal:** `capstone_proposal.pdf` (April 2026)
**Status of project at design time:** Data pipeline + EDA + LightGBM baseline complete.
**Scope of this design:** Finish TSFM evaluation and ablations (A, B, C), add statistical rigor and deeper analysis, polish completed components, leave the project ready for the remaining classical baselines (MSTL+ETS, SARIMAX) and PatchTST, and for final report writing.
**Date:** 2026-05-13

---

## 1. Goal and Constraints

Deliver a defensible, reproducible benchmark in line with the capstone proposal:
- Zero-shot evaluation of 4 time-series foundation models (TSFMs) under Hijri-calendar regime shifts.
- All three proposal ablations (A: Ramadan-indicator inclusion, B: heatwave × Ramadan interaction, C: TSFM context-length sensitivity).
- Per-regime stratified metrics with Diebold-Mariano significance testing and block-bootstrap CIs.
- Hardware-aware execution: A100/RunPod preferred path, local RTX 4070 mobile (8GB) fallback path. Both produce identical artifacts.

**Hard constraints:**
- Local GPU: RTX 4070 mobile, 8GB VRAM. Caps Time-MoE-Large unless A100 is used.
- Test split fixed at 2024-01-01 .. 2025-03-31 per proposal §3.
- No fine-tuning of TSFMs (zero-shot only, per proposal §4.3).
- Reproducibility: every prediction artifact tied to a checkpoint SHA, env hash, and seed.

**Out of scope of this design** (handled in follow-on implementation plans):
- Writing the final paper / report prose.
- Implementing MSTL+ETS, SARIMAX, and PatchTST. The harness must accommodate them; their implementation is a separate ticket.

---

## 2. Repository Architecture

```
src/
  data/
    preprocess_epias.py          # PATCHED: clean t+24 features, no leakage
    spatial_weights.py           # unchanged
    download_epias.py            # rename of dounload_epias.py typo
    download_era5.py             # unchanged
  features/
    hijri.py                     # is_ramadan, day_of_ramadan, is_eid
    regimes.py                   # 4-regime labeler per proposal Table 3
    calendar.py                  # cyclical encodings, weekend
    weather_nonlinear.py         # temp^2, T_above_35, interactions
  models/
    base.py                      # Model protocol
    classical/
      mstl_ets.py                # placeholder file with NotImplementedError stub
      sarimax.py                 # placeholder stub
    ml/
      lgbm.py                    # wraps existing LightGBM + Optuna
    dl/
      patchtst.py                # placeholder stub
    tsfm/
      _adapter.py                # context windowing, bf16, batched generation
      chronos_bolt.py
      timesfm.py
      moirai.py
      time_moe.py
  evaluation/
    metrics.py                   # MAE, RMSE, MAPE, MASE
    regime_eval.py
    dm_test.py                   # DM + Newey-West HAC + Holm-Bonferroni
    bootstrap.py                 # stationary block bootstrap (Politis & Romano)
    residual_correction.py       # post-hoc LGBM head for univariate TSFMs
  reporting/
    tables.py
    plots.py
data/
  raw/                           # unchanged
  processed/
    epias_processed_final.csv    # output of preprocess_epias.py (already exists, regenerated)
    weather_proxy.csv            # output of spatial_weights.py (already exists)
    final_training_set_v1.csv    # KEPT for traceability vs current LGBM run
    final_training_set_v2.csv    # NEW: clean t+24 framing
    final_training_set_v2.meta.json
  predictions/                   # NEW: one parquet per (model, variant, L, seed)
  optuna/                        # NEW: Optuna SQLite stores
notebooks/
  01_eda.ipynb                   # existing eda_final_dataset.ipynb, lightly cleaned
  02_lgbm.ipynb                  # thin runner using src/models/ml/lgbm.py
  03_classical.ipynb             # placeholder for MSTL+ETS, SARIMAX
  04_patchtst.ipynb              # placeholder for PatchTST
  05_tsfm.ipynb                  # all 4 TSFMs zero-shot
  06_ablations.ipynb             # A, B, C tables + DM tests + deeper analysis
  07_report_artifacts.ipynb      # final figures and tables for the report
scripts/
  run_all.py                     # CLI orchestrator
docs/
  tsfm_execution_guide.md        # NEW: RunPod + local instructions (sibling deliverable)
  superpowers/specs/             # this design doc + future spec docs
```

**Architectural rule.** Every model implements `src/models/base.py::Model` and emits predictions to a parquet under `data/predictions/`. The evaluation, statistical-testing, and reporting layers consume only the parquet store, never the models directly. This makes the harness uniform across LGBM, classical baselines, PatchTST, and TSFMs, and means a re-run of any single model never invalidates downstream results.

---

## 3. Forecast Framing and Evaluation Protocol

**Forecast equation (proposal-exact):** for each forecast issue time `t`, predict `y[t+24]`.

**Row convention in `final_training_set_v2.csv`.** Each row indexed by forecast time `τ = t+24`. Issuance time is `τ - 24`.

```
τ                       (UTC, hourly, the time being forecast)
y_target          = y[τ]
y_lag_24h         = y[τ-24]   = y at issuance time t
y_lag_48h         = y[τ-48]   = y at t-24
y_lag_168h        = y[τ-168]  = y a week before τ
y_lag_336h        = y[τ-336]  = y two weeks before τ
y_roll24_mean     = mean(y[τ-47 .. τ-24])               # 24h window ending at issuance
y_roll24_std      = std (y[τ-47 .. τ-24])
y_roll168_mean    = mean(y[τ-191 .. τ-24])              # 168h window ending at issuance
y_roll168_std     = std (y[τ-191 .. τ-24])
temp_c, dewpoint_c, wind_speed, solar_rad   = weather at τ (treated as known forecast)
temp_sq, temp_above_35
hour, day_of_week, month + sin/cos
is_weekend
is_ramadan, day_of_ramadan, is_eid
ramadan_x_hour_sin, ramadan_x_hour_cos, ramadan_x_weekend
heatwave_x_temp, ramadan_x_heatwave, ramadan_x_temp_above_35
regime                  (Normal | Ramadan | Heatwave | Compound)
```

**Leakage fix (critical).** Old code used `s = df['actual_load'].shift(1)` then `.rolling(24)`. At row `τ` that window covers `[τ-24, τ-1]` — peeks 23h past the issuance time `t = τ-24`. New code uses `s = df['actual_load'].shift(24)` then `.rolling(24)`, giving `[τ-47, τ-24]` — the window ends *at* issuance.

**Versioning.** Old `final_training_set_v1.csv` is kept for traceability; LGBM technical report compares v1 vs v2 deltas. Future model runs all use v2.

**TSFM evaluation.** For each row `τ` in test split:
- Build context `y[τ-24-L+1 .. τ-24]` of length `L` ending at issuance.
- Build dynamic covariates over context + horizon when supported.
- Call TSFM, get 24-step block forecast, retain the **24th** entry as `y_pred[τ]`.
- Block-forecasters (PatchTST, all TSFMs) additionally save the full 24-step block for the per-horizon decomposition analysis (§7).

**Why take only h=24:** matches proposal `y_{t+24}` definition; makes TSFM and tabular predictions directly comparable on a single number per `τ`.

**Effective test window.** Full test span is 2024-01-01 .. 2025-03-31. For TSFMs with `L=720` (≈30 days), the first ~30 days lack sufficient context and are dropped. Cross-model DM tests run on the *intersection of valid τ across compared models* (longest-context window). Per-model solo tables run on each model's own valid set, footnoted with row count.

**Predictions parquet schema:**
```
τ : datetime[utc] (idx) | y_true : float | y_pred : float | y_block : list[float, 24]
model : str | variant : str | context_length : int | seed : int | regime : str
```
One parquet per `(model, variant, context_length, seed)` tuple. Single source of truth for every downstream table.

---

## 4. Data Pipeline Polish

### 4.1 Leakage-free feature derivation
Rewrite `src/data/preprocess_epias.py` to:
- Index every output row by forecast time `τ` (not issuance time).
- Build all lag features as `y.shift(k)` with `k ≥ 24`.
- Build all rolling features as `y.shift(24).rolling(w)` with `w` matching the window.
- Compute Hijri features at `τ` via `hijridate` (existing logic, moved to `src/features/hijri.py`).
- Drop the buffer-month logic only after lag/rolling computation is complete (i.e., compute lags on the full series including 2017 buffer, then slice).

### 4.2 Regime labeling
Replace the LGBM notebook's `quantile(0.95)` heatwave proxy with the proposal-specified definition (`src/features/regimes.py`):

```python
def label_regimes(df: pd.DataFrame) -> pd.Series:
    # df has DatetimeIndex (UTC), columns include temp_c, is_ramadan
    daily_max = df.groupby(df.index.date)['temp_c'].max()
    hot_day   = daily_max >= 35.0                              # absolute threshold

    # Heatwave = >= 3 consecutive days hot
    run_id    = (hot_day != hot_day.shift()).cumsum()
    run_len   = hot_day.groupby(run_id).transform('size')
    heatwave_day = hot_day & (run_len >= 3)

    hot_map   = heatwave_day.to_dict()
    df_is_hw  = pd.Series([hot_map.get(ts.date(), False) for ts in df.index],
                          index=df.index)

    return pd.Series(
        np.select(
            [df['is_ramadan'].astype(bool) &  df_is_hw,
             df['is_ramadan'].astype(bool) & ~df_is_hw,
             ~df['is_ramadan'].astype(bool) &  df_is_hw],
            ['Compound', 'Ramadan', 'Heatwave'],
            default='Normal'
        ),
        index=df.index,
    )
```

### 4.3 Sidecar metadata
`final_training_set_v2.meta.json` written alongside the CSV with:
- SHA256 of source CSVs and NetCDF files.
- `hijridate` package version.
- Heatwave threshold params (35.0°C, 3 consecutive days).
- Feature list with leak-free annotations.
- Generation timestamp + git commit hash.

---

## 5. Model Layer

All models implement `Model` protocol (`src/models/base.py`):

```python
class Model(Protocol):
    name: str
    supports_dynamic_covariates: bool
    needs_training: bool

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            hijri: bool, seed: int) -> None: ...

    def predict(self, test_df: pd.DataFrame,
                context_length: int | None = None) -> pd.DataFrame: ...
    # returns DataFrame[τ, y_true, y_pred, y_block (optional, 24 floats)]
```

### 5.1 LightGBM (`src/models/ml/lgbm.py`)
- Wraps existing notebook logic.
- 50-trial Optuna TPE on val MAE, persisted to `data/optuna/lgbm.db`.
- 5 seeds: 42, 43, 44, 45, 46.
- Three feature variants:
  - `nohijri`: BASE only.
  - `hijri`: BASE + HIJRI.
  - `hijri_plusB`: BASE + HIJRI + `ramadan_x_heatwave` + `ramadan_x_temp_above_35`.

### 5.2 TSFMs (`src/models/tsfm/`)
Shared `_adapter.py`:
- Context-window slicing per `τ` and `L`.
- bf16 cast for inference; fp32 cast for output assembly.
- Batched generation with adaptive batch size on OOM retry.
- Output unpacking to unified prediction-DataFrame schema.

**Per-model wrappers:**

| File | Checkpoint (HF) | Covariate handling | VRAM (bf16, batch≈64) |
|---|---|---|---|
| `chronos_bolt.py` | `amazon/chronos-bolt-base` (univariate) | None | ~2 GB |
| `timesfm.py` | `google/timesfm-2.0-500m-pytorch` | Dynamic real covariates over context + horizon | ~3 GB |
| `moirai.py` | `Salesforce/moirai-1.1-R-large` via `uni2ts` | Any-variate input | ~5 GB |
| `time_moe.py` | Local: `Maple728/TimeMoE-200M`; A100: `Maple728/TimeMoE` (the large MoE) | None | Local ~3 GB / A100 ~25 GB |

**Hijri ablation per architecture:**
- TimesFM, Moirai: feed `is_ramadan`, `day_of_ramadan`, `is_eid`, `temp_c` as dynamic real covariates over both context and horizon. Hijri values for the forecast horizon are deterministic (calendar lookup).
- Chronos, Time-MoE: no inline covariate channel → ablation A is implemented via post-hoc LGBM residual correction (§7.3).

**Critical rule:** TSFM context windows contain only raw `y` and proposal-specified dynamic covariates. No tabular engineered features (rolling means, sin/cos, etc.). Adding them is not zero-shot evaluation and would compromise contribution C1.

### 5.3 Classical and PatchTST (placeholder stubs)
`mstl_ets.py`, `sarimax.py`, `patchtst.py` get `NotImplementedError` stubs implementing the `Model` protocol's signature. The protocol must be wide enough to fit them when their implementation tickets land. Specifically:
- MSTL+ETS will call `fit` per Ramadan window for re-estimated daily seasonal; `predict` is a rolling-origin daily refit.
- SARIMAX will call `fit` weekly with exogenous regressors.
- PatchTST will call `fit` once with train+val; `predict` is single-pass over test.

---

## 6. Ablations

| Ablation | Models in scope | Variant additions |
|---|---|---|
| A: Ramadan-indicator (primary) | All 9 | `nohijri` vs `hijri` (TimesFM/Moirai: covariates; Chronos/Time-MoE: post-hoc residual; LGBM/PatchTST/SARIMAX: feature set; MSTL+ETS: Ramadan-window seasonal) |
| B: Heatwave × Ramadan interaction | LGBM, PatchTST, SARIMAX, TimesFM, Moirai | `hijri_plusB` adds `ramadan_x_heatwave`, `ramadan_x_temp_above_35` |
| C: Context-length sensitivity | 4 TSFMs | `L ∈ {96, 168, 336, 720}` |

**Run counts (in-scope for this design):**
- LGBM: 3 variants × 5 seeds = 15 runs (~5 min each, ~1.25h total).
- TSFMs: Chronos+Time-MoE 8 passes (4 L × 2 models, ablation A via post-hoc); TimesFM+Moirai 16 passes (4 L × 2 hijri-variants × 2 models). Total 24 zero-shot inference passes.

**Run counts (out-of-scope models the harness must accommodate, listed for budgeting only):**
- PatchTST: 3 variants × 5 seeds = 15 training runs (~30 min each on 4070 mobile, ~1h each on A100 → 7.5h local / ~1h A100).
- SARIMAX: 3 variants × 1 seed (deterministic) ≈ 9h CPU.
- MSTL+ETS: 2 variants × 1 seed (fast).

These three models get `Model`-protocol stubs in this design (§5.3) but their full implementations and run executions live in follow-on tickets.

**Runtime envelopes:**
- 4070 mobile fallback: 24–36h GPU for TSFM passes, run overnight × 2.
- A100 preferred: 6–10h single session for all TSFM passes including Time-MoE-Large.

---

## 7. Statistical Testing and Deeper Analysis

### 7.1 Diebold-Mariano test (`src/evaluation/dm_test.py`)
- Loss differential `d_t = |e_A,t| - |e_B,t|` (MAE-based; squared variant available).
- HAC variance via `statsmodels.stats.sandwich_covariance.cov_hac`, lag = h-1 = 23.
- Two-sided, α=0.05.
- Pairwise across all model pairs within each regime per ablation variant.
- Multiple-comparison correction: Holm-Bonferroni *within* each regime's family.
- Cross-context-length comparisons use intersection-τ window.
- Output: `data/predictions/dm_results.parquet`.

### 7.2 Block bootstrap CIs (`src/evaluation/bootstrap.py`)
- 1000 resamples, Politis & Romano stationary block bootstrap, mean block length = 24h.
- 95% CI on MAE / RMSE / MAPE / MASE per regime per (model, variant, L, seed).
- For multi-seed models, report median-seed metric with the bootstrap CI; also report seed std as a separate column.
- Every table cell takes the form `metric [ci_low, ci_high]`.

### 7.3 Post-hoc residual correction (`src/evaluation/residual_correction.py`)
Applies to Chronos-Bolt and Time-MoE only.

1. Run TSFM zero-shot on train+val splits (2018-01-01 .. 2023-12-31). Compute residuals `r_t = y_true_t - y_pred_t`.
2. Train LGBM on `(features → r_t)` with feature set = Hijri block only (`is_ramadan`, `day_of_ramadan`, `is_eid`, `ramadan_x_hour_sin/cos`). 3 seeds, 50-trial Optuna with val = 2023.
3. At test time, predict `r̂_t`. Corrected forecast `y_pred_corrected = y_pred_raw + r̂_t`.

**Trained only on train residuals.** Never touches test. This is the principled implementation of proposal §4.3's "post-hoc LightGBM residual correction on Ramadan errors."

The residual head's feature importance is reported in F7 to surface which Hijri signals matter most.

### 7.4 Per-horizon decomposition (deeper analysis)
For block-forecasters (PatchTST, all 4 TSFMs):
- Retain full 24-step forecast in `y_block` column.
- Compute MAE at `h ∈ {1, 4, 8, 12, 16, 20, 24}` per regime per model.
- Plot F3: MAE vs horizon, faceted by regime, one line per model.

Single-point models (LGBM, SARIMAX, MSTL+ETS) do not get a horizon ladder in this design. F3 is computed only for block-forecasters. A separate follow-on ticket may add `h ∈ {1, 12, 24}` LGBM models if the report needs full-horizon comparison parity.

### 7.5 Error-vs-temperature curve (deeper analysis)
Per model, bin test errors by `temp_c` (5°C bins, 0–45°C). Plot F4: mean `|error|` vs bin midpoint with bootstrap CIs. Tests the hypothesis that TSFMs degrade superlinearly above 35°C and that ablation B closes the gap on covariate-capable models.

### 7.6 Hour-of-day signed error during Ramadan (deeper analysis)
For Ramadan and Compound regimes: mean signed error by UTC hour 0–23 per model. Compare `nohijri` vs `hijri` variants. Highlight iftar (~UTC17) and suhoor (~UTC01). Surfaces exactly which hours the Hijri features fix and which remain biased.

### 7.7 Reproducibility hygiene
- `requirements.txt` + `requirements.lock` via `pip-compile --generate-hashes`.
- `data/predictions/_manifest.json` per run: model, HF checkpoint SHA (queried at load), torch + CUDA versions, seed list, runtime seconds, host (local | runpod_a100), peak VRAM.
- Optuna studies persisted as SQLite under `data/optuna/<model>.db`.
- Git commit hash logged in every parquet's metadata.

### 7.8 Multi-seed protocol
For models with stochastic training (LGBM, PatchTST):
- Run 5 seeds: 42, 43, 44, 45, 46.
- For headline metric tables: report mean ± std across seeds AND the median-seed predictions for DM tests.
- For DM tests: use the median-seed prediction series. Document this choice in the test report.
- For block bootstrap CIs: applied to the median-seed predictions.

---

## 8. Dual-Path Execution Model

Two reproducible execution paths, both producing identical artifact layouts in `data/predictions/`.

### 8.1 A100 preferred path (RunPod, Vast.ai, Lambda)
- TSFM inference for ablations A, B, C runs on a single A100 (40 or 80GB).
- Time-MoE upgrades from -200M (local cap) to -Large for strict proposal compliance.
- All TSFM passes complete in ~6–10h wall-clock.
- PatchTST training also moves to A100 (~1h vs 7.5h local).
- Predictions parquets are downloaded back to local machine before DM tests and reporting (orchestrated entirely from local).
- LGBM, MSTL+ETS, SARIMAX stay on local — CPU-bound; A100 doesn't help.

### 8.2 Local 4070 mobile fallback path
- TSFM inference uses smaller checkpoints where required (Time-MoE-200M instead of -Large; Chronos-Bolt-Base instead of -Large).
- bf16 inference with batch sizes 32–128 depending on model.
- Total TSFM wall-clock: ~24–36h across two overnight sessions.
- A footnote in the final report names the substitutions explicitly.

### 8.3 Execution guide deliverable
A sibling document `docs/tsfm_execution_guide.md` provides concrete step-by-step instructions for both paths:
- RunPod: account, template, pod spec, volume mounts, SSH/Jupyter access, repo+env setup, dataset upload via scp, running the pipeline, downloading predictions, pod teardown, cost estimates.
- Local: conda/venv bootstrap, HuggingFace cache config, model download verification, running each notebook in order, expected runtimes per model on the 4070 mobile.

Both paths must round-trip a `data/predictions/_manifest.json` that identifies the host so downstream consumers can interpret the results.

---

## 9. Reporting Artifacts

Generated by `notebooks/07_report_artifacts.ipynb` from `data/predictions/`.

### 9.1 Tables
- **T1** — Aggregate test metrics per model variant (headline). MAE, RMSE, MAPE, MASE; rows = 9 models × variant. Block-bootstrap 95% CIs per cell.
- **T2** — Per-regime metrics: 4 regimes × 9 models. Headline ablation A table.
- **T3** — Ablation A deltas: `MAE[nohijri] − MAE[hijri]` per regime per model. DM significance stars (Holm-corrected).
- **T4** — Ablation B compound-regime deltas, covariate-capable models only.
- **T5** — Ablation C: TSFM `MAE(L)` heatmap, `L × model × regime`.
- **T6** — Reproducibility appendix: checkpoint SHAs, env hash, seed list, runtime per model, host.

### 9.2 Figures
- **F1** — Actual vs predicted test series, best-per-class (best classical, best ML/DL, best TSFM). Ramadan windows shaded.
- **F2** — Per-regime MAE bar chart with bootstrap whiskers.
- **F3** — Per-horizon MAE curve, faceted by regime, one line per block-forecaster.
- **F4** — Error vs temperature with 35°C threshold marked.
- **F5** — Hour-of-day signed error during Ramadan, `nohijri` vs `hijri` side by side per model.
- **F6** — Context-length sweep curve per TSFM per regime.
- **F7** — LGBM gain importance; residual-correction LGBM importance for Chronos/Time-MoE.

---

## 10. Polish to Existing Assets

- Rename `src/data/dounload_epias.py` → `src/data/download_epias.py`.
- Add `requirements.txt` + `requirements.lock`.
- Add `README.md` with reproduction steps that point to `docs/tsfm_execution_guide.md`.
- Convert the three existing `.docx` reports (`eda_findings_report.docx`, `lgbm_technical_report.docx`, `preprocessing_pipeline_report.docx`) into markdown sections of the final report scaffold. Keep `.docx` files in `docs/` for traceability.
- Rerun LGBM notebook on `final_training_set_v2.csv`. Expect small metric shifts vs current v1 results. Document the v1→v2 delta in a "data quality fix" subsection of the LGBM technical report.
- Fix the missing-cell bug in `notebooks/lgbm_training.ipynb` cell 26 — references `regime_metrics_base` / `regime_metrics_tuned` which are never assigned in the notebook. After refactor this becomes a thin call into `src/evaluation/regime_eval.py`.

---

## 11. Build Sequence

Suggested execution order. Steps 1–2 are blocking; everything below can be reordered around hardware availability.

1. **Polish foundation** — fix `preprocess_epias.py`, build `src/features/{hijri,regimes,calendar,weather_nonlinear}.py`. Regenerate `final_training_set_v2.csv` with meta sidecar. Assert test-window data coverage (no NaN `actual_load` rows in 2024-01-01 .. 2025-03-31; document and truncate if source ends earlier). Blocks everything downstream.
2. **Evaluation harness** — `src/evaluation/{metrics,regime_eval,dm_test,bootstrap}.py`. Unit tests on synthetic series. Blocks all model evaluation.
3. **Refactor LGBM** — move into `src/models/ml/lgbm.py`. Rerun on v2 data, all 5 seeds × 3 variants. First end-to-end working slice; validates the harness against known-good numbers.
4. **TSFM adapters** — `_adapter.py` skeleton, then in order: Chronos-Bolt (smallest API surface), TimesFM 2.0, Moirai-1.1, Time-MoE. Each model: zero-shot on test, predictions to parquet, sanity-check vs LGBM on Normal regime. Heavy step; ideal target for A100 session.
5. **Post-hoc residual correction** — `src/evaluation/residual_correction.py`. Train residual heads for Chronos + Time-MoE on train-split residuals.
6. **Ablations A, B, C orchestration** — `scripts/run_all.py`. Ablation C TSFM context-length sweep is the longest run. A100 path: single overnight session. Local path: two overnight sessions.
7. **Statistical analysis** — DM tests + Holm-Bonferroni + block bootstrap.
8. **Deeper analysis** — per-horizon decomp, error-vs-temperature, hour-of-day signed error.
9. **Report artifacts** — `07_report_artifacts.ipynb` generates all T1–T6 and F1–F7 from the predictions parquet store.

Step 4 should start as early as possible because it gates the GPU-bound work. Steps 6–9 are pure orchestration and analysis from the artifact store.

---

## 12. Non-Goals (this design)
- MSTL+ETS, SARIMAX, PatchTST implementation. Stubs + protocol fit; full implementations are separate tickets.
- Writing the final paper prose.
- Conformal prediction intervals (interesting but not in proposal scope).
- Cross-grid generalization experiments (proposal is single-grid; out of scope).

---

## 13. Open Questions / Risks
- **Moirai 1.1 R Large packaging.** `uni2ts` API changes have been frequent; pin to a specific tag in `requirements.lock`.
- **TimesFM 2.0 dynamic covariate handling.** The official `timesfm` package's `forecast_with_covariates` API requires alignment-checking; we will verify on a 100-row smoke test before running the full ablation.
- **Time-MoE-Large on A100.** The "Large" sparse variant has ~6.9B parameters total (~2.4B active). On a 40GB A100 in bf16 it fits with batch 32–64; on 80GB SXM it fits comfortably. RunPod default is 80GB.
- **Tail of test set (2025-03 partial month).** Ramadan 2025 runs March 1–30 in Turkey; the test split ending 2025-03-31 should cover it fully. But the source CSV `electricity_consumption_2018_2025.csv` may not extend to 2025-03-31. Step 1 of the build sequence must `head/tail` the source data and assert the test window has zero NaNs in `actual_load` before any model runs. If actual data ends earlier, truncate the test split to match and document the deviation in the LGBM technical report.
- **`hijridate` ambiguity at month boundaries.** Hijri days start at maghrib (sunset), not midnight. The current code uses Gregorian-day → Hijri-day. Boundary hours can be miscategorized by up to ~6h. Acceptable per proposal but worth documenting in the spec footnote.
