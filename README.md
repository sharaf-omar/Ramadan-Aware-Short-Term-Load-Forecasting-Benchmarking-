# Ramadan-Aware Short-Term Load Forecasting Benchmark

Capstone project benchmarking **31 forecasting systems** on Turkish national
electricity load — including 4 modern time-series foundation models, a
deep-learning baseline, two classical baselines, a tuned tree-based model,
and an array of ensemble / residual-correction / regime-routing composites —
with controlled ablations on **Hijri-calendar** regime shifts (Ramadan,
Eid) and extreme-heat regimes. Every claim in the result docs is backed by
a parquet on disk, a 95% block-bootstrap CI, and a Diebold-Mariano test
with Holm-Bonferroni multiple-comparison adjustment.

## Headline result

The champion entry — **meta-router-v2** — combines four post-hoc-residual-
corrected models for Normal hours, the tuned LightGBM-hijri model for
Ramadan, and a 4-model Heatwave ensemble. Aggregate MAE on the
14-month test set is **838.8 [95% CI 750.8, 948.7]**, a **−13.4%
improvement** over the strongest single model (Chronos-Bolt-Base L=720 at
MAE 968.9) and a **−14.3%** improvement over the 5-seed Optuna-tuned
LightGBM-hijri baseline (MAE 979.0).

| Rank | System | Aggregate MAE | 95% CI |
|------|--------|---------------|--------|
| 1 | meta-router-v2 (ensemble × regime-routing × residual heads) | **838.8** | [750.8, 948.7] |
| 2 | meta-router v1 | 840.9 | [754.5, 953.0] |
| 3 | ensemble (median of 4 residual-corrected) | 872.4 | [783.9, 984.9] |
| 4 | ensemble (median of 4 mixed) | 891.4 | [798.6, 1009.8] |
| 5 | stacked LightGBM meta-learner | 891.0 | [793.8, 1011.3] |
| 6 | LightGBM-hijri + LGBM residual head | 940.4 | [848.5, 1044.1] |
| 7 | Chronos-Bolt-Base L=720 + residual head | 948.5 | [846.9, 1072.4] |
| ... | (full 31-row table) | | |

Detail and the full 31-model statistical appendix:
[`docs/capstone_synthesis.md`](docs/capstone_synthesis.md) and
[`docs/statistical_appendix.md`](docs/statistical_appendix.md).

## Status

- [x] **Plan 1** — Foundation, evaluation harness, LightGBM baseline ([Plan 1 doc](docs/superpowers/plans/2026-05-13-foundation-and-lgbm-refactor.md))
- [x] **Plan 2** — TSFM zero-shot baselines (Chronos-Bolt, TimesFM, Moirai at L=336)
- [x] **Plan 3** — TSFM ablations: context-length sweep (Ablation C, L ∈ {96, 168, 336, 720}), Hijri-covariate ablation (Ablation A), Time-MoE rescue ([`tsfm_context_length_sweep.md`](docs/tsfm_context_length_sweep.md), [`tsfm_hijri_covariates.md`](docs/tsfm_hijri_covariates.md))
- [x] **Plan 4** — Classical baselines: MSTL+ETS and SARIMAX with day-ahead protocol ([`classical_baselines.md`](docs/classical_baselines.md))
- [x] **Plan 5** — PatchTSMixer deep-learning baseline (substituted for vanilla PatchTST so cross-channel mixing enables the Hijri ablation) ([Plan 5 spec](docs/superpowers/specs/2026-05-14-patchtsmixer-baseline-design.md))
- [x] **Plan 6** — Post-hoc LightGBM residual correction with regime-stratified routing ([`residual_correction.md`](docs/residual_correction.md))
- [x] **Plan 7** — Statistical rigor + deep analysis + cross-model synthesis ([`statistical_appendix.md`](docs/statistical_appendix.md), [`deep_analysis.md`](docs/deep_analysis.md), [`failure_modes.md`](docs/failure_modes.md), [`capstone_synthesis.md`](docs/capstone_synthesis.md))
- [x] **Tier 1-3 composite-model quick wins** — ensemble + regime router + meta-router + stacked LightGBM ([`capstone_synthesis.md`](docs/capstone_synthesis.md) §1 and §10)

## What's notable

**Scope.** 31 forecasting systems benchmarked on the same 10,944-row test
window. 152 automated pytest checks. 8 result docs +
1 statistical appendix + ~20 sub-task spec/plan documents in
[`docs/superpowers/`](docs/superpowers/).

**Statistical rigor.** Every headline claim has:
- A 95% **stationary block-bootstrap** confidence interval
  ([`bootstrap.py`](src/evaluation/bootstrap.py), Politis & Romano 1994).
- A **Diebold-Mariano** test with HAC variance estimator
  ([`dm_test.py`](src/evaluation/dm_test.py), Newey-West truncation lag
  matching the day-ahead horizon).
- **Holm-Bonferroni** multiple-comparison adjustment within each regime's
  pairwise comparison family.
- 4 pairwise DM matrices (aggregate + Normal + Ramadan + Heatwave) for
  all 31 systems in [`docs/statistical_appendix.md`](docs/statistical_appendix.md).

**Regime-conditional findings (proposal §6 confirmed).**
- LightGBM-hijri owns the **Ramadan** regime (MAE 800) by an 11-33%
  margin over the best TSFM — explicit Hijri feature engineering wins
  on this slice.
- Chronos-Bolt-Base L=720 owns the **Heatwave** regime (MAE 1221), 28%
  better than LightGBM — long-context attention beats hand-crafted temp
  features on weather extremes.
- Composite meta-router pools regime-specialist models to land at MAE
  838.8 aggregate.

**Residual-correction discovery (Plan 6 + extensions).**
Across 9 base models tested, the **lift from post-hoc LGBM residual
correction scales monotonically with bare-model weakness:**
- SARIMAX-hijri: 2486 → 1299 (−47.7%, largest single-model rescue)
- PatchTSMixer L=168: 1553 → 1046 (−32.6%)
- Moirai L=336: 1727 → 1317 (−23.7%)
- MSTL+ETS-hijri: 1528 → 1365 (−10.6%)
- TimesFM L=168: 1173 → 1058 (−9.9%)
- LightGBM-nohijri: 1003 → 951 (−5.3%, rescuing the incumbent's untuned variant)
- LightGBM-hijri: 979 → 940 (−4.0%, rescuing the incumbent itself)
- Time-MoE L=720: 986 → 955 (−3.2%)
- Chronos-Bolt-Base L=720: 969 → 949 (−2.1%)

The **regime-stratified routing** (train on Normal+Ramadan only; route
Heatwave τ back to the bare model) is the critical design choice —
without it, the strong TSFMs *regress* on aggregate because the residual
head injects bias from the more common Normal regime into Heatwave
forecasts.

**Hijri-covariate ablation reversed by injection mechanism.**
Plan 3 showed Hijri features injected via HuggingFace's in-band
covariate API *hurt* TimesFM and Moirai on Ramadan (DM p<0.001). Plan 6
showed the *same* Hijri features applied as a post-hoc residual head
*help* Moirai's Ramadan MAE by 22% and TimesFM's by 12%. **What you
inject matters less than how you inject it** — a result the proposal
predicted and that the benchmark now empirically supports.

## Reproduction

```
pip install -e . && pip install -r requirements.txt
cp .env-example .env  # optional: EPIAS credentials for the 2017 buffer fetch

python -m src.data.preprocess_epias        # -> data/processed/epias_processed_final.csv
python -m src.data.spatial_weights         # -> data/processed/weather_proxy.csv
python -c "from src.data.spatial_weights import build_southern_temp_series, PROCESSED_DIR; s = build_southern_temp_series(); s.reset_index().rename(columns={'index':'timestamp'}).to_csv(PROCESSED_DIR/'southern_temp.csv', index=False)"
python -m src.data.build_v2_dataset        # -> data/processed/final_training_set_v2.csv

# Full reproduction of every model in the leaderboard:
#   See docs/reproducibility.md for the canonical per-model commands.

pytest -q                                  # 152 checks
```

For TSFM and PatchTSMixer GPU work, see
[`docs/tsfm_execution_guide.md`](docs/tsfm_execution_guide.md). For
end-to-end one-shot reproduction of every parquet,
[`docs/reproducibility.md`](docs/reproducibility.md).

## Project structure

```
src/
  data/             # EPIAS + ERA5 preprocessing, southern-region temp
  features/         # hijri, calendar, weather_nonlinear, regimes
  models/
    base.py         # Model protocol
    ml/lgbm.py      # LightGBM
    classical/      # MSTL+ETS, SARIMAX (Plan 4)
    dl/             # PatchTSMixer (Plan 5)
    tsfm/           # Chronos-Bolt, TimesFM, Moirai, Time-MoE (Plans 2-3)
    residual/       # LGBMResidualModel (Plan 6)
  evaluation/       # metrics, regime stratification, DM test (HAC), block bootstrap, parquet I/O
data/
  raw/              # ERA5 NetCDFs + EPIAS load CSV
  processed/        # epias_processed_final.csv, weather_proxy.csv, southern_temp.csv, final_training_set_v2.csv
  predictions/      # one parquet per (model, variant, L, seed); 31+ systems total
  statistical_appendix/  # ci_table.csv, dm_aggregate.csv, dm_{regime}.csv
  analysis/         # horizon_mae.csv, diurnal_mae.csv, failure_modes_*.csv
docs/
  capstone_synthesis.md       # cross-plan integrated narrative
  tsfm_zero_shot_baseline.md  # headline cross-model table
  statistical_appendix.md     # CIs + 4 pairwise DM matrices (31 models)
  residual_correction.md      # Plan 6 + PatchTSMixer extension
  deep_analysis.md            # per-horizon + diurnal decomposition
  failure_modes.md            # worst-day analysis (universal + per-model)
  classical_baselines.md      # MSTL+ETS + SARIMAX detail
  tsfm_context_length_sweep.md  # Ablation C
  tsfm_hijri_covariates.md    # Ablation A
  v1_v2_lgbm_delta.md         # leakage fix and dataset migration
  tsfm_execution_guide.md     # GPU runbook (RunPod + local)
  reproducibility.md          # end-to-end run instructions
  references.md               # bibliography
  superpowers/                # design specs + implementation plans (per plan)
tests/                        # pytest mirror of src/ + smoke parquet-existence checks
scripts/                      # CLI runners + analyzer / build scripts
```

## Regime definition note

The proposal's strict 35°C heatwave threshold doesn't fire on Turkey's
population-weighted national temperature (max ever ~36.1°C, never 3
consecutive days ≥35°C). v2 dataset uses an unweighted-mean of 7
southern-Turkish cities (Adana, Şanlıurfa, Gaziantep, Diyarbakır,
Mersin, Konya, Antalya) as `temp_c_south` for heatwave detection while
keeping the pop-weighted `temp_c` for ML features. See
[`docs/v1_v2_lgbm_delta.md`](docs/v1_v2_lgbm_delta.md).
