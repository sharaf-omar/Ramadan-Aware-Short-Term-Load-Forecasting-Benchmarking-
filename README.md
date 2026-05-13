# Ramadan-Aware Short-Term Load Forecasting Benchmark

Capstone project benchmarking time-series foundation models (TSFMs) against
classical and ML baselines on Turkish national electricity load, with
controlled ablations on Hijri-calendar regime shifts (Ramadan, Eid) and
extreme-heat regimes.

## Status (current milestone)

- [x] Plan 1: Foundation + evaluation harness + LightGBM refactor
- [x] Plan 2: TSFM zero-shot baseline (Chronos, TimesFM, Moirai at L=336; Time-MoE deferred)
- [ ] Plan 3: Classical baselines (MSTL+ETS, SARIMAX)
- [ ] Plan 4: PatchTST
- [ ] Plan 5: Post-hoc residual correction + ablation orchestration
- [ ] Plan 6: Statistical analysis + deeper analysis + report artifacts

See [docs/superpowers/specs/2026-05-13-tsfm-ablations-completion-design.md](docs/superpowers/specs/2026-05-13-tsfm-ablations-completion-design.md)
for the full design and [docs/superpowers/plans/](docs/superpowers/plans/) for active plans.

## Reproduction

1. `pip install -e .` and `pip install -r requirements.txt`.
2. Copy `.env-example` to `.env` and fill EPIAS credentials (optional — without
   it the 2017 buffer fetch is skipped and early 2018 lag features will be NaN,
   which is harmless because those rows are dropped from train).
3. `python -m src.data.preprocess_epias` → builds `data/processed/epias_processed_final.csv`.
4. `python -m src.data.spatial_weights` → builds `data/processed/weather_proxy.csv` (full pop-weighted weather).
5. Generate the southern-region temperature series for heatwave detection:
   ```
   python -c "from src.data.spatial_weights import build_southern_temp_series, PROCESSED_DIR; s = build_southern_temp_series(); s.reset_index().rename(columns={'index':'timestamp'}).to_csv(PROCESSED_DIR/'southern_temp.csv', index=False)"
   ```
6. `python -m src.data.build_v2_dataset` → builds `data/processed/final_training_set_v2.csv` + meta sidecar.
7. Open `notebooks/02_lgbm.ipynb` and run all cells (50-trial Optuna + 15 model runs, ~30 min).
8. `pytest -v` to verify the harness.

For TSFM and PatchTST GPU work, see [docs/tsfm_execution_guide.md](docs/tsfm_execution_guide.md).

## Project structure

```
src/
  data/          # EPIAS + ERA5 preprocessing, southern-region temp
  features/      # hijri, calendar, weather_nonlinear, regimes
  models/
    base.py      # Model protocol
    ml/lgbm.py   # LightGBM (current)
    classical/   # (Plan 3) MSTL+ETS, SARIMAX
    dl/          # (Plan 4) PatchTST
    tsfm/        # (Plan 2) Chronos, TimesFM, Moirai, Time-MoE
  evaluation/    # metrics, regime stratification, DM test (HAC), bootstrap, parquet I/O
data/
  raw/           # ERA5 NetCDFs + EPIAS load CSV
  processed/     # epias_processed_final.csv, weather_proxy.csv, southern_temp.csv, final_training_set_v2.csv
  predictions/   # one parquet per (model, variant, context_length, seed)
notebooks/       # thin runners; logic lives in src/
docs/            # design specs, plans, technical reports
tests/           # pytest mirror of src/
```

## Regime definition note

The proposal's strict 35 C heatwave threshold doesn't fire on Turkey's
pop-weighted national temperature (max ever ~36.1 C, never 3 consecutive days
≥35 C). v2 dataset uses an unweighted-mean of 7 southern-Turkish cities
(Adana, Şanlıurfa, Gaziantep, Diyarbakır, Mersin, Konya, Antalya) as
`temp_c_south` for heatwave detection, while keeping the pop-weighted
`temp_c` for ML features. See `docs/v1_v2_lgbm_delta.md` for details.
