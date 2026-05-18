# Reproducibility Manifest

Single canonical reference for reproducing every artifact in the
benchmark — from raw EPIAS load CSV to the 31 prediction parquets, 5
analysis CSVs, statistical appendix, and 6 figures. Read top-to-bottom
once for a clean-room build, or jump to the section matching what you
want to regenerate.

---

## 0. Environment

```bash
# Python 3.12 venv (uv recommended; pip works)
python -m venv .venv
.venv/Scripts/activate     # Windows; or `source .venv/bin/activate` on Linux
pip install -e .
pip install -r requirements.txt
```

For TSFM GPU work, a CUDA-capable PyTorch build matched to the local
CUDA driver is required. We tested on:
- Windows 11, RTX 4070 Laptop (8 GB VRAM)
- `torch==2.4.1+cu124`, `transformers==4.48.3`

`timesfm` must be installed from GitHub HEAD; the PyPI 1.0 build is
broken under Python 3.12. See [`docs/tsfm_execution_guide.md`](tsfm_execution_guide.md)
for the per-TSFM dependency notes and a RunPod recipe for cloud GPUs.

```bash
pytest -q   # 152 checks — sanity baseline
```

---

## 1. Data pipeline (raw → v2)

```bash
# Optional: copy .env-example to .env and fill EPIAS credentials.
# Without it, the 2017 buffer fetch is skipped (early-2018 lag rows
# get dropped at v2 build time; harmless).
cp .env-example .env

# Stage 1: EPIAS load (hourly Turkish consumption, 2018-present)
python -m src.data.preprocess_epias
# -> data/processed/epias_processed_final.csv

# Stage 2: ERA5 → population-weighted national weather + 7-city southern temp
python -m src.data.spatial_weights
# -> data/processed/weather_proxy.csv

python -c "from src.data.spatial_weights import build_southern_temp_series, PROCESSED_DIR; s = build_southern_temp_series(); s.reset_index().rename(columns={'index':'timestamp'}).to_csv(PROCESSED_DIR/'southern_temp.csv', index=False)"
# -> data/processed/southern_temp.csv

# Stage 3: feature-engineered v2 dataset (load + weather + Hijri + lags + regimes)
python -m src.data.build_v2_dataset
# -> data/processed/final_training_set_v2.csv + meta sidecar
```

The v2 dataset is the single source of truth for every downstream
model. Splits are fixed chronologically: train = 2018-2022, val = 2023,
test = 2024-01-01 .. 2025-03-31 (10,944 hours).

---

## 2. Plan 1 — LightGBM baseline (15 parquets)

```bash
# Tunes hyperparams via 50-trial Optuna on val 2023, then trains
# 3 variants (nohijri, hijri, hijri_plusB) × 5 seeds (42-46).
jupyter nbconvert --to notebook --execute notebooks/02_lgbm.ipynb
# -> data/predictions/lgbm__<variant>__seed<s>.parquet  (15 files)
```

Wall-clock: ~30 min on a modern CPU.

---

## 3. Plans 2-3 — TSFM zero-shot baselines + Ablation A/C (18 parquets)

```bash
# Ablation C: context-length sweep, all 4 TSFMs, nohijri only
for L in 96 168 336 720; do
  python scripts/run_tsfm.py --model chronos  --context-length $L
  python scripts/run_tsfm.py --model timesfm  --context-length $L
  python scripts/run_tsfm.py --model moirai   --context-length $L
  python scripts/run_tsfm.py --model timemoe  --context-length $L
done
# -> 16 parquets

# Ablation A: Hijri covariate variants (TimesFM and Moirai only — the two
# TSFMs with native covariate ingestion)
python scripts/run_tsfm.py --model timesfm --context-length 336 --variant hijri
python scripts/run_tsfm.py --model moirai  --context-length 336 --variant hijri
# -> 2 more parquets
```

Wall-clock on RTX 4070 Laptop: ~90 min total (Time-MoE L=720 is the
slow tail at ~30-40 min alone).

---

## 4. Plan 4 — Classical baselines (5 parquets)

```bash
python scripts/run_classical.py --model mstl_ets --variant nohijri
python scripts/run_classical.py --model mstl_ets --variant hijri
python scripts/run_classical.py --model sarimax  --variant nohijri
python scripts/run_classical.py --model sarimax  --variant hijri
python scripts/run_classical.py --model sarimax  --variant hijri_plusB
```

Wall-clock: ~30 s for MSTL+ETS (both variants); ~2 h for each SARIMAX
variant on CPU.

---

## 5. Plan 5 — PatchTSMixer deep baseline (~16 parquets when complete)

```bash
# L-probe (4 runs): nohijri only, seed 0, all 4 context lengths
for L in 96 168 336 720; do
  python scripts/run_patchtsmixer.py --variant nohijri --context-length $L --seed 0
done

# Headline grid (chosen best L = 168 in this benchmark): 3 variants × 5 seeds
# Project shipped with seed 42 only as a sentinel; full 5-seed grid is open work.
for VARIANT in nohijri hijri hijri_plusB; do
  for SEED in 42 43 44 45 46; do
    python scripts/run_patchtsmixer.py --variant $VARIANT --context-length 168 --seed $SEED
  done
done
```

Wall-clock on RTX 4070 Laptop: ~1 h L-probe, ~20 h full 15-run grid.
Single-seed run at L=168 is ~80 min.

---

## 6. Plan 6 — Post-hoc LightGBM residual heads (8 parquets)

The Plan 6 hijri-variant parquets use a tuned configuration (wider
LightGBM + dense Hijri features + regime-stratified routing); the
nohijri-variant uses the simpler 3-fold time-block CV in
[`scripts/run_residual.py`](../scripts/run_residual.py).

```bash
# nohijri variant (simpler residual head, all-regime training)
for spec in "chronos_bolt_base 720" "moirai_1_1_small 336" \
            "timesfm_2_5 168" "time_moe_200m 720"; do
  read -r name L <<<"$spec"
  python scripts/run_residual.py \
      --tsfm-parquet ${name}__nohijri__L${L}__seed0.parquet \
      --tsfm-name ${name} --context-length ${L} \
      --variant nohijri --seed 0
done

# hijri variant (tuned LightGBM with regime-stratified routing)
python scripts/tune_residual.py
```

Wall-clock: ~30 s total. The tune script overwrites the
`*__residual__hijri__*` parquets in place with the tuned versions.

---

## 7. Tier 1, 2, 3 quick wins — composite models (8 parquets)

```bash
# Tier 1: ensemble (mixed members) + regime router + PatchTSMixer+residual
python scripts/build_quick_wins.py
# -> ensemble__top4__seed0.parquet
# -> routed__best_per_regime__seed0.parquet
# -> patchtsmixer__residual__hijri__L168__seed42.parquet

# Tier 2: residual heads on LightGBM + classical baselines + improved ensemble
python scripts/build_tier2_wins.py
# -> lgbm__{nohijri,hijri}__residual_h__seed44.parquet
# -> mstl_ets__hijri__residual_h__seed0.parquet
# -> sarimax__hijri__residual_h__seed0.parquet
# -> ensemble__top4_residual__seed0.parquet
# -> *__residual_per_regime__hijri__*.parquet  (4 documented null findings)

# Tier 3: meta-router + stacked LGBM
python scripts/build_tier3_wins.py
# -> meta_router__seed0.parquet
# -> meta_router_v2__seed0.parquet  (current champion, MAE 838.8)
# -> stacked_lgbm__seed0.parquet
```

Wall-clock: ~30 s total — all CPU, all use existing prediction
parquets as input.

---

## 8. Plan 7 — statistical appendix + deep analysis + failure modes

```bash
python scripts/build_statistical_appendix.py
# -> docs/statistical_appendix.md
# -> data/statistical_appendix/{ci_table, dm_aggregate, dm_Normal, dm_Ramadan, dm_Heatwave}.csv

python scripts/build_deep_analysis.py
# -> docs/deep_analysis.md
# -> data/analysis/{horizon_mae, diurnal_mae}.csv

python scripts/build_failure_modes.py
# -> docs/failure_modes.md
# -> data/analysis/failure_modes_{per_model,common}.csv
```

Wall-clock: ~6-10 min for the statistical appendix (bootstrap CIs +
pairwise DM tests over 31 models × 4 regimes); seconds for the other
two scripts.

---

## 9. Figures

```bash
python scripts/build_figures.py
# -> docs/figures/{fig1_leaderboard_forest, fig2_per_horizon, fig3_diurnal_heatmap,
#                  fig4_per_regime_bars, fig5_residual_impact, fig6_failure_days}.png
```

Wall-clock: <30 s.

---

## 10. Full clean-room rebuild

For a complete rebuild from scratch on a clean checkout:

```bash
# 1. environment
pip install -e . && pip install -r requirements.txt
cp .env-example .env

# 2. data
python -m src.data.preprocess_epias
python -m src.data.spatial_weights
python -c "from src.data.spatial_weights import build_southern_temp_series, PROCESSED_DIR; s = build_southern_temp_series(); s.reset_index().rename(columns={'index':'timestamp'}).to_csv(PROCESSED_DIR/'southern_temp.csv', index=False)"
python -m src.data.build_v2_dataset

# 3. models — see sections 2 through 7 above.
#    (~90 min GPU + ~5 h CPU for the complete model cohort.)

# 4. analyses
python scripts/build_statistical_appendix.py
python scripts/build_deep_analysis.py
python scripts/build_failure_modes.py
python scripts/build_figures.py

# 5. verify
pytest -q   # 152 passed
```

---

## 11. Predictions parquet inventory

| Family | Count | Naming pattern |
|---|---|---|
| LightGBM | 15 | `lgbm__<variant>__seed<s>.parquet` (variants: nohijri / hijri / hijri_plusB; seeds 42-46) |
| TSFMs (Ablation C) | 16 | `<tsfm>__nohijri__L<L>__seed0.parquet` (4 TSFMs × 4 L) |
| TSFMs (Ablation A) | 2 | `<tsfm>__hijri__L336__seed0.parquet` (TimesFM, Moirai) |
| MSTL+ETS | 2 | `mstl_ets__<variant>__seed0.parquet` |
| SARIMAX | 3 | `sarimax__<variant>__seed0.parquet` |
| PatchTSMixer | 4 | `patchtsmixer__<variant>__L168__seed42.parquet` + L-probe nohijri seed0 at L∈{96,168} |
| LightGBM + residual | 2 | `lgbm__<variant>__residual_h__seed44.parquet` |
| Classical + residual | 2 | `<model>__hijri__residual_h__seed0.parquet` |
| TSFM + residual (Plan 6) | 8 | `<tsfm>__residual__<variant>__L<L>__seed0.parquet` |
| PatchTSMixer + residual | 1 | `patchtsmixer__residual__hijri__L168__seed42.parquet` |
| Per-regime separate residual (null finding) | 4 | `<tsfm>__residual_per_regime__hijri__L<L>__seed0.parquet` |
| Composite ensembles | 3 | `ensemble__top4__seed0.parquet`, `ensemble__top4_residual__seed0.parquet`, `routed__best_per_regime__seed0.parquet` |
| Composite routers (Tier 3) | 3 | `meta_router__seed0.parquet`, `meta_router_v2__seed0.parquet`, `stacked_lgbm__seed0.parquet` |

Total: ~65 prediction parquets on disk, of which 31 are in the
statistical appendix headline cohort.

---

## 12. Open / deferred work

Documented elsewhere; listed here for completeness so future
reproducers know what's NOT in the parquet store:

- **L=336 and L=720 PatchTSMixer probe runs** (~5 h GPU; only L=96 and
  L=168 nohijri-seed0 probe parquets are on disk).
- **5-seed × 3-variant PatchTSMixer headline grid** (~20 h GPU at
  L=168; project shipped with seed 42 only).
- **Chronos-Bolt-Base fine-tune** — script at
  [`scripts/finetune_chronos.py`](../scripts/finetune_chronos.py),
  rationale for skipping at [`docs/chronos_finetune.md`](chronos_finetune.md).
- **Plan-6 "v2" with full historical TSFM error training** (~40 min
  GPU; current Plan-6 uses 3-fold time-block CV within the test
  window).
- **`.docx` versions** of the markdown result docs for capstone
  submission.
