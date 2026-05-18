# PatchTSMixer Baseline — Design Spec

**Date:** 2026-05-14
**Plan:** 5 (deep-learning baseline)
**Branch:** `plan-5-patchtst` off `main` after the Plan 4 merge lands.

## Goal

Add a deep-learning baseline trained from scratch on Turkish load to the
benchmark, slotting it between LightGBM/TSFMs (top tier) and the
classical baselines (MSTL+ETS, SARIMAX) in the headline table.

The original proposal §4.2 specified PatchTST. We use **PatchTSMixer**
(HuggingFace `transformers.models.patchtsmixer`) instead — the
multivariate cross-channel-mixing sibling of PatchTST by the same
authors. Reason: HuggingFace's `PatchTSTForPrediction` is
channel-independent by design (same weights process each channel
separately, no cross-channel attention), so feeding `is_ramadan` etc.
as auxiliary channels would not actually let them condition the y
forecast — the Hijri ablation would be a no-op. PatchTSMixer's
`mix_channel` mode performs MLP-Mixer-style cross-channel mixing,
which makes the Hijri-covariate ablation meaningful for a
deep-learning model — the central proposal question.

Headline references will use "PatchTSMixer" with a footnote naming the
substitution. Filename: `src/models/dl/patchtsmixer.py`.

## Scope

19 total runs, ~5–10h wall-clock on RTX 4070 Laptop (8GB VRAM):

| Phase            | Runs | Description |
|------------------|------|-------------|
| L-probe          | 4    | `nohijri × seed 0 × L ∈ {96, 168, 336, 720}` to pick best L |
| Headline grid    | 15   | `3 variants × 5 seeds (42–46)` at best L |

## Architecture

`PatchTSMixerForPrediction` with config:

| Field                | Value         | Rationale |
|----------------------|---------------|-----------|
| `patch_length`       | 16            | PatchTST paper default; ~2/3 of a day at hourly resolution |
| `patch_stride`       | 8             | 50% overlap, paper default |
| `d_model`            | 128           | Proposal §4.2 |
| `num_layers`         | 3             | Proposal §4.2 |
| `expansion_factor`   | 2             | PatchTSMixer default |
| `dropout`            | 0.1           | Proposal §4.2 |
| `head_dropout`       | 0.1           | Same |
| `mode`               | `mix_channel` | **The whole point of choosing PatchTSMixer** — enables cross-channel mixing so Hijri/weather channels condition y |
| `scaling`            | `std`         | RevIN-style per-instance normalization; robust to level shifts |
| `prediction_length`  | 24            | Day-ahead horizon |
| `num_input_channels` | 7 / 10 / 12   | Per variant (see channels table) |
| `prediction_channel_indices` | `[0]` | Predict only the y channel from the multivariate output; other channels are computed but discarded |
| `loss`               | `mse`         | HuggingFace `PatchTSMixerConfig.loss` only supports `mse` or `nll`. We use MSE for training and MAE for eval — standard train/eval mismatch; if it materially hurts headline numbers, the fallback is a custom `Trainer.compute_loss` override for L1 |
| `context_length`     | {96,168,336,720} | Probed then fixed at best |

## Input channels per variant

| Variant       | Channels (in order)                                                          | n  |
|---------------|------------------------------------------------------------------------------|----|
| `nohijri`     | `y, temp_c, dewpoint_c, wind_speed, solar_rad, temp_sq, temp_above_35`       | 7  |
| `hijri`       | + `is_ramadan, day_of_ramadan, is_eid`                                        | 10 |
| `hijri_plusB` | + `ramadan_x_heatwave, ramadan_x_temp_above_35`                              | 12 |

`temp_sq` and `temp_above_35` are kept (matching LightGBM features), so
PatchTSMixer-hijri vs LightGBM-hijri is a clean deep-vs-tree comparison
on the same feature set. The `hijri_plusB` features are expected to be
structurally inactive (Compound regime empty 2018–2025) — included for
proposal alignment; predictions will be statistically indistinguishable
from `hijri` (consistent with the SARIMAX finding in Plan 4).

## Day-ahead protocol (inference)

Per-τ inference (matches TSFM pattern):

```
for each test τ:
    context = all_channels[τ - 24 - L : τ - 24]      # shape (L, num_channels)
    forecast = model(past_values=context, num_targets=1)  # shape (24,)
    y_pred[τ] = forecast[23]                          # last step = horizon-24
    y_block[τ] = forecast                              # full 24-step for analysis
```

10,944 test τ values batch into ~43 mini-batches of 256. Total inference
~seconds per run on the 4070.

## Training protocol

**Sampling:** every valid (context, target) pair from train (2018–2022) +
val (2023):
- Context: `(y, exog)[t-L : t]` (L hours, all channels)
- Target: `y[t+1 : t+25]` (24 hours, y-only)
- Chronological split: training pairs use `t+24 ≤ 2022-12-31 23:00`;
  val pairs use `t+24 ∈ 2023-01-01 .. 2023-12-31`.
- At L=336: ~43k train samples + ~8.4k val samples per epoch.

**Optimizer + schedule (from proposal §4.2):**
- AdamW, `lr = 1e-4`, `weight_decay = 1e-2`
- Cosine schedule with 500 warmup steps
- Batch size 32, gradient clip 1.0
- Max 100 epochs, early stop on val MAE, patience 10
- bf16 mixed precision (4070 supports it natively)
- HuggingFace `Trainer` (gets us logging, EarlyStoppingCallback,
  mixed precision, gradient clipping for free)

**Loss:** MSE (the only `PatchTSMixerConfig.loss` value that ships with
HuggingFace `Trainer`-compatible gradients besides `nll`). Eval is MAE
per the rest of the benchmark — standard train/eval mismatch, accepted
for the simpler `Trainer` integration. If the L-probe shows a material
hit vs LightGBM at the same feature set, fallback is a custom
`compute_loss` override returning `L1Loss(pred, future_values)`.

**Reproducibility:** Per run, seed `torch`, `numpy`, `random`, and call
`transformers.set_seed(s)`. Seeds for the headline grid:
`{42, 43, 44, 45, 46}` — matches LightGBM convention.

**Wall-clock (estimated):**
- ~43k samples / batch 32 ≈ 1,350 steps/epoch
- ~30ms/step on 4070 with L=336, d=128 → ~40s/epoch
- 100 epochs (often early-stops at 30–50) → 15–25 min/run
- L=720: roughly 2× slower → 30–45 min/run
- Total 19 runs: **5–10h wall-clock**

Fallback if too slow: `batch_size = 64` or `max_epochs = 50`.

## Evaluation and reporting

**Predictions schema** — `data/predictions/patchtsmixer__<variant>__L<ctx>__seed<s>.parquet`
with columns `y_true, y_pred, y_block, regime`. Same schema as TSFMs,
consumed unchanged by `src/evaluation/{regime_eval, dm_test, bootstrap,
predictions_io}`.

**Per-seed reduction:** 5 seeds × 3 variants at best L. Headline uses
the median-seed parquet (seed 44); supplementary table shows
mean ± std across seeds. Matches LightGBM reporting.

**Diebold-Mariano tests (HAC, h=24, Holm-Bonferroni adjusted):**
- PatchTSMixer-hijri vs LightGBM-hijri (agg, Ramadan, Heatwave)
- PatchTSMixer-hijri vs Chronos-Bolt-L720 (agg, Heatwave)
- Within-PatchTSMixer: nohijri vs hijri (does Hijri help when the model
  can actually exploit it cross-channel?)
- Within-PatchTSMixer: hijri vs hijri_plusB (expected: indistinguishable)

## Files

**Added:**
- `src/models/dl/patchtsmixer.py` — `PatchTSMixerModel` conforming to
  `src/models/base.py::Model` protocol (`name="patchtsmixer"`,
  `supports_dynamic_covariates=True`, `needs_training=True`, with
  `fit(train_df, val_df, hijri, seed)` and `predict(test_df,
  context_length)` methods).
- `src/models/dl/__init__.py` — re-export `PatchTSMixerModel`.
- `scripts/run_patchtsmixer.py` — CLI runner (`--variant`,
  `--context-length`, `--seed`).
- `tests/models/test_patchtsmixer.py` — unit tests (model attributes,
  variant rejects unknown, feature set per variant, predict returns
  unified schema on synthetic data).
- `docs/patchtsmixer_baseline.md` — results doc (L-probe table,
  headline grid, regime metrics, DM tests, runtime).

**Modified:**
- `tests/test_smoke_pipeline.py` — extend with `patchtsmixer__*` parquet
  existence checks (19 parametrized cases: 4 L-probe + 15 grid).
- `README.md` — mark Plan 5 complete, link to results doc.
- `docs/tsfm_zero_shot_baseline.md` — add PatchTSMixer rows to
  aggregate and per-regime headline tables.

**No dep changes:** `transformers 4.48.3` already includes
`PatchTSMixerForPrediction`. No `pyproject.toml` or requirements edits
needed.

## Risk register

| Risk | Mitigation |
|------|-----------|
| L=720 with d=128 might OOM on 8GB VRAM | Fallback to `batch_size=16` or gradient accumulation; L=720 is a probe run, so OOM is recoverable |
| Training instability at lr=1e-4 (loss explodes) | Gradient clip already at 1.0; warmup 500 steps; can drop to lr=5e-5 |
| HuggingFace `Trainer` early-stopping patience burns full 100 epochs | Patience=10 on val MAE → typically stops by epoch 30–50 in practice |
| `mix_channel` mode regresses vs channel-independent because cross-channel mixing overfits Hijri features | Documented as a finding; we already have channel-independent as a known fallback |
| Day-ahead protocol per-τ requires 10,944 forward passes | Batched 256 at a time = 43 batches, <30s on 4070 |

## Open questions (deferred to plan)

- Exact training-set window: do we use **only** `(t, t+24)` pairs where
  `t` is in train, or include sliding-window samples with overlap? The
  default in the plan will be full sliding window (one sample per hour)
  for maximum data; this is the standard PatchTST/PatchTSMixer
  convention.
- Whether to use HuggingFace `Trainer` directly or wrap in a thin
  training loop. Default: `Trainer` for the smoke checks; if it
  proves restrictive, switch to a 40-line PyTorch loop.
- Whether the L-probe should use 1 or 3 epochs of validation to be
  representative. Default: full early-stop training to convergence —
  the L-probe is part of the budget, not a quick prototype.
