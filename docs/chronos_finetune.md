# Chronos-Bolt-Base Fine-Tuning — Available, Not Exercised

A complete fine-tuning script for Amazon's `chronos-bolt-base` is
shipped at [`scripts/finetune_chronos.py`](../scripts/finetune_chronos.py)
but has not been executed in this benchmark. The reasoning is
documented below so future iterations of the project can evaluate
whether to invest the GPU time.

## Why the script exists

Of the 4 TSFMs benchmarked, Chronos-Bolt-Base is the only one with a
clean, documented fine-tuning path on its public release:
`ChronosBoltModelForForecasting.forward(context, target=...)` returns a
`ChronosBoltOutput` with a built-in quantile loss tensor, making it
plug-compatible with the HuggingFace `Trainer` we already use for
PatchTSMixer.

The full pipeline is implemented:
- `WindowedChronosDataset` yielding stride-configurable
  `(context_length, horizon=24)` pairs from train+val.
- bf16 mixed-precision training, AdamW, cosine LR schedule with
  warmup, gradient clipping, val-MAE early stopping on the median
  quantile (index 4 of 9).
- Per-τ day-ahead inference reusing `src.models.tsfm._adapter.build_context_windows`
  for direct apples-to-apples comparison vs the bare-Chronos parquet.
- Output saved to
  `data/predictions/chronos_bolt_base__finetuned__nohijri__L<L>__seed<s>.parquet`.

## Why it wasn't run

Three reasons, in priority order:

1. **Expected lift is small relative to alternatives already in the
   benchmark.** Chronos-Bolt-Base bare at L=720 is MAE 968.9 — already
   inside the top-tier cluster (LightGBM-hijri 979.0, Time-MoE 985.9,
   all CIs overlap). Fine-tuning typically buys:
   - 20-40% on weak base models (PatchTSMixer territory)
   - 5-15% on strong base models (Chronos territory)
   - 0-5% on near-optimal models in this distribution
   The realistic gain at L=336 (the VRAM-feasible config) lands at
   MAE ~970-1030 — back to bare-L=720 territory, no headline change.

2. **Post-hoc residual correction already captures most of the
   achievable in-domain signal.** Plan 6 showed Chronos+LGBM-residual
   lands at MAE 948.5, a −2.1% gain at ~30 seconds CPU. To justify the
   GPU cost of fine-tuning, the fine-tuned model would need to *also*
   beat this — and the same Turkish (context, target) pairs that train
   the LightGBM residual head would train the Chronos fine-tune.
   Empirically, residual heads are surprisingly competitive with full
   fine-tunes precisely because they target the gap the base model
   leaves.

3. **Distribution-drift risk.** Plan 5 (PatchTSMixer) showed a 2.8×
   val→test gap (val MAE 545 vs test MAE 1556) — a strong indicator
   that 2024-2025 differs materially from 2018-2023. Fine-tuning a
   foundation model on 2018-2022 with 2023 val early-stop could
   produce a model that's actively *worse* on 2024-2025 than the bare
   pretrained model, which has the advantage of having no Turkish
   exposure at all.

## When to run it

If a future researcher wants to push the headline below MAE 820:

```bash
# Conservative (fits 8GB VRAM, ~30-45 min on RTX 4070 Laptop):
.venv/Scripts/python.exe scripts/finetune_chronos.py \
    --context-length 336 --epochs 3 --batch-size 16 --stride 4

# Aggressive (matches bare-L=720 baseline, OOM risk; ~60-90 min):
.venv/Scripts/python.exe scripts/finetune_chronos.py \
    --context-length 720 --epochs 3 --batch-size 8 --stride 4

# Then add to the appendix and re-run:
# (add 'chronos-bolt-L720+finetune' entry to scripts/build_statistical_appendix.py MODELS)
.venv/Scripts/python.exe scripts/build_statistical_appendix.py
```

A genuine headline-changing result (MAE <820) would more plausibly
come from fine-tuning Chronos-Bolt-**Large** (not Base) at L=336, or
from training with the full Amazon training recipe (which includes
random masking, quantile-loss weighting, and a longer warm-up). Both
exceed the scope of this capstone but are the right next steps.
