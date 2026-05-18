# TSFM Ablations + Context-Length Sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the proposal's TSFM ablation matrix by adding (a) Time-MoE-200M via manual autoregressive inference (rescuing the deferred 4th model), (b) the context-length sensitivity sweep `L ∈ {96, 168, 336, 720}` for all 4 TSFMs (Ablation C), and (c) Hijri-feature dynamic covariate variants for the two covariate-capable TSFMs TimesFM and Moirai (Ablation A on TSFMs). Produces the full benchmark grid in `data/predictions/` and a per-model context-length sensitivity table.

**Architecture:** Extend `src/models/tsfm/_adapter.py` with a `_forecast_batch_with_covariates` hook on `TSFMBase`. Time-MoE gets a manual AR loop in its `_forecast_batch` that bypasses the broken bundled `generate()`. TimesFM and Moirai wrappers grow `predict_with_covariates(test_df, context_length, covariate_cols)` methods. `scripts/run_tsfm.py` gains `--variant hijri` and a `--sweep-context-lengths` mode that runs L=96/168/336/720 in one invocation. Predictions parquets land at the same canonical path scheme used by Plan 2.

**Scope decisions (vs the full spec):**
- **All 4 TSFMs**: revives Time-MoE.
- **All 4 context lengths**: 96, 168, 336, 720.
- **Hijri covariate variant**: TimesFM, Moirai only (Chronos and Time-MoE are univariate-architecture; their Hijri ablation is post-hoc residual correction, deferred to Plan 5).
- **Same single seed** (0) — TSFMs deterministic.
- **Ablation B (Compound regime features)**: skipped — Compound regime is empty in 2018-2025 test window per Plan 1 finding. Re-test when Plan 6's longer historical dataset arrives.

**Tech Stack:** Same as Plan 2 (torch 2.4.1+cu124, transformers 4.48.3, chronos-forecasting 1.5.2, uni2ts 2.0.0, timesfm 2.0.0 from GitHub). No new dependencies.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-13-tsfm-ablations-completion-design.md` (§5.2, §6, §7)
- Plan 2 results: `docs/tsfm_zero_shot_baseline.md`
- Plan 2 plan: `docs/superpowers/plans/2026-05-13-tsfm-zero-shot-baseline.md`

---

## Phase 1 — Time-MoE Rescue (Manual Autoregressive Forward)

The bundled `generate()` is broken on transformers 4.48.3 because Time-MoE's
`prepare_inputs_for_generation` uses the pre-4.46 tuple-of-tuples Cache API.
Plan 3 bypasses `generate()` entirely with a manual AR loop calling
`model.forward(input_ids=...)` directly.

### Task 1.1: Discover Time-MoE forward output shape

**Files:**
- Create: `scripts/probe_time_moe.py` (temporary; delete after Task 1.2)

- [ ] **Step 1: Write the probe script**

`scripts/probe_time_moe.py`:
```python
"""One-shot script to inspect Time-MoE's forward output. Delete after."""
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Maple728/TimeMoE-200M",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda().eval()

ctx = torch.randn(2, 336, dtype=torch.bfloat16, device="cuda")
mean = ctx.mean(-1, keepdim=True)
std = ctx.std(-1, keepdim=True).clamp(min=1e-8)
ctx_n = (ctx - mean) / std

with torch.no_grad():
    out = model(input_ids=ctx_n)

print("type:", type(out).__name__)
if hasattr(out, "keys"):
    print("keys:", list(out.keys()))
for attr in ["predictions", "logits", "last_hidden_state"]:
    if hasattr(out, attr):
        t = getattr(out, attr)
        if t is not None:
            print(f"  .{attr}: shape={t.shape}, dtype={t.dtype}")
```

- [ ] **Step 2: Run the probe**

Run:
```bash
.venv/Scripts/python.exe scripts/probe_time_moe.py
```

Expected output (record exactly what the model returns — the next task depends on it):
```
type: TimeMoeModelOutput (or similar)
keys: [...]
  .predictions: shape=torch.Size([2, 336, H_max]) ...
```

If `predictions` has shape `(B, L, H_max)` where `H_max ∈ {1, 8, 32, 64}`,
the model produces per-position multi-horizon forecasts; we take
`predictions[:, -1, :HORIZON]` as the 24-step forecast from the last position.

If `predictions` has shape `(B, L)` (single-step), we'll need true autoregressive
generation: append one step, re-run forward, repeat 24 times.

- [ ] **Step 3: Document findings in the wrapper module**

(No commit yet — just inspect.)

---

### Task 1.2: Implement Time-MoE manual forecast

**Files:**
- Modify: `src/models/tsfm/time_moe.py`
- Modify: `tests/models/test_time_moe.py`

- [ ] **Step 1: Rewrite the test to assert real inference works**

`tests/models/test_time_moe.py` (replace existing):
```python
import numpy as np
import pytest

from src.models.tsfm.time_moe import TimeMoEModel


def test_timemoe_model_attributes():
    m = TimeMoEModel()
    assert m.name == "time_moe_200m"
    assert m.needs_training is False
    assert m.supports_dynamic_covariates is False  # channel-independent


def test_timemoe_forecast_batch_shape_smoke():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    m = TimeMoEModel()
    rng = np.random.default_rng(0)
    contexts = rng.normal(size=(4, 336)).astype(np.float32)
    out = m._forecast_batch(contexts)
    assert out.shape == (4, 24)
    assert np.isfinite(out).all()
```

- [ ] **Step 2: Run test, verify it fails with NotImplementedError**

Run: `.venv/Scripts/python.exe -m pytest tests/models/test_time_moe.py::test_timemoe_forecast_batch_shape_smoke -v`
Expected: FAIL with `NotImplementedError: ... deferred to Plan 3 ...`.

- [ ] **Step 3: Rewrite `src/models/tsfm/time_moe.py` using the manual approach**

The body depends on the probe finding from Task 1.1. The two branches:

**If probe showed `predictions` shape `(B, L, H_max)` with H_max ≥ 24** —
single-shot multi-horizon head:
```python
"""Time-MoE 200M zero-shot wrapper.

Model: Maple728/TimeMoE-200M (Mixture-of-Experts decoder).
Plan 3 rescue: bypass the bundled generate() (broken on transformers 4.48.3)
and call model.forward() directly; the 32-step prediction head gives us a
24-step forecast in one pass.
"""
from __future__ import annotations

import numpy as np
import torch

from ._adapter import TSFMBase, HORIZON


class TimeMoEModel(TSFMBase):
    name = "time_moe_200m"
    supports_dynamic_covariates = False

    def __init__(
        self,
        checkpoint: str = "Maple728/TimeMoE-200M",
        batch_size: int = 32,
    ):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self._model = None
        self._device = None
        self._dtype = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device).eval()
        self._device = device
        self._dtype = dtype

    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """contexts: (B, L). Returns (B, HORIZON) point forecast via the
        32-step prediction head, sliced to 24 steps."""
        self._load()
        all_blocks: list[np.ndarray] = []
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(contexts), bs):
                batch = torch.tensor(
                    contexts[i : i + bs], dtype=torch.float32, device=self._device
                )
                mean = batch.mean(dim=1, keepdim=True)
                std = batch.std(dim=1, keepdim=True).clamp(min=1e-8)
                batch_norm = ((batch - mean) / std).to(self._dtype)
                out = self._model(input_ids=batch_norm)
                # predictions shape: (B, L, H_max). Take last position's head.
                preds = out.predictions[:, -1, :HORIZON].float()
                denorm = preds * std + mean
                all_blocks.append(denorm.cpu().numpy())
        return np.concatenate(all_blocks, axis=0)
```

**If probe showed `predictions` shape `(B, L)` (single-step only)** — true AR loop:
```python
    def _forecast_batch(self, contexts: np.ndarray) -> np.ndarray:
        """contexts: (B, L). Manual AR: append one step, re-run forward, repeat."""
        self._load()
        all_blocks: list[np.ndarray] = []
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(contexts), bs):
                batch = torch.tensor(
                    contexts[i : i + bs], dtype=torch.float32, device=self._device
                )
                mean = batch.mean(dim=1, keepdim=True)
                std = batch.std(dim=1, keepdim=True).clamp(min=1e-8)
                seq = ((batch - mean) / std).to(self._dtype)
                generated = []
                for _ in range(HORIZON):
                    out = self._model(input_ids=seq)
                    next_val = out.predictions[:, -1:]  # (B, 1)
                    generated.append(next_val)
                    seq = torch.cat([seq, next_val], dim=1)
                preds = torch.cat(generated, dim=1).float()  # (B, HORIZON)
                denorm = preds * std + mean
                all_blocks.append(denorm.cpu().numpy())
        return np.concatenate(all_blocks, axis=0)
```

Pick the branch matching the probe output. Replace the entire `time_moe.py`
file with the chosen branch's content (including the module docstring,
imports, and `__init__`).

- [ ] **Step 4: Run smoke test, verify PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/models/test_time_moe.py -v`
Expected: 2 passed.

If `out.predictions` is None or doesn't exist: re-run the probe to find the
correct output attribute (it may be `out.logits` or `out[0]` depending on
the model's output class).

- [ ] **Step 5: Commit**

```bash
git add src/models/tsfm/time_moe.py tests/models/test_time_moe.py
git rm scripts/probe_time_moe.py
git commit -m "feat(tsfm): Time-MoE-200M manual forecast (bypass broken generate)"
```

---

### Task 1.3: Run Time-MoE on the full test set at L=336

- [ ] **Step 1: Run via the script**

Run:
```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model timemoe --context-length 336
```
Expected:
- "test forecast hours: 10,944"
- ~10,944 predictions written
- Wall-clock 30-90s on RTX 4070 (faster with single-shot head, slower with AR loop)

- [ ] **Step 2: Sanity-check per-regime metrics**

Run:
```bash
.venv/Scripts/python.exe -c "
import pandas as pd
from src.evaluation.predictions_io import read_predictions
from src.evaluation.regime_eval import evaluate_by_regime

TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load'].values
p = read_predictions(model='time_moe_200m', variant='nohijri', context_length=336, seed=0)
print(f'rows: {len(p):,}')
print(evaluate_by_regime(p['y_true'].values, p['y_pred'].values, regimes=p['regime'], y_train=TRAIN, period=168).to_string(index=False))
"
```
Expected: 4-regime table with finite MAE in 800-3000 MW range for Normal/Ramadan/Heatwave. If MAE >5000, something is wrong with the wrapper (likely de-normalization or output indexing).

- [ ] **Step 3: Commit predictions**

```bash
git add data/predictions/time_moe_200m__nohijri__L336__seed0.parquet
git commit -m "feat(tsfm): run Time-MoE-200M on v2 test set at L=336"
```

---

## Phase 2 — Context-Length Sensitivity Sweep (Ablation C)

### Task 2.1: Add a sweep mode to `scripts/run_tsfm.py`

**Files:**
- Modify: `scripts/run_tsfm.py`

- [ ] **Step 1: Replace single-L flag with sweep**

Modify the argparse block from:
```python
    parser.add_argument("--context-length", type=int, required=True)
```
to:
```python
    parser.add_argument(
        "--context-length", type=int, action="append",
        help="Pass multiple times for a sweep: --context-length 96 --context-length 168 ...",
    )
```
And wrap the forecasting code in `main()` in a loop over `args.context_length`:
```python
    for L in args.context_length:
        print(f"[3/4] Forecasting (L={L}) ...")
        t0 = time.time()
        preds_all = model.predict(df, context_length=L)
        test_preds = preds_all.loc[test_window.index.intersection(preds_all.index)]
        elapsed = time.time() - t0
        print(f"      L={L} done in {elapsed:.1f}s  ({len(test_preds):,} test predictions)")

        print(f"[4/4] Writing parquet for L={L} ...")
        path = write_predictions(
            test_preds,
            model=model.name,
            variant=args.variant,
            context_length=L,
            seed=args.seed,
        )
        print(f"      -> {path}")
```

- [ ] **Step 2: Smoke-test the new CLI shape**

Run:
```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model chronos --context-length 96 --context-length 168
```
Expected: writes BOTH `chronos_bolt_base__nohijri__L96__seed0.parquet` and `chronos_bolt_base__nohijri__L168__seed0.parquet`. Each pass ~10-30s.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_tsfm.py
git commit -m "feat(tsfm): scripts/run_tsfm.py supports multi-L sweep in one invocation"
```

---

### Task 2.2: Run Chronos-Bolt full L sweep

(L=336 already done in Plan 2. This adds L=96, 168, 720.)

- [ ] **Step 1: Run**

Run:
```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model chronos --context-length 96 --context-length 168 --context-length 720
```
Expected: 3 new parquets written. L=720 takes longest (~60-90s on Chronos).

- [ ] **Step 2: Verify all 4 Chronos parquets exist**

Run:
```bash
ls data/predictions/chronos_bolt_base__nohijri__L*.parquet
```
Expected: 4 files at L=96, 168, 336, 720.

- [ ] **Step 3: Commit**

```bash
git add data/predictions/chronos_bolt_base__nohijri__L{96,168,720}__seed0.parquet
git commit -m "feat(tsfm): Chronos-Bolt L sweep predictions at L=96,168,720"
```

---

### Task 2.3: Run TimesFM full L sweep

- [ ] **Step 1: Run**

Run:
```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model timesfm --context-length 96 --context-length 168 --context-length 720
```
Expected: 3 new parquets. TimesFM is slower — total ~10-15 min.

Watch for: TimesFM has an internal `max_context` setting baked into the
`ForecastConfig.compile()` call. The wrapper currently sets it from the L
passed to `_forecast_batch`. If a fresh `TimesFMModel` is reused across L
values inside the same Python process, the first call's `max_context` may be
locked in. Workaround: the script instantiates a fresh `cls()` per invocation
of `model.predict()`, so this is only a concern if we sweep within a single
process. Verify by checking that the L=96 prediction file's `y_pred` range
is plausible (15000-50000 MW).

- [ ] **Step 2: Commit**

```bash
git add data/predictions/timesfm_2_5__nohijri__L{96,168,720}__seed0.parquet
git commit -m "feat(tsfm): TimesFM L sweep predictions at L=96,168,720"
```

---

### Task 2.4: Run Moirai full L sweep

- [ ] **Step 1: Run**

Run:
```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model moirai --context-length 96 --context-length 168 --context-length 720
```
Expected: 3 new parquets. ~5 min total.

Moirai's `patch_size=32` from Plan 2 works for L=96 (3 context patches), L=168 (5.25 patches — Moirai pads), L=720 (22.5 patches). If you get a shape mismatch error at L=96, switch that one run to `patch_size=16`. The wrapper currently has a hard-coded 32 in `__init__`; we accept any small inconsistency at the smallest context if Moirai needs patch_size=16 there.

- [ ] **Step 2: Commit**

```bash
git add data/predictions/moirai_1_1_small__nohijri__L{96,168,720}__seed0.parquet
git commit -m "feat(tsfm): Moirai L sweep predictions at L=96,168,720"
```

---

### Task 2.5: Run Time-MoE full L sweep

- [ ] **Step 1: Run**

Run:
```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model timemoe --context-length 96 --context-length 168 --context-length 720
```
Expected: 3 new parquets. ~5-10 min total (Time-MoE is one of the slowest).

- [ ] **Step 2: Commit**

```bash
git add data/predictions/time_moe_200m__nohijri__L{96,168,720}__seed0.parquet
git commit -m "feat(tsfm): Time-MoE L sweep predictions at L=96,168,720"
```

---

### Task 2.6: Build the L-sweep summary table

**Files:**
- Create: `docs/tsfm_context_length_sweep.md`

- [ ] **Step 1: Compute the table**

Run:
```bash
.venv/Scripts/python.exe -c "
import pandas as pd
from src.evaluation.predictions_io import read_predictions
from src.evaluation.regime_eval import evaluate_by_regime
from src.evaluation.metrics import mae

TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load'].values
MODELS = ['chronos_bolt_base', 'timesfm_2_5', 'moirai_1_1_small', 'time_moe_200m']
LS = [96, 168, 336, 720]

print(f'=== Aggregate MAE on the intersection of valid taus across all 4 L per model ===')
print(f'{\"model\":<22}' + ''.join(f'{f\"L={L}\":>10}' for L in LS))
for m in MODELS:
    p_by_L = {L: read_predictions(model=m, variant='nohijri', context_length=L, seed=0) for L in LS}
    shared = p_by_L[96].index
    for L in LS[1:]:
        shared = shared.intersection(p_by_L[L].index)
    row = [m]
    for L in LS:
        sub = p_by_L[L].loc[shared]
        row.append(f'{mae(sub.y_true, sub.y_pred):>10.1f}')
    print(f'{m:<22}' + ''.join(row[1:]))

print()
print(f'=== Per-regime MAE: best L per model ===')
for m in MODELS:
    print(f'\\n[{m}]')
    rows = []
    for L in LS:
        p = read_predictions(model=m, variant='nohijri', context_length=L, seed=0)
        tab = evaluate_by_regime(p['y_true'].values, p['y_pred'].values, regimes=p['regime'], y_train=TRAIN, period=168)
        tab.insert(0, 'L', L)
        rows.append(tab)
    print(pd.concat(rows).to_string(index=False))
" > /tmp/lsweep.txt 2>&1
cat /tmp/lsweep.txt
```

- [ ] **Step 2: Write `docs/tsfm_context_length_sweep.md`**

Use this template, pasting in the step-1 output where the placeholders go:

```markdown
# TSFM Context-Length Sensitivity Sweep (Ablation C)

Tests the proposal hypothesis that longer contexts spanning Ramadan
transitions improve regime-shift forecast accuracy.

## Setup

- 4 TSFMs × 4 context lengths (L ∈ {96, 168, 336, 720}) = 16 inference passes.
- Univariate framing throughout (Hijri covariates are Ablation A, see
  `docs/tsfm_hijri_covariates.md`).
- Single seed = 0.

## Aggregate MAE (intersection-tau across all L per model)

(paste step-1 aggregate table here)

## Per-regime MAE by L (best L bolded per row)

(paste step-1 per-regime tables here)

## Findings to write

After seeing the table, write 3-5 sentences answering:
1. Does each model show monotone improvement with longer L, or a U-shape?
2. Does longer L help disproportionately on Ramadan vs Normal hours? (The
   spec's hypothesis: yes — L=720 spans ~30 days, enough to capture a
   Ramadan transition.)
3. Which L is the sweet spot per model architecture? Tabular models like
   LGBM would have used implicit lag features; TSFMs use raw context.

## Files

- 16 prediction parquets at `data/predictions/<model>__nohijri__L{96,168,336,720}__seed0.parquet`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/tsfm_context_length_sweep.md
git commit -m "docs: context-length sweep summary (4 TSFMs x 4 L values)"
```

---

## Phase 3 — Hijri Dynamic Covariates (Ablation A on TSFMs)

TimesFM and Moirai natively accept dynamic real covariates over both context
AND forecast horizon. Inject `is_ramadan`, `day_of_ramadan`, `is_eid`, and
`temp_c` (all known at forecast time per proposal §2) and re-run at L=336.
Compare against the nohijri runs from Plan 2 / Phase 2.

### Task 3.1: Extend `TSFMBase` to accept covariate arrays

**Files:**
- Modify: `src/models/tsfm/_adapter.py`
- Modify: `tests/models/test_tsfm_base.py`

- [ ] **Step 1: Write failing test**

Append to `tests/models/test_tsfm_base.py`:
```python
class _FakeCovTSFM(TSFMBase):
    """Test double that uses covariates: returns mean(past_cov[0]) + last_y."""
    name = "fake_cov_tsfm"
    supports_dynamic_covariates = True
    needs_training = False

    def _forecast_batch(self, contexts):
        # Univariate fallback - not exercised here.
        return np.tile(contexts[:, -1:], (1, HORIZON))

    def _forecast_batch_with_covariates(self, contexts, past_cov, future_cov):
        """past_cov: (B, L, C), future_cov: (B, HORIZON, C). Returns (B, HORIZON)."""
        # Predict context_last + future_cov[:,:,0] (mean-zero arithmetic for test)
        last = contexts[:, -1:]  # (B, 1)
        return np.tile(last, (1, HORIZON)) + future_cov[:, :, 0]


def test_tsfm_predict_with_covariates_returns_unified_schema():
    df = _make_test_df(500)
    # Add a fake covariate column shaped to align with df.index.
    df["cov0"] = np.arange(500, dtype=float) * 0.01
    model = _FakeCovTSFM()
    out = model.predict_with_covariates(
        df, context_length=168, covariate_cols=["cov0"],
    )
    assert "y_true" in out.columns
    assert "y_pred" in out.columns
    assert "y_block" in out.columns
    assert len(out) == 309  # same dropoff as Plan 2 base test
    # _FakeCovTSFM forecast = last_context + future_cov_value
    # For row tau=191: last_context = y[167] = 167; future_cov_value at horizon[h]
    # is cov0[issuance+1+h] = cov0[168+h] for h in [0..23]. y_pred = block[-1] = 167 + cov0[191] = 167 + 1.91
    assert np.isclose(out["y_pred"].iloc[0], 167.0 + 1.91)
```

- [ ] **Step 2: Run test, verify fail**

Run: `.venv/Scripts/python.exe -m pytest tests/models/test_tsfm_base.py::test_tsfm_predict_with_covariates_returns_unified_schema -v`
Expected: AttributeError on `model.predict_with_covariates`.

- [ ] **Step 3: Implement covariate support in `TSFMBase`**

Append to `src/models/tsfm/_adapter.py` (in the `TSFMBase` class):
```python
    def _forecast_batch_with_covariates(
        self,
        contexts: np.ndarray,
        past_cov: np.ndarray,
        future_cov: np.ndarray,
    ) -> np.ndarray:
        """Override in subclasses that support dynamic real covariates.

        contexts:    (B, L)        target series context
        past_cov:    (B, L, C)     covariates aligned with context window
        future_cov:  (B, HORIZON, C) covariates aligned with horizon

        Returns (B, HORIZON) point forecast.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support dynamic covariates."
        )

    def predict_with_covariates(
        self,
        test_df: pd.DataFrame,
        context_length: int,
        covariate_cols: list[str],
    ) -> pd.DataFrame:
        """Like predict(), but feeds dynamic real covariates over context + horizon.

        covariate_cols must be columns of test_df aligned to the test_df index.
        """
        if not self.supports_dynamic_covariates:
            raise ValueError(
                f"{self.name} does not support dynamic covariates "
                "(supports_dynamic_covariates=False)."
            )

        contexts = build_context_windows(
            test_df["actual_load"], test_df.index, context_length=context_length,
        )
        valid_mask = ~np.isnan(contexts).any(axis=1)
        valid_contexts = contexts[valid_mask]

        # Build past_cov (B, L, C) and future_cov (B, HORIZON, C).
        cov_values = test_df[covariate_cols].values  # (T, C)
        cov_index = test_df.index
        positions = cov_index.get_indexer(test_df.index[valid_mask])

        L = context_length
        H = HORIZON
        C = len(covariate_cols)
        past_cov = np.zeros((len(positions), L, C), dtype=np.float32)
        future_cov = np.zeros((len(positions), H, C), dtype=np.float32)
        for i, tau_pos in enumerate(positions):
            issuance = tau_pos - ISSUANCE_OFFSET
            past_cov[i] = cov_values[issuance - L + 1 : issuance + 1]
            future_cov[i] = cov_values[issuance + 1 : issuance + 1 + H]

        blocks = self._forecast_batch_with_covariates(
            valid_contexts, past_cov, future_cov,
        )
        if blocks.shape != (len(valid_contexts), H):
            raise AssertionError(
                f"_forecast_batch_with_covariates returned {blocks.shape}, "
                f"expected ({len(valid_contexts)}, {H})"
            )

        return pd.DataFrame({
            "y_true": test_df["actual_load"].values[valid_mask],
            "y_pred": blocks[:, -1],
            "y_block": [b.tolist() for b in blocks],
            "regime": test_df["regime"].values[valid_mask],
        }, index=test_df.index[valid_mask])
```

- [ ] **Step 4: Run tests, verify pass**

Run: `.venv/Scripts/python.exe -m pytest tests/models/test_tsfm_base.py -v`
Expected: 5 passed (4 from Plan 2 + 1 new covariate test).

- [ ] **Step 5: Commit**

```bash
git add src/models/tsfm/_adapter.py tests/models/test_tsfm_base.py
git commit -m "feat(tsfm): predict_with_covariates() in TSFMBase (past + future dynamic real)"
```

---

### Task 3.2: TimesFM covariate-capable forecast

**Files:**
- Modify: `src/models/tsfm/timesfm.py`

TimesFM 2.5's `model.forecast()` method has a `forecast_with_xreg` API.
Library signature (verify with `inspect.signature` before implementing):
```python
model.forecast_with_xreg(
    horizon=24,
    inputs=list_of_target_arrays,
    dynamic_numerical_covariates=dict[name, list_of_past_cov_arrays],
    dynamic_categorical_covariates=dict[name, list_of_past_cov_arrays],
    future_dynamic_numerical_covariates=dict[name, list_of_future_cov_arrays],
    ...,
)
```
The exact kwarg names vary between TimesFM versions; the wrapper below uses
the most common variant. If the actual signature differs, fix at runtime.

- [ ] **Step 1: Discover the covariate API**

Run:
```bash
.venv/Scripts/python.exe -c "
from timesfm import TimesFM_2p5_200M_torch
import inspect
m_cls = TimesFM_2p5_200M_torch
for name in dir(m_cls):
    if 'xreg' in name.lower() or 'covariate' in name.lower() or 'forecast' in name.lower():
        attr = getattr(m_cls, name)
        if callable(attr):
            try:
                print(f'{name}: {inspect.signature(attr)}')
            except (TypeError, ValueError):
                pass
"
```
Record the exact method name and signature. Common names: `forecast_with_xreg`,
`forecast_with_covariates`, `predict`.

- [ ] **Step 2: Append `_forecast_batch_with_covariates` to `src/models/tsfm/timesfm.py`**

Inside the `TimesFMModel` class (after `_forecast_batch`):
```python
    def _forecast_batch_with_covariates(
        self,
        contexts: np.ndarray,
        past_cov: np.ndarray,
        future_cov: np.ndarray,
    ) -> np.ndarray:
        """contexts: (B, L); past_cov: (B, L, C); future_cov: (B, H, C).
        Returns (B, HORIZON)."""
        L = contexts.shape[1]
        self._load(max_context=L)

        forecast_inputs = [row.astype(np.float32) for row in contexts]
        B, _, C = past_cov.shape
        # Build per-covariate lists of length B; each entry is a 1D float array.
        # TimesFM expects covariates split per-feature.
        past_cov_dict = {
            f"cov{c}": [past_cov[b, :, c].astype(np.float32) for b in range(B)]
            for c in range(C)
        }
        future_cov_dict = {
            f"cov{c}": [future_cov[b, :, c].astype(np.float32) for b in range(B)]
            for c in range(C)
        }

        # Method name verified in Task 3.2 Step 1 — adjust if different.
        point_forecast, _quantiles = self._model.forecast_with_xreg(
            horizon=HORIZON,
            inputs=forecast_inputs,
            dynamic_numerical_covariates=past_cov_dict,
            future_dynamic_numerical_covariates=future_cov_dict,
        )
        return np.asarray(point_forecast)
```

If Step 1 revealed the API uses different kwarg names, edit the call site
accordingly. Save the file.

- [ ] **Step 3: Smoke-test covariate path**

Run:
```bash
.venv/Scripts/python.exe -c "
import numpy as np, pandas as pd
from src.models.tsfm.timesfm import TimesFMModel

df = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp')
df.index = df.index.tz_convert('UTC') if df.index.tz is not None else df.index.tz_localize('UTC')

# Small slice
slice_df = df.loc['2024-01-15':'2024-01-20']
m = TimesFMModel()
out = m.predict_with_covariates(
    df.loc[:'2024-01-20'],
    context_length=336,
    covariate_cols=['is_ramadan', 'day_of_ramadan', 'is_eid', 'temp_c'],
)
print(f'predictions: {len(out)}')
print(f'y_pred range: [{out.y_pred.min():.1f}, {out.y_pred.max():.1f}]')
print(f'MAE: {(out.y_true - out.y_pred).abs().mean():.1f}')
"
```
Expected: ~120 predictions; y_pred in plausible MW range; MAE plausibly
similar to or better than the nohijri TimesFM L=336 prediction.

If the call raises (signature mismatch, dict format wrong): fix and retry.
Don't move to the full run until the smoke succeeds.

- [ ] **Step 4: Commit**

```bash
git add src/models/tsfm/timesfm.py
git commit -m "feat(tsfm): TimesFM dynamic covariates (predict_with_covariates path)"
```

---

### Task 3.3: Run TimesFM Hijri variant at L=336

- [ ] **Step 1: Extend `scripts/run_tsfm.py` for the hijri variant**

Modify `main()` in `scripts/run_tsfm.py` — after the model instantiation,
inside the L loop:
```python
        if args.variant == "hijri":
            if not model.supports_dynamic_covariates:
                raise ValueError(
                    f"{model.name} does not support dynamic covariates; "
                    "use --variant nohijri."
                )
            preds_all = model.predict_with_covariates(
                df,
                context_length=L,
                covariate_cols=["is_ramadan", "day_of_ramadan", "is_eid", "temp_c"],
            )
        else:
            preds_all = model.predict(df, context_length=L)
```

- [ ] **Step 2: Run TimesFM with hijri variant**

Run:
```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model timesfm --context-length 336 --variant hijri
```
Expected: ~10,944 predictions, ~3-5 min, writes
`timesfm_2_5__hijri__L336__seed0.parquet`.

- [ ] **Step 3: Compare TimesFM nohijri vs hijri**

Run:
```bash
.venv/Scripts/python.exe -c "
import pandas as pd
from src.evaluation.predictions_io import read_predictions
from src.evaluation.regime_eval import evaluate_by_regime

TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load'].values

for variant in ['nohijri', 'hijri']:
    p = read_predictions(model='timesfm_2_5', variant=variant, context_length=336, seed=0)
    tab = evaluate_by_regime(p['y_true'].values, p['y_pred'].values, regimes=p['regime'], y_train=TRAIN, period=168)
    print(f'\nTimesFM 2.5 {variant}:')
    print(tab.to_string(index=False))
"
```
Expected: Ramadan MAE drops in the hijri variant; Heatwave roughly unchanged.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_tsfm.py data/predictions/timesfm_2_5__hijri__L336__seed0.parquet
git commit -m "feat(tsfm): TimesFM Hijri covariate variant at L=336"
```

---

### Task 3.4: Moirai covariate-capable forecast

**Files:**
- Modify: `src/models/tsfm/moirai.py`

Moirai's `MoiraiForecast` accepts `feat_dynamic_real` (future) and
`past_feat_dynamic_real` (past) as `(B, time, feat)` tensors. The
constructor takes `feat_dynamic_real_dim` and `past_feat_dynamic_real_dim`
to size the embedder. For the Hijri ablation we have 4 covariates known
over both windows, so we set both dims = 4.

- [ ] **Step 1: Append `_forecast_batch_with_covariates` to `src/models/tsfm/moirai.py`**

Inside the `MoiraiModel` class:
```python
    def _forecast_batch_with_covariates(
        self,
        contexts: np.ndarray,
        past_cov: np.ndarray,
        future_cov: np.ndarray,
    ) -> np.ndarray:
        """contexts: (B, L); past_cov: (B, L, C); future_cov: (B, H, C).
        Returns (B, HORIZON)."""
        self._load()
        from uni2ts.model.moirai import MoiraiForecast

        L = contexts.shape[1]
        C = past_cov.shape[2]
        forecaster = MoiraiForecast(
            module=self._module,
            prediction_length=HORIZON,
            context_length=L,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=C,
            past_feat_dynamic_real_dim=C,
        ).to(self._device).eval()

        past_target = torch.tensor(
            contexts[..., None], dtype=torch.float32, device=self._device
        )
        past_observed = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros(
            past_target.shape[:2], dtype=torch.bool, device=self._device,
        )
        past_feat = torch.tensor(past_cov, dtype=torch.float32, device=self._device)
        past_feat_observed = torch.ones_like(past_feat, dtype=torch.bool)
        # Moirai's feat_dynamic_real spans context + horizon concatenated.
        feat = torch.cat([past_feat, torch.tensor(future_cov, dtype=torch.float32, device=self._device)], dim=1)
        feat_observed = torch.ones_like(feat, dtype=torch.bool)

        all_blocks: list[np.ndarray] = []
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, past_target.shape[0], bs):
                pt = past_target[i : i + bs]
                po = past_observed[i : i + bs]
                pp = past_is_pad[i : i + bs]
                pf = past_feat[i : i + bs]
                pfo = past_feat_observed[i : i + bs]
                f = feat[i : i + bs]
                fo = feat_observed[i : i + bs]
                samples = forecaster(
                    past_target=pt,
                    past_observed_target=po,
                    past_is_pad=pp,
                    past_feat_dynamic_real=pf,
                    past_observed_feat_dynamic_real=pfo,
                    feat_dynamic_real=f,
                    observed_feat_dynamic_real=fo,
                )
                if samples.dim() == 4:
                    samples = samples.squeeze(-1)
                med = samples.float().median(dim=1).values
                all_blocks.append(med.cpu().numpy())
        return np.concatenate(all_blocks, axis=0)
```

- [ ] **Step 2: Smoke-test the covariate path**

Run:
```bash
.venv/Scripts/python.exe -c "
import pandas as pd
from src.models.tsfm.moirai import MoiraiModel

df = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp')
df.index = df.index.tz_convert('UTC') if df.index.tz is not None else df.index.tz_localize('UTC')
m = MoiraiModel()
out = m.predict_with_covariates(
    df.loc[:'2024-01-20'],
    context_length=336,
    covariate_cols=['is_ramadan', 'day_of_ramadan', 'is_eid', 'temp_c'],
)
print(f'predictions: {len(out)}')
print(f'y_pred range: [{out.y_pred.min():.1f}, {out.y_pred.max():.1f}]')
print(f'MAE: {(out.y_true - out.y_pred).abs().mean():.1f}')
"
```
Expected: ~120 predictions; MAE in 1000-3000 MW range (Moirai-Small is not strong).

If you get a shape error from MoiraiForecast: the `feat_dynamic_real` dim may
need to be (B, L+H, C) or (B, H, C) only — uni2ts versions differ. Adjust the
`feat = torch.cat([...])` line accordingly.

- [ ] **Step 3: Run Moirai Hijri variant at L=336**

Run:
```bash
.venv/Scripts/python.exe scripts/run_tsfm.py --model moirai --context-length 336 --variant hijri
```
Expected: ~10,944 predictions, ~2-5 min.

- [ ] **Step 4: Commit**

```bash
git add src/models/tsfm/moirai.py data/predictions/moirai_1_1_small__hijri__L336__seed0.parquet
git commit -m "feat(tsfm): Moirai Hijri covariate variant at L=336"
```

---

## Phase 4 — Ablation A Summary

### Task 4.1: Compare nohijri vs hijri across TimesFM and Moirai

**Files:**
- Create: `docs/tsfm_hijri_covariates.md`

- [ ] **Step 1: Build the comparison table**

Run:
```bash
.venv/Scripts/python.exe -c "
import pandas as pd
from src.evaluation.predictions_io import read_predictions
from src.evaluation.regime_eval import evaluate_by_regime
from src.evaluation.dm_test import dm_test, holm_bonferroni

TRAIN = pd.read_csv('data/processed/final_training_set_v2.csv', parse_dates=['timestamp']).set_index('timestamp').loc['2018':'2022']['actual_load'].values

COVARIATE_MODELS = ['timesfm_2_5', 'moirai_1_1_small']

p_values_raw = []
labels = []
for m in COVARIATE_MODELS:
    nh = read_predictions(model=m, variant='nohijri', context_length=336, seed=0)
    hh = read_predictions(model=m, variant='hijri',   context_length=336, seed=0)
    shared = nh.index.intersection(hh.index)
    nh, hh = nh.loc[shared], hh.loc[shared]
    print(f'\n=== {m} (n={len(shared):,}) ===')
    print('nohijri:'); print(evaluate_by_regime(nh.y_true, nh.y_pred, nh.regime, TRAIN, 168).to_string(index=False))
    print('hijri:');   print(evaluate_by_regime(hh.y_true, hh.y_pred, hh.regime, TRAIN, 168).to_string(index=False))
    # Per-regime DM tests
    for regime in ['Normal', 'Ramadan', 'Heatwave']:
        rmask = (nh['regime'] == regime).values
        if rmask.sum() < 30:
            continue
        stat, p = dm_test(nh.y_true.values[rmask], nh.y_pred.values[rmask], hh.y_pred.values[rmask], h=24)
        labels.append(f'{m}-{regime}')
        p_values_raw.append(p)
        delta = (nh.y_true[rmask] - nh.y_pred[rmask]).abs().mean() - (hh.y_true[rmask] - hh.y_pred[rmask]).abs().mean()
        print(f'  DM {regime}: stat={stat:+.3f} p_raw={p:.4f} delta_MAE={delta:+.1f}')

p_adj = holm_bonferroni(p_values_raw)
print('\n=== Holm-Bonferroni adjusted p-values ===')
for lab, p_raw, p_h in zip(labels, p_values_raw, p_adj):
    sig = '***' if p_h < 0.001 else '**' if p_h < 0.01 else '*' if p_h < 0.05 else ''
    print(f'  {lab:<30} p_raw={p_raw:.4f}  p_holm={p_h:.4f}  {sig}')
" > /tmp/ablation_a.txt 2>&1
cat /tmp/ablation_a.txt
```

- [ ] **Step 2: Write `docs/tsfm_hijri_covariates.md`**

Capture the step-1 output inside this template:

```markdown
# TSFM Ablation A: Hijri Dynamic Covariates

Tests whether feeding `is_ramadan`, `day_of_ramadan`, `is_eid`, and `temp_c`
as dynamic real covariates over both context and horizon improves TSFM
forecasts on covariate-capable models (TimesFM, Moirai). Chronos-Bolt and
Time-MoE are univariate-architecture; their Hijri ablation is post-hoc
residual correction (Plan 5).

## Setup

- 2 TSFMs (TimesFM 2.5-200M, Moirai-1.1-R-Small)
- L = 336, single seed = 0
- Covariates: `is_ramadan`, `day_of_ramadan`, `is_eid`, `temp_c` (all known
  at issuance time per proposal §2)

## Results

(paste step-1 per-regime tables here for both models)

## DM significance (Newey-West HAC, Holm-Bonferroni adjusted)

(paste step-1 DM table here)

## Headline expected pattern

(describe pattern after seeing numbers: e.g., "Ramadan MAE drops X% on
TimesFM but Y% on Moirai; significant at p_holm < 0.01 for ...")
```

- [ ] **Step 3: Commit**

```bash
git add docs/tsfm_hijri_covariates.md
git commit -m "docs: Ablation A summary (TimesFM + Moirai Hijri covariates at L=336)"
```

---

## Phase 5 — Updated Cross-Model Headline

### Task 5.1: Update `docs/tsfm_zero_shot_baseline.md` with full grid

**Files:**
- Modify: `docs/tsfm_zero_shot_baseline.md`

- [ ] **Step 1: Rewrite the doc with the full grid**

Replace the contents of `docs/tsfm_zero_shot_baseline.md` to include:
- A summary line noting Plan 3 added Time-MoE and L-sweep and Hijri variants.
- The full 4-TSFM × 4-L grid (from `tsfm_context_length_sweep.md`).
- The Hijri-covariate Δ table (from `tsfm_hijri_covariates.md`).
- A best-result row per model: the lowest-MAE (L, variant) combo per TSFM,
  compared against LightGBM `hijri` seed-44 on shared τ.

Use the runtime numbers from the live runs to fill the wall-clock section.

- [ ] **Step 2: Commit**

```bash
git add docs/tsfm_zero_shot_baseline.md
git commit -m "docs: refresh TSFM baseline doc with Plan 3 sweep + Hijri results"
```

---

## Phase 6 — Smoke Tests and Plan Wrap-Up

### Task 6.1: Extend `tests/test_smoke_pipeline.py`

**Files:**
- Modify: `tests/test_smoke_pipeline.py`

- [ ] **Step 1: Append the full-grid existence assertions**

Add to `tests/test_smoke_pipeline.py`:
```python
# Plan 3: TSFM context-length sweep (L=96, 168, 336, 720) nohijri × 4 models.
TSFM_MODELS_FULL = [
    "chronos_bolt_base",
    "timesfm_2_5",
    "moirai_1_1_small",
    "time_moe_200m",
]
TSFM_LS = [96, 168, 336, 720]


@pytest.mark.parametrize("model_name", TSFM_MODELS_FULL)
@pytest.mark.parametrize("L", TSFM_LS)
def test_tsfm_prediction_exists_sweep(model_name, L):
    p = PRED_DIR / f"{model_name}__nohijri__L{L}__seed0.parquet"
    assert p.exists(), f"Missing {p}. Re-run Plan 3 Phase 2."
    df = pd.read_parquet(p)
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert "y_block" in df.columns
    assert df["y_pred"].notna().all()
    assert all(len(b) == 24 for b in df["y_block"].head(20))


# Plan 3: Hijri-covariate variants for the 2 covariate-capable TSFMs at L=336.
COVARIATE_TSFMS = ["timesfm_2_5", "moirai_1_1_small"]


@pytest.mark.parametrize("model_name", COVARIATE_TSFMS)
def test_tsfm_hijri_prediction_exists(model_name):
    p = PRED_DIR / f"{model_name}__hijri__L336__seed0.parquet"
    assert p.exists(), f"Missing {p}. Re-run Plan 3 Phase 3."
    df = pd.read_parquet(p)
    assert "y_true" in df.columns
    assert "y_pred" in df.columns
    assert df["y_pred"].notna().all()
```

- [ ] **Step 2: Run smoke**

Run: `.venv/Scripts/python.exe -m pytest tests/test_smoke_pipeline.py -v`
Expected: prior 20 smoke tests pass + 16 new sweep tests + 2 hijri-covariate tests = 38 passing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke_pipeline.py
git commit -m "test: smoke checks for Plan 3 full TSFM grid (4 L × 4 models + 2 hijri)"
```

---

### Task 6.2: Update README milestone tracker

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Mark Plan 3 done**

Change:
```
- [x] Plan 2: TSFM zero-shot baseline (Chronos, TimesFM, Moirai at L=336; Time-MoE deferred)
- [ ] Plan 3: ...
```
to:
```
- [x] Plan 2: TSFM zero-shot baseline (Chronos, TimesFM, Moirai at L=336; Time-MoE deferred)
- [x] Plan 3: TSFM ablations + L sweep (Time-MoE rescue, L∈{96,168,336,720}, Hijri covariates)
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: mark Plan 3 milestone complete in README"
```

---

### Task 6.3: Full pytest green

- [ ] **Step 1: Run full suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: all tests pass (88 from Plan 2 + 16 sweep + 2 hijri + 1 covariate-base ≈ 107 passing).

- [ ] **Step 2: Commit any fix-ups**

If any test fails, fix and commit. Otherwise skip.

---

## Self-Review Checklist

Before claiming Plan 3 complete:

1. **All 16 sweep parquets exist:** 4 models × 4 L values, nohijri.
2. **Time-MoE has real predictions** at L=336 (not the NotImplementedError stub).
3. **Hijri-covariate parquets** exist for TimesFM and Moirai at L=336.
4. **`docs/tsfm_context_length_sweep.md`** has all 4 × 4 cells filled.
5. **`docs/tsfm_hijri_covariates.md`** has DM significance for at least 4 (model, regime) pairs.
6. **README** shows Plan 3 as `[x]`.
7. **Full pytest green** (≥ ~107 tests).

## Risks and Escalation

- **TimesFM `forecast_with_xreg` signature**: API has changed across versions; Step 3.2.1 must verify before implementing. If the method doesn't exist (only `forecast()`), report BLOCKED — the proposal-spec Hijri ablation on TimesFM is not feasible on the installed version.
- **Moirai patch_size mismatch at L=720**: 720/32 = 22.5 (non-integer). If it errors, try patch_size=16 or patch_size=8 for just L=720.
- **Time-MoE forward output shape**: if the probe shows neither `.predictions` nor `.logits` exposing a usable forecast, the manual approach is harder. Report NEEDS_CONTEXT with the actual output structure.
- **Disk space**: 16 new prediction parquets at ~250KB each = 4 MB; negligible.
- **VRAM at L=720**: largest context fits in 8GB at batch=32 for Chronos/Time-MoE; TimesFM may need batch=16; Moirai is fine. The harness has no auto-OOM fallback — if a model OOMs, drop its batch_size in the wrapper and re-run.
