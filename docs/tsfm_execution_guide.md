# TSFM Execution Guide: A100 (RunPod) and Local (4070 mobile)

Concrete step-by-step instructions for running the TSFM zero-shot inference and ablations on either a rented A100 (RunPod) or a local RTX 4070 mobile (8GB VRAM). Both paths produce identical artifacts in `data/predictions/`.

> Read this together with `docs/superpowers/specs/2026-05-13-tsfm-ablations-completion-design.md`. The design doc explains the *what*; this doc explains the *how*.

---

## Decision: which path?

| Criterion | A100 path | Local 4070 mobile |
|---|---|---|
| TSFM total wall-clock | 6–10h single session | 24–36h across two overnight sessions |
| Time-MoE variant | Time-MoE-Large (strict proposal compliance) | Time-MoE-200M (documented substitution) |
| PatchTST training | ~1h | ~7.5h |
| Hardware cost | ~$10–25 in cloud credits | $0 incremental |
| Setup overhead | ~30 min one-time RunPod setup | Already set up |
| Best for | One-shot full benchmark run | Iterative dev + final fallback |

**Recommended:** develop and debug everything locally; do the final ablation C and Time-MoE-Large runs on A100.

---

## Path A — RunPod A100

### A.1 Account and template setup
1. Create a RunPod account at https://runpod.io. Add $25 in credits (covers full ablation C with margin).
2. Generate an SSH keypair locally if you don't have one: `ssh-keygen -t ed25519`. Upload the public key under Settings → SSH Public Keys.
3. Choose a template: **"RunPod PyTorch 2.4"** (or the latest PyTorch 2.x template). It ships with CUDA 12.x, JupyterLab, and a pre-warmed PyTorch install.

### A.2 Pod creation
1. From the GPU Pods tab, pick **A100 80GB SXM** (preferred) or **A100 80GB PCIe** (cheaper, slightly slower). 40GB is fine if 80GB is unavailable; you'll just batch smaller.
2. Volume disk: **50 GB** is enough (HuggingFace model cache + dataset + predictions). Set to persistent if you may resume the next day.
3. Container disk: 20 GB.
4. Expose ports: 8888 (Jupyter), 22 (SSH). Both should be exposed by the template defaults.
5. Pod type: **On-Demand** for the final benchmark (don't get preempted mid-run). Spot is fine for dev.
6. Click Deploy.

### A.3 Connect
Once the pod is "Running" (typically <90s):
```bash
# SSH (replace with your pod's connect string from the Connect button)
ssh root@<pod-ip> -p <ssh-port> -i ~/.ssh/id_ed25519
```
Or open JupyterLab via the pod's HTTPS link.

### A.4 Repo and environment bootstrap
On the pod:
```bash
cd /workspace
git clone <your-github-url> ramadan-stlf
cd ramadan-stlf

# Use the project's pinned environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-tsfm.txt   # split file: torch + transformers + uni2ts + chronos-forecasting + timesfm

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA A100-SXM4-80GB  (or PCIe equivalent)
```

### A.5 Upload the processed dataset
The raw NetCDF files are huge (~1.7 GB) — don't re-upload them. Only the merged training CSV is needed.

From your local machine:
```bash
scp -P <ssh-port> -i ~/.ssh/id_ed25519 \
    data/processed/final_training_set_v2.csv \
    data/processed/final_training_set_v2.meta.json \
    root@<pod-ip>:/workspace/ramadan-stlf/data/processed/
```

Alternative for repeat use: push `final_training_set_v2.csv` to a private HuggingFace Dataset or S3 bucket; `wget` it on the pod. Faster on large transfers and avoids re-uploading per session.

### A.6 Pre-warm the HuggingFace model cache
Models download on first use (some are several GB). Pre-warm before kicking off the long run so the first inference isn't blocked on a download:
```bash
export HF_HOME=/workspace/hf-cache
python scripts/prewarm_tsfm_cache.py
# downloads: chronos-bolt-base, timesfm-2.0-500m-pytorch,
#            moirai-1.1-R-large, TimeMoE (large)
```

Expected: ~15 GB total in the cache after this completes.

### A.7 Run the TSFM ablations
```bash
# Smoke test on 200 test rows for each model (sanity check before the long run)
python scripts/run_all.py --smoke-test --models chronos timesfm moirai timemoe

# Full ablation C (the big one)
python scripts/run_all.py --ablation C \
    --models chronos timesfm moirai timemoe \
    --context-lengths 96 168 336 720 \
    --host runpod_a100 \
    --output-dir data/predictions/

# Ablation A for TimesFM and Moirai (Hijri covariate variant)
python scripts/run_all.py --ablation A \
    --models timesfm moirai \
    --host runpod_a100

# Residual correction prep: run Chronos and Time-MoE on train+val splits
python scripts/run_all.py --residual-prep \
    --models chronos timemoe \
    --host runpod_a100
```

Tip: run inside `tmux` or `screen` so the session survives if your laptop disconnects:
```bash
tmux new -s tsfm
# inside tmux:
python scripts/run_all.py --ablation C ...
# detach with Ctrl-b then d; reattach later with: tmux attach -t tsfm
```

### A.8 Download predictions back to local
After the run finishes, from your local machine:
```bash
rsync -avz -e "ssh -p <ssh-port> -i ~/.ssh/id_ed25519" \
    root@<pod-ip>:/workspace/ramadan-stlf/data/predictions/ \
    ./data/predictions/
```

Verify the manifest:
```bash
python -c "import json; m = json.load(open('data/predictions/_manifest.json')); print(json.dumps(m, indent=2))"
```
You should see entries for every (model, variant, context_length, seed) tuple with `host: runpod_a100`.

### A.9 Tear down the pod
1. **Run the residual-correction LGBM head training locally** (CPU-bound) before tearing down — no reason to keep paying GPU $/h for this.
2. From RunPod dashboard, stop the pod. If you set the volume persistent and may resume, leave it; otherwise destroy.
3. Confirm credits remaining match expectation.

**Cost estimate (May 2026 RunPod pricing):** A100 80GB SXM at ~$1.89/h on-demand × 8h = ~$15. Add ~$2 for storage and idle bootstrap.

---

## Path B — Local RTX 4070 mobile (8GB VRAM)

### B.1 Environment bootstrap (one-time)
```powershell
# from the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-tsfm.txt
```

Verify GPU:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX 4070 Laptop GPU
```

### B.2 HuggingFace cache configuration
The default Windows cache path can blow up your C:\ drive. Point it elsewhere if you have a secondary disk:
```powershell
# Add to .env (loaded by the project's dotenv setup) or system environment
$env:HF_HOME = "D:\hf-cache"   # adjust path
```

### B.3 Pre-warm model cache
```powershell
python scripts/prewarm_tsfm_cache.py
# downloads Chronos-Bolt-Base, TimesFM 2.0, Moirai-1.1-R-Large, Time-MoE-200M
# (~8 GB total — Time-MoE-Large is skipped on this path)
```

### B.4 Smoke test
```powershell
python scripts/run_all.py --smoke-test --models chronos timesfm moirai timemoe
```
Each model should complete a 200-row test slice in <2 minutes. Confirms env, GPU, and dataset are wired up.

### B.5 Run schedule (recommended: split across two overnight sessions)

**Night 1 — Chronos and Time-MoE (univariate, no Hijri ablation in this run)**
```powershell
python scripts/run_all.py --ablation C `
    --models chronos timemoe `
    --context-lengths 96 168 336 720 `
    --host local_4070m `
    --bf16 --batch-size 64
```
Wall-clock estimate: 8–12h. Run before going to bed.

**Night 2 — TimesFM and Moirai (covariate-capable, both Hijri variants)**
```powershell
python scripts/run_all.py --ablation C `
    --ablation A `
    --models timesfm moirai `
    --context-lengths 96 168 336 720 `
    --host local_4070m `
    --bf16 --batch-size 32
```
Wall-clock estimate: 12–18h. (Smaller batch because we feed dynamic covariates, which raises per-sample memory.)

### B.6 Residual-correction prep
After the test predictions are saved:
```powershell
python scripts/run_all.py --residual-prep `
    --models chronos timemoe `
    --host local_4070m
```
This runs Chronos and Time-MoE on the train+val splits to compute residuals for the post-hoc LGBM head. Fast on GPU (<30 min total).

Then train the residual head on CPU:
```powershell
python scripts/train_residual_head.py --models chronos timemoe
```

### B.7 OOM recovery
If a model OOMs mid-run:
1. Drop `--batch-size` by half.
2. If still OOM, drop the largest context length first (L=720) and re-run with `--context-lengths 96 168 336` only. Add L=720 in a separate pass with batch=16.
3. Last resort: switch the offending model to its smaller variant (e.g., `Moirai-Base` instead of `Moirai-Large`) and add a footnote to the final report.

The harness auto-retries with halved batch size on OOM by default; manual override only needed if persistent.

### B.8 Expected per-model runtimes (4070 mobile, bf16)

| Model | L=96 | L=168 | L=336 | L=720 |
|---|---|---|---|---|
| Chronos-Bolt-Base | 25 min | 35 min | 60 min | 100 min |
| TimesFM 2.0 (univariate) | 40 min | 55 min | 90 min | 150 min |
| TimesFM 2.0 (with covariates) | 55 min | 75 min | 120 min | 200 min |
| Moirai-1.1-Large (univariate) | 60 min | 80 min | 130 min | 220 min |
| Moirai-1.1-Large (with covariates) | 75 min | 100 min | 160 min | 280 min |
| Time-MoE-200M | 30 min | 40 min | 70 min | 120 min |

These are rough estimates; first-run numbers establish the actual budget. The harness logs runtime to the manifest after every pass.

---

## PatchTST training (GPU-bound, A100 strongly preferred)

PatchTST is small enough that it can train on either path, but the runtime gap is large.

**A100 path** — same pod as the TSFM session:
```bash
python scripts/run_all.py --model patchtst \
    --variants nohijri hijri hijri_plusB \
    --seeds 42 43 44 45 46 \
    --host runpod_a100
```
Wall-clock: ~1.5–2.5h for all 15 runs.

**Local 4070 mobile**:
```powershell
python scripts/run_all.py --model patchtst `
    --variants nohijri hijri hijri_plusB `
    --seeds 42 43 44 45 46 `
    --host local_4070m
```
Wall-clock: ~7.5h. Best run overnight.

---

## Classical baselines (CPU-only, run locally)

MSTL+ETS and SARIMAX are CPU-bound and don't benefit from GPU. Run them locally even when the TSFM session is on RunPod.

**MSTL+ETS:**
```powershell
python scripts/run_all.py --model mstl_ets `
    --variants nohijri hijri `
    --host local_cpu
```
Wall-clock: ~1.5h.

**SARIMAX (long-running, background):**
```powershell
# Option 1: foreground with nohup-like behavior on Windows
Start-Process powershell -ArgumentList "python scripts/run_all.py --model sarimax --variants nohijri hijri hijri_plusB --host local_cpu *> sarimax.log 2>&1" -WindowStyle Hidden

# Option 2: simpler, leave a terminal open overnight
python scripts/run_all.py --model sarimax `
    --variants nohijri hijri hijri_plusB `
    --host local_cpu
```
Wall-clock: ~18h. Kick off in parallel with the GPU work — they don't compete for resources.

---

## Shared: artifact verification

Regardless of path, after all TSFM runs are complete, verify:

```bash
# Every expected (model, variant, L) combination has a parquet
python scripts/verify_predictions.py

# Expected output (truncated):
# [OK] chronos_bolt_base    nohijri  L=96   seed=42   rows=10654
# [OK] chronos_bolt_base    nohijri  L=168  seed=42   rows=10582
# ...
# [OK] manifest entries: 50
# [OK] checkpoint SHAs recorded: 4
# [OK] all rows have y_true ≠ NaN and y_pred ≠ NaN
```

`scripts/verify_predictions.py` is part of the polish step; it cross-references the predictions on disk against the expected ablation matrix in `src/evaluation/expected_runs.py` and reports any missing or malformed runs.

---

## Shared: failure escalation
- **GPU disconnect / hang.** Kill the process, reduce batch size, retry. If it persists, restart the pod / reboot the laptop.
- **HuggingFace 401 / model not found.** Some checkpoints require accepting a license. Run `huggingface-cli login` and accept the terms on the model's HF page.
- **Numerical NaNs in TSFM output.** Usually a bf16 underflow with extreme covariate values. Re-run that specific pass in fp32 (`--no-bf16`) at the cost of ~2× wall-clock.
- **DM test reports zero significant pairs.** Likely insufficient effect size or holm correction crushing alpha. Sanity-check the per-regime MAE differences in T3 first; if they're physically small, that's a finding, not a bug.

---

## Cost sanity check
- Local path: $0 incremental.
- A100 path (RunPod, 8h on-demand A100 80GB SXM): ~$15.
- A100 path (RunPod, spot): ~$8 with a non-zero preemption risk.

Either way the spend ceiling for the entire benchmark is < $25.
