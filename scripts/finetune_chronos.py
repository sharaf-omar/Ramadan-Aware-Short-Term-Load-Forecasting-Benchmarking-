"""Fine-tune Chronos-Bolt-Base on Turkish electricity load.

The pretrained Amazon model already has strong zero-shot performance on
our test set (MAE 968.9 at L=720). This script continues training on
2018-2022 with 2023 as the validation set and writes a corrected test
parquet at `data/predictions/chronos_bolt_base__finetuned__nohijri__L<L>__seed<s>.parquet`.

Architecture notes:
  - Chronos-Bolt-Base is a T5 backbone with a 9-quantile regression
    head. The model's `forward(context, target=...)` returns a
    `ChronosBoltOutput` with `.loss` (built-in quantile loss) and
    `.quantile_preds` of shape (B, 9, 64). We use the 50th quantile
    (index 4) as the point forecast.
  - Test inference uses our existing per-tau day-ahead protocol via
    `src.models.tsfm._adapter.build_context_windows` to stay consistent
    with the bare Chronos parquet.

Wall-clock on RTX 4070 Laptop (8GB VRAM):
  - L=336, batch=16, 3 epochs: ~30-45 min
  - L=720, batch=8,  3 epochs: ~60-90 min

Usage:
    .venv/Scripts/python.exe scripts/finetune_chronos.py --context-length 336 --epochs 3
"""
from __future__ import annotations

import argparse
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from chronos import ChronosBoltPipeline

from src.evaluation.predictions_io import write_predictions


ROOT = Path(__file__).resolve().parents[1]
V2_CSV = ROOT / "data" / "processed" / "final_training_set_v2.csv"
HORIZON = 24
ISSUANCE_OFFSET = 24


class WindowedChronosDataset(Dataset):
    """Sliding-window (context, target) pairs for Chronos fine-tuning.

    Yields per-sample dicts:
        - context: (context_length,) float32 — y history
        - target:  (horizon,)        float32 — y future (24h)
    """

    def __init__(
        self,
        y_arr: np.ndarray,
        context_length: int,
        horizon: int = HORIZON,
        stride: int = 1,
    ):
        assert y_arr.dtype == np.float32, f"y_arr must be float32, got {y_arr.dtype}"
        assert y_arr.ndim == 1, f"y_arr must be 1D, got {y_arr.shape}"
        if len(y_arr) < context_length + horizon:
            raise ValueError(
                f"y_arr length {len(y_arr)} < L+h ({context_length}+{horizon})"
            )
        self.y = y_arr
        self.L = context_length
        self.H = horizon
        self.stride = stride
        n_full = len(y_arr) - self.L - self.H + 1
        self._starts = np.arange(0, n_full, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        s = int(self._starts[i])
        return {
            "context": torch.from_numpy(self.y[s : s + self.L]),
            "target":  torch.from_numpy(self.y[s + self.L : s + self.L + self.H]),
        }


def _compute_mae(eval_pred) -> dict[str, float]:
    """Compute MAE on the median (q=0.5, index 4 of 9) quantile.

    eval_pred.predictions: tuple where first element is quantile_preds
        of shape (N, 9, K). We take index 4 (median) and the first H steps.
    eval_pred.label_ids:   targets shape (N, H).
    """
    preds = eval_pred.predictions
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = np.asarray(preds)             # (N, 9, K)
    labels = np.asarray(eval_pred.label_ids)  # (N, H)
    H = labels.shape[1]
    median = preds[:, 4, :H]              # (N, H)
    mae = float(np.mean(np.abs(median - labels)))
    return {"mae": mae}


def fine_tune(
    context_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    train_y: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    stride: int = 1,
):
    """Fine-tune Chronos-Bolt-Base and return the fitted pipeline."""
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"  loading pretrained Chronos-Bolt-Base ...")
    pipeline = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )
    model = pipeline.model

    print(f"  building (context, target) windows: L={context_length}, H={HORIZON}, stride={stride} ...")
    train_ds = WindowedChronosDataset(train_y, context_length, HORIZON, stride=stride)
    val_ds   = WindowedChronosDataset(val_y,   context_length, HORIZON, stride=1)
    print(f"      train n={len(train_ds):,}   val n={len(val_ds):,}")

    with tempfile.TemporaryDirectory() as tmpdir:
        args = TrainingArguments(
            output_dir=str(Path(tmpdir) / "out"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=1e-2,
            warmup_steps=200,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="mae",
            greater_is_better=False,
            logging_steps=100,
            report_to="none",
            seed=seed,
            data_seed=seed,
            dataloader_num_workers=0,
            label_names=["target"],
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=_compute_mae,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()

    pipeline.model = trainer.model.eval()
    return pipeline


def predict_test(
    pipeline: ChronosBoltPipeline,
    df_full: pd.DataFrame,
    test_window: pd.DataFrame,
    context_length: int,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Run per-tau day-ahead inference using the fine-tuned pipeline.

    Mirrors the bare Chronos test protocol so the corrected parquet is
    directly comparable to data/predictions/chronos_bolt_base__nohijri__L*__seed0.parquet.
    """
    from src.models.tsfm._adapter import build_context_windows

    y_full = df_full["actual_load"]
    contexts = build_context_windows(y_full, test_window.index, context_length)
    valid_mask = ~np.isnan(contexts).any(axis=1)
    valid_idx = np.where(valid_mask)[0]
    valid_contexts = contexts[valid_mask]
    print(f"  inference: {len(valid_contexts):,} valid test windows")

    device = next(pipeline.model.parameters()).device
    out_dtype = next(pipeline.model.parameters()).dtype
    H = HORIZON
    y_pred = np.empty(len(valid_idx), dtype=np.float32)
    y_block = np.empty((len(valid_idx), H), dtype=np.float32)

    pipeline.model.eval()
    with torch.no_grad():
        for b_start in range(0, len(valid_contexts), batch_size):
            batch = valid_contexts[b_start : b_start + batch_size]
            ctx_t = torch.from_numpy(batch).to(device=device, dtype=out_dtype)
            output = pipeline.model(context=ctx_t)
            qp = output.quantile_preds.float().cpu().numpy()  # (B, 9, K)
            median = qp[:, 4, :H]  # (B, H)
            y_block[b_start : b_start + len(batch)] = median
            y_pred[b_start : b_start + len(batch)] = median[:, -1]  # horizon-24

    out_idx = test_window.index[valid_idx]
    out = pd.DataFrame({
        "y_true": test_window["actual_load"].values[valid_idx],
        "y_pred": y_pred,
        "regime": test_window["regime"].values[valid_idx],
        "y_block": [list(map(float, row)) for row in y_block],
    }, index=out_idx)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, default=336,
                        choices=[96, 168, 336, 720])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--stride", type=int, default=4,
                        help="Take every Nth training window (1 = all windows). "
                             "Larger = faster training, smaller effective dataset.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"[1/5] Loading v2 dataset ...")
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = (df.index.tz_convert("UTC")
                if df.index.tz is not None else df.index.tz_localize("UTC"))

    train = df.loc["2018":"2022"]
    val   = df.loc["2023"]
    test  = df.loc["2024-01-01":"2025-03-31"]
    print(f"      train={len(train):,}  val={len(val):,}  test={len(test):,}")

    train_y = train["actual_load"].to_numpy(dtype=np.float32, copy=True)
    val_y   = val["actual_load"].to_numpy(dtype=np.float32, copy=True)

    print(f"[2/5] Fine-tuning Chronos-Bolt-Base (L={args.context_length}, "
          f"epochs={args.epochs}, batch={args.batch_size}, lr={args.learning_rate}, "
          f"stride={args.stride}) ...")
    t0 = time.time()
    pipeline = fine_tune(
        context_length=args.context_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_y=train_y,
        val_y=val_y,
        seed=args.seed,
        stride=args.stride,
    )
    print(f"      fine-tune done in {time.time()-t0:.1f}s")

    print(f"[3/5] Running test inference (per-tau day-ahead) ...")
    t0 = time.time()
    preds = predict_test(pipeline, df, test, context_length=args.context_length)
    print(f"      inference done in {time.time()-t0:.1f}s  ({len(preds):,} rows)")

    print(f"[4/5] Writing parquet ...")
    path = write_predictions(
        preds,
        model="chronos_bolt_base__finetuned",
        variant="nohijri",
        context_length=args.context_length,
        seed=args.seed,
    )
    print(f"      -> {path}")

    print(f"[5/5] Quick MAE summary:")
    print(f"      agg     MAE = {(preds.y_true - preds.y_pred).abs().mean():.1f}")
    for r in ("Normal", "Ramadan", "Heatwave"):
        sub = preds[preds.regime == r]
        if len(sub):
            print(f"      {r:9s} MAE = {(sub.y_true - sub.y_pred).abs().mean():.1f}")


if __name__ == "__main__":
    main()
