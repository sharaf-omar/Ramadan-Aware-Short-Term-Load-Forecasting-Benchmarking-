"""Run PatchTSMixer on the v2 test set and save predictions to parquet.

Usage:
    .venv/Scripts/python.exe scripts/run_patchtsmixer.py --variant nohijri --context-length 336 --seed 42
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.evaluation.predictions_io import write_predictions
from src.models.dl import PatchTSMixerModel


ROOT = Path(__file__).resolve().parents[1]
V2_CSV = ROOT / "data" / "processed" / "final_training_set_v2.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["nohijri", "hijri", "hijri_plusB"])
    parser.add_argument("--context-length", type=int, required=True,
                        choices=[96, 168, 336, 720])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"[1/5] Loading v2 dataset ...")
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = (df.index.tz_convert("UTC")
                if df.index.tz is not None else df.index.tz_localize("UTC"))
    df = df.dropna(subset=["y_lag_336h", "y_roll168_mean"])

    train = df.loc["2018":"2022"]
    val   = df.loc["2023"]
    test  = df.loc["2024-01-01":"2025-03-31"]
    print(f"      train={len(train):,}  val={len(val):,}  test={len(test):,}")

    print(f"[2/5] Instantiating patchtsmixer variant={args.variant} L={args.context_length} ...")
    model = PatchTSMixerModel(
        variant=args.variant,
        context_length=args.context_length,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )

    print(f"[3/5] Fitting ...")
    t0 = time.time()
    model.fit(train, val, hijri=(args.variant != "nohijri"), seed=args.seed)
    print(f"      fit done in {time.time()-t0:.1f}s")

    print(f"[4/5] Forecasting ...")
    t0 = time.time()
    preds = model.predict(test)
    print(f"      forecast done in {time.time()-t0:.1f}s  ({len(preds):,} rows)")

    print(f"[5/5] Writing parquet ...")
    path = write_predictions(
        preds, model=model.name, variant=args.variant,
        context_length=args.context_length, seed=args.seed,
    )
    print(f"      -> {path}")


if __name__ == "__main__":
    main()
