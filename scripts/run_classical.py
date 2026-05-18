"""Run a classical model on the v2 test set and save predictions to parquet.

Usage:
    .venv/Scripts/python.exe scripts/run_classical.py --model mstl_ets --variant nohijri
    .venv/Scripts/python.exe scripts/run_classical.py --model mstl_ets --variant hijri
    .venv/Scripts/python.exe scripts/run_classical.py --model sarimax --variant nohijri
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.evaluation.predictions_io import write_predictions


ROOT = Path(__file__).resolve().parents[1]
V2_CSV = ROOT / "data" / "processed" / "final_training_set_v2.csv"


MODEL_REGISTRY = {
    "mstl_ets": ("src.models.classical.mstl_ets", "MSTLETSModel"),
    "sarimax":  ("src.models.classical.sarimax",  "SARIMAXModel"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--variant", required=True,
                        choices=["nohijri", "hijri", "hijri_plusB"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"[1/5] Loading v2 dataset ...")
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")

    df = df.dropna(subset=["y_lag_336h", "y_roll168_mean"])

    train = df.loc["2018":"2022"]
    val = df.loc["2023"]
    test = df.loc["2024-01-01":"2025-03-31"]
    print(f"      train={len(train):,}  val={len(val):,}  test={len(test):,}")

    print(f"[2/5] Instantiating {args.model} variant={args.variant} ...")
    import importlib
    mod_path, cls_name = MODEL_REGISTRY[args.model]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    model = cls(variant=args.variant)
    print(f"      name={model.name}")

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
        context_length=None, seed=args.seed,
    )
    print(f"      -> {path}")


if __name__ == "__main__":
    main()
