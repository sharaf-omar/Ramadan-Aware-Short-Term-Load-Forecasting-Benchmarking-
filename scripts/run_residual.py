"""Fit and apply a LightGBM residual head to a TSFM's test predictions.

For each of three chronological folds within the test window, the
residual model is trained on the other two folds and predicts on the
held-out fold. The three out-of-fold predictions are concatenated to
produce a corrected forecast for every test tau.

Usage:
    .venv/Scripts/python.exe scripts/run_residual.py \\
        --tsfm-parquet chronos_bolt_base__nohijri__L720__seed0.parquet \\
        --tsfm-name chronos_bolt_base --context-length 720 \\
        --variant hijri --seed 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.predictions_io import predictions_path
from src.models.residual import LGBMResidualModel


ROOT = Path(__file__).resolve().parents[1]
V2_CSV = ROOT / "data" / "processed" / "final_training_set_v2.csv"
PRED_DIR = ROOT / "data" / "predictions"

# Three chronological splits of the 2024-01-01..2025-03-31 test window.
FOLD_BOUNDARIES = [
    pd.Timestamp("2024-01-01", tz="UTC"),
    pd.Timestamp("2024-06-01", tz="UTC"),
    pd.Timestamp("2024-11-01", tz="UTC"),
    pd.Timestamp("2025-04-01", tz="UTC"),  # exclusive upper
]


def _load_v2() -> pd.DataFrame:
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = (df.index.tz_convert("UTC")
                if df.index.tz is not None else df.index.tz_localize("UTC"))
    df = df.dropna(subset=["y_lag_336h", "y_roll168_mean"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsfm-parquet", required=True,
                        help="Filename in data/predictions/ (the bare TSFM test parquet).")
    parser.add_argument("--tsfm-name", required=True,
                        help="Output filename prefix, e.g. chronos_bolt_base.")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--variant", required=True, choices=["nohijri", "hijri"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("[1/5] Loading TSFM test predictions ...")
    tsfm_path = PRED_DIR / args.tsfm_parquet
    if not tsfm_path.exists():
        raise FileNotFoundError(f"Missing {tsfm_path}")
    tsfm_df = pd.read_parquet(tsfm_path)
    if tsfm_df.index.tz is None:
        tsfm_df.index = tsfm_df.index.tz_localize("UTC")
    print(f"      {len(tsfm_df):,} rows, span {tsfm_df.index.min()} .. {tsfm_df.index.max()}")

    print("[2/5] Loading v2 features ...")
    v2 = _load_v2()
    print(f"      {len(v2):,} v2 rows")

    print(f"[3/5] Building 3 chronological folds ...")
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for i in range(3):
        lo, hi = FOLD_BOUNDARIES[i], FOLD_BOUNDARIES[i + 1]
        fold_mask = (tsfm_df.index >= lo) & (tsfm_df.index < hi)
        fold_idx = tsfm_df.index[fold_mask]
        other_idx = tsfm_df.index[~fold_mask]
        print(f"      fold {i+1}: held-out n={len(fold_idx):,}  train n={len(other_idx):,}")
        folds.append((fold_idx, other_idx))

    print(f"[4/5] Fitting + applying residual heads per fold (variant={args.variant}) ...")
    corrected_pieces: list[pd.DataFrame] = []
    t0 = time.time()
    for i, (fold_idx, other_idx) in enumerate(folds):
        # Use 10% of `other` as val for early stopping (last 10% chronologically).
        cutoff = int(len(other_idx) * 0.9)
        train_idx = other_idx[:cutoff]
        val_idx = other_idx[cutoff:]
        train_tsfm = tsfm_df.loc[train_idx]
        val_tsfm = tsfm_df.loc[val_idx]
        held_tsfm = tsfm_df.loc[fold_idx]

        m = LGBMResidualModel(variant=args.variant)
        m.fit_residual(train_tsfm, v2, val_tsfm, v2, seed=args.seed)
        corrected = m.correct(held_tsfm, v2)
        corrected_pieces.append(corrected)
        print(f"      fold {i+1} done")

    out = pd.concat(corrected_pieces).sort_index()
    print(f"      total {len(out):,} corrected rows in {time.time()-t0:.1f}s")

    print("[5/5] Writing parquet ...")
    out_path = predictions_path(
        model=f"{args.tsfm_name}__residual",
        variant=args.variant,
        context_length=args.context_length,
        seed=args.seed,
    )
    out.to_parquet(out_path)
    print(f"      -> {out_path}")


if __name__ == "__main__":
    main()
