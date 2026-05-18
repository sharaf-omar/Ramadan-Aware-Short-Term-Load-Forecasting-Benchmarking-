"""Run a TSFM on the v2 test set and save predictions to parquet.

Usage:
    # Single L
    python scripts/run_tsfm.py --model chronos --context-length 336

    # Sweep (one model loaded once, multiple L values):
    python scripts/run_tsfm.py --model chronos \
        --context-length 96 --context-length 168 --context-length 336 --context-length 720

    # Hijri-covariate variant (only on covariate-capable models):
    python scripts/run_tsfm.py --model timesfm --context-length 336 --variant hijri
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.evaluation.predictions_io import write_predictions


ROOT = Path(__file__).resolve().parents[1]
V2_CSV = ROOT / "data" / "processed" / "final_training_set_v2.csv"

HIJRI_COVARIATE_COLS = ["is_ramadan", "day_of_ramadan", "is_eid", "temp_c"]


MODEL_REGISTRY = {
    "chronos": ("src.models.tsfm.chronos_bolt", "ChronosBoltModel"),
    "timesfm": ("src.models.tsfm.timesfm", "TimesFMModel"),
    "moirai":  ("src.models.tsfm.moirai",  "MoiraiModel"),
    "timemoe": ("src.models.tsfm.time_moe", "TimeMoEModel"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument(
        "--context-length", type=int, action="append", required=True,
        help="Pass multiple times for a sweep.",
    )
    parser.add_argument("--variant", default="nohijri", choices=["nohijri", "hijri"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"[1/4] Loading v2 dataset ...")
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")

    test_window = df.loc["2024-01-01":"2025-03-31"]
    print(f"      test forecast hours: {len(test_window):,}")

    print(f"[2/4] Instantiating {args.model} ...")
    import importlib
    mod_path, cls_name = MODEL_REGISTRY[args.model]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    model = cls()
    print(f"      name={model.name} supports_dynamic_covariates={model.supports_dynamic_covariates}")

    if args.variant == "hijri" and not model.supports_dynamic_covariates:
        raise ValueError(
            f"{model.name} does not support dynamic covariates; use --variant nohijri."
        )

    for L in args.context_length:
        print(f"[3/4] Forecasting (L={L}, variant={args.variant}) ...")
        t0 = time.time()
        if args.variant == "hijri":
            preds_all = model.predict_with_covariates(
                df, context_length=L, covariate_cols=HIJRI_COVARIATE_COLS,
            )
        else:
            preds_all = model.predict(df, context_length=L)
        test_preds = preds_all.loc[test_window.index.intersection(preds_all.index)]
        elapsed = time.time() - t0
        print(f"      L={L} done in {elapsed:.1f}s  ({len(test_preds):,} test predictions; {len(preds_all):,} total)")

        print(f"[4/4] Writing parquet for L={L} ...")
        path = write_predictions(
            test_preds,
            model=model.name,
            variant=args.variant,
            context_length=L,
            seed=args.seed,
        )
        print(f"      -> {path}")


if __name__ == "__main__":
    main()
