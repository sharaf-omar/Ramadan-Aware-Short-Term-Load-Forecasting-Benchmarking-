"""Build docs/statistical_appendix.md and CSVs from headline parquets.

See docs/superpowers/specs/2026-05-14-statistical-appendix-design.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "data" / "predictions"
OUT_DIR = ROOT / "data" / "statistical_appendix"
DOC_PATH = ROOT / "docs" / "statistical_appendix.md"


# (display_name, parquet_filename) — order matters; this is the row/column order
# in CI table and DM matrices.
MODELS: list[tuple[str, str]] = [
    ("lgbm-nohijri",              "lgbm__nohijri__seed44.parquet"),
    ("lgbm-hijri",                "lgbm__hijri__seed44.parquet"),
    ("chronos-bolt-L720",         "chronos_bolt_base__nohijri__L720__seed0.parquet"),
    ("timesfm-2.5-L168",          "timesfm_2_5__nohijri__L168__seed0.parquet"),
    ("moirai-1.1-small-L336",     "moirai_1_1_small__nohijri__L336__seed0.parquet"),
    ("time-moe-200m-L720",        "time_moe_200m__nohijri__L720__seed0.parquet"),
    ("mstl_ets-nohijri",          "mstl_ets__nohijri__seed0.parquet"),
    ("mstl_ets-hijri",            "mstl_ets__hijri__seed0.parquet"),
    ("sarimax-nohijri",           "sarimax__nohijri__seed0.parquet"),
    ("sarimax-hijri",             "sarimax__hijri__seed0.parquet"),
    ("patchtsmixer-nohijri-L168", "patchtsmixer__nohijri__L168__seed42.parquet"),
    ("patchtsmixer-hijri-L168",   "patchtsmixer__hijri__L168__seed42.parquet"),
]

REGIMES = ["aggregate", "Normal", "Ramadan", "Heatwave"]


def load_predictions(
    spec: list[tuple[str, str]] = MODELS,
) -> dict[str, pd.DataFrame]:
    """Load each parquet and intersect on the τ index.

    Returns dict[name -> DataFrame] where every DataFrame shares the
    exact same τ index (the intersection across all models).
    """
    dfs: dict[str, pd.DataFrame] = {}
    for name, fname in spec:
        path = PRED_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing parquet for {name}: {path}. "
                f"Re-run the upstream model script first."
            )
        dfs[name] = pd.read_parquet(path)

    common = None
    for df in dfs.values():
        idx = df.index
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) == 0:
        raise ValueError("No τ overlap across model parquets — aborting.")

    return {name: df.loc[common].sort_index() for name, df in dfs.items()}
