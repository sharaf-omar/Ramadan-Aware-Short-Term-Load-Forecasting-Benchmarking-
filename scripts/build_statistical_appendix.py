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


from src.evaluation.bootstrap import block_bootstrap_ci


def _abs_err(df: pd.DataFrame, regime: str) -> np.ndarray:
    if regime == "aggregate":
        sub = df
    else:
        sub = df[df["regime"] == regime]
    if len(sub) == 0:
        return np.array([], dtype=float)
    return np.abs(sub["y_true"].values - sub["y_pred"].values).astype(float)


def compute_ci_table(
    preds: dict[str, pd.DataFrame],
    regimes: list[str] = REGIMES,
    n_resamples: int = 1000,
    block_size: int = 24,
    seed: int = 0,
) -> pd.DataFrame:
    """For each (model, regime) compute MAE + 95% block-bootstrap CI.

    Returns long-format DataFrame: model, regime, mae, ci_lo, ci_hi.
    Empty-regime rows have NaN for all three numeric columns.
    """
    rows = []
    for model_name in preds:
        df = preds[model_name]
        for regime in regimes:
            err = _abs_err(df, regime)
            if len(err) == 0:
                rows.append({
                    "model": model_name, "regime": regime,
                    "mae": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                })
                continue
            mae = float(err.mean())
            ci_lo, ci_hi = block_bootstrap_ci(
                err, block_size=block_size,
                n_resamples=n_resamples, alpha=0.05, seed=seed,
                statistic=np.mean,
            )
            rows.append({
                "model": model_name, "regime": regime,
                "mae": mae, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            })
    return pd.DataFrame(rows, columns=["model", "regime", "mae", "ci_lo", "ci_hi"])


from src.evaluation.dm_test import dm_test, holm_bonferroni


def _regime_mask(df: pd.DataFrame, regime: str) -> np.ndarray:
    if regime == "aggregate":
        return np.ones(len(df), dtype=bool)
    return (df["regime"] == regime).values


def compute_dm_matrix(
    preds: dict[str, pd.DataFrame],
    regime: str,
) -> pd.DataFrame:
    """Pairwise DM tests over the lower triangle of the model list.

    Returns long format: model_i, model_j, dm_stat, p_raw, p_holm.
    DM convention from src.evaluation.dm_test: dm_stat > 0 means model_j
    has lower loss. Holm-Bonferroni applied within this single call's
    family of comparisons.
    """
    names = list(preds.keys())
    pairs = []
    raw_p = []
    stats = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_i, name_j = names[i], names[j]
            df_i = preds[name_i]
            df_j = preds[name_j]
            mask = _regime_mask(df_i, regime)
            n = int(mask.sum())
            if n < 5:
                pairs.append((name_i, name_j))
                raw_p.append(np.nan)
                stats.append(np.nan)
                continue
            y_true = df_i["y_true"].values[mask]
            y_pred_a = df_i["y_pred"].values[mask]
            y_pred_b = df_j["y_pred"].values[mask]
            stat, p = dm_test(y_true, y_pred_a, y_pred_b, h=24, loss="mae")
            pairs.append((name_i, name_j))
            raw_p.append(p)
            stats.append(stat)

    valid_idx = [k for k, p in enumerate(raw_p) if not np.isnan(p)]
    if valid_idx:
        valid_p = [raw_p[k] for k in valid_idx]
        adj = holm_bonferroni(valid_p)
        p_holm = [np.nan] * len(raw_p)
        for k, p_adj in zip(valid_idx, adj):
            p_holm[k] = p_adj
    else:
        p_holm = [np.nan] * len(raw_p)

    rows = [
        {"model_i": a, "model_j": b, "dm_stat": s, "p_raw": pr, "p_holm": ph}
        for (a, b), s, pr, ph in zip(pairs, stats, raw_p, p_holm)
    ]
    return pd.DataFrame(rows, columns=["model_i", "model_j", "dm_stat", "p_raw", "p_holm"])
