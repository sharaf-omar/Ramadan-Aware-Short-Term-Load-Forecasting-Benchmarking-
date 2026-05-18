"""Tier 1 quick wins on top of the existing parquets.

Produces three new prediction parquets:

  1. ensemble__top4__seed0.parquet
        Median of (Chronos-L720, LGBM-hijri-seed44, Time-MoE-L720,
        Moirai+residual-hijri-L336) — the four strongest single
        models in the headline cohort.

  2. patchtsmixer__residual__hijri__L168__seed42.parquet
        Apply the same regime-stratified-routing residual head to
        PatchTSMixer that Plan 6 used on the four TSFMs.

  3. routed__best_per_regime__seed0.parquet
        Per-regime model routing: Heatwave -> Chronos-L720, Ramadan ->
        LGBM-hijri, Normal -> (Chronos+residual hijri). Operationalizes
        the deployment story.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tune_residual import fit_residual_cv, load_v2


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "data" / "predictions"


def _load(fname: str) -> pd.DataFrame:
    df = pd.read_parquet(PRED_DIR / fname)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def mae(df: pd.DataFrame, regime: str | None = None) -> float:
    s = df if regime is None else df[df.regime == regime]
    return (s.y_true - s.y_pred).abs().mean() if len(s) else float("nan")


def summary(label: str, df: pd.DataFrame) -> str:
    return (
        f"  {label:40s} agg={mae(df):7.1f}  "
        f"Normal={mae(df, 'Normal'):7.1f}  "
        f"Ramadan={mae(df, 'Ramadan'):7.1f}  "
        f"Heatwave={mae(df, 'Heatwave'):7.1f}"
    )


# ---------- 1. Ensemble (median of 4) ----------

def build_ensemble() -> pd.DataFrame:
    """Median across the 4 strongest models — intersection-on-tau."""
    members = {
        "chronos":  _load("chronos_bolt_base__nohijri__L720__seed0.parquet"),
        "lgbm":     _load("lgbm__hijri__seed44.parquet"),
        "timemoe":  _load("time_moe_200m__nohijri__L720__seed0.parquet"),
        "moirai+r": _load("moirai_1_1_small__residual__hijri__L336__seed0.parquet"),
    }
    # Intersection on tau
    common = None
    for d in members.values():
        idx = d.index
        common = idx if common is None else common.intersection(idx)

    preds_matrix = np.stack(
        [members[k].loc[common, "y_pred"].values for k in members],
        axis=1,
    )  # shape (n, 4)
    ensemble_pred = np.median(preds_matrix, axis=1)
    ref = next(iter(members.values())).loc[common]
    out = pd.DataFrame({
        "y_true": ref["y_true"].values,
        "y_pred": ensemble_pred,
        "regime": ref["regime"].values,
    }, index=common)
    return out


# ---------- 2. PatchTSMixer + residual (reusing tune_residual.fit_residual_cv) ----------

def build_patchtsmixer_residual() -> pd.DataFrame:
    pts = _load("patchtsmixer__hijri__L168__seed42.parquet")
    v2 = load_v2()
    return fit_residual_cv(pts, v2, seed=0, only_regimes=["Normal", "Ramadan"])


# ---------- 3. Regime-routed best-of-models ----------

def build_regime_routed() -> pd.DataFrame:
    """Route each tau to the per-regime best model:
       Heatwave -> Chronos-L720 (1221)
       Ramadan  -> LGBM-hijri-seed44 (800)
       Normal   -> Chronos+residual-hijri (878)
    """
    chronos = _load("chronos_bolt_base__nohijri__L720__seed0.parquet")
    lgbm = _load("lgbm__hijri__seed44.parquet")
    chronos_res = _load("chronos_bolt_base__residual__hijri__L720__seed0.parquet")

    # Intersection on tau
    common = chronos.index.intersection(lgbm.index).intersection(chronos_res.index)
    chronos = chronos.loc[common]
    lgbm = lgbm.loc[common]
    chronos_res = chronos_res.loc[common]

    out = chronos.copy()  # use Chronos's regime + y_true as reference
    regime = out["regime"].values
    new_pred = chronos["y_pred"].values.copy()
    # Replace Ramadan rows with LGBM
    ramadan_mask = regime == "Ramadan"
    new_pred[ramadan_mask] = lgbm["y_pred"].values[ramadan_mask]
    # Replace Normal rows with Chronos+residual
    normal_mask = regime == "Normal"
    new_pred[normal_mask] = chronos_res["y_pred"].values[normal_mask]
    # Heatwave stays as Chronos (already in new_pred)
    out["y_pred"] = new_pred
    # Drop y_block to avoid confusion (mixed-origin block doesn't make sense)
    if "y_block" in out.columns:
        out = out.drop(columns="y_block")
    return out


def main() -> None:
    print("[1/3] Building ensemble (median of 4) ...")
    ens = build_ensemble()
    out_path = PRED_DIR / "ensemble__top4__seed0.parquet"
    ens.to_parquet(out_path)
    print(summary("ensemble-median-top4", ens))
    print(f"      -> {out_path}")

    print("\n[2/3] Building PatchTSMixer + residual (tuned, regime-stratified) ...")
    pts_res = build_patchtsmixer_residual()
    out_path = PRED_DIR / "patchtsmixer__residual__hijri__L168__seed42.parquet"
    pts_res.to_parquet(out_path)
    bare_pts = _load("patchtsmixer__hijri__L168__seed42.parquet")
    print(summary("patchtsmixer-hijri-L168 (bare)", bare_pts))
    print(summary("patchtsmixer + residual-hijri", pts_res))
    print(f"      -> {out_path}")

    print("\n[3/3] Building regime-routed best-of-models ...")
    routed = build_regime_routed()
    out_path = PRED_DIR / "routed__best_per_regime__seed0.parquet"
    routed.to_parquet(out_path)
    print(summary("routed-best-per-regime", routed))
    print(f"      -> {out_path}")

    print("\n=== Final headline comparison (top entries) ===")
    print(summary("ensemble-median-top4", ens))
    print(summary("routed-best-per-regime", routed))
    print(summary("patchtsmixer + residual", pts_res))


if __name__ == "__main__":
    main()
