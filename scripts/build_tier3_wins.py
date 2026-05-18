"""Tier 3 quick wins — meta-router + stacked meta-learner.

Two new prediction parquets:

  1. meta_router__seed0.parquet
        Route per regime: Normal -> ensemble-top4-residual,
        Ramadan -> LGBM-hijri (best Ramadan single model),
        Heatwave -> Chronos-Bolt-Base L=720 (best Heatwave single model).
        Math predicts ~841 agg MAE vs current champion 872.

  2. stacked_lgbm__seed0.parquet
        LightGBM trained on (8 member predictions, regime one-hot,
        hour-sin/cos) -> y_true. 3-fold time-block CV. The LGBM picks
        optimal per-row weights, often beating simple median.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tune_residual import FOLD_BOUNDARIES


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
    return (f"  {label:40s} agg={mae(df):7.1f}  "
            f"N={mae(df,'Normal'):7.1f}  R={mae(df,'Ramadan'):7.1f}  "
            f"H={mae(df,'Heatwave'):7.1f}")


# ---------- 1. Meta-router ----------

def build_meta_router() -> pd.DataFrame:
    """Ensemble for Normal, LGBM-hijri for Ramadan, Chronos for Heatwave."""
    ensemble = _load("ensemble__top4_residual__seed0.parquet")
    lgbm = _load("lgbm__hijri__seed44.parquet")
    chronos = _load("chronos_bolt_base__nohijri__L720__seed0.parquet")

    common = ensemble.index.intersection(lgbm.index).intersection(chronos.index)
    ensemble = ensemble.loc[common]
    lgbm = lgbm.loc[common]
    chronos = chronos.loc[common]

    out = ensemble.copy()
    regime = out["regime"].values
    new_pred = ensemble["y_pred"].values.copy()
    ramadan_mask = regime == "Ramadan"
    heatwave_mask = regime == "Heatwave"
    new_pred[ramadan_mask] = lgbm["y_pred"].values[ramadan_mask]
    new_pred[heatwave_mask] = chronos["y_pred"].values[heatwave_mask]
    out["y_pred"] = new_pred
    if "y_block" in out.columns:
        out = out.drop(columns="y_block")
    return out


# ---------- 2. Stacked LightGBM meta-learner ----------

MEMBERS = {
    "chronos_res": "chronos_bolt_base__residual__hijri__L720__seed0.parquet",
    "lgbm_h_res":  "lgbm__hijri__residual_h__seed44.parquet",
    "lgbm_nh_res": "lgbm__nohijri__residual_h__seed44.parquet",
    "timemoe_res": "time_moe_200m__residual__hijri__L720__seed0.parquet",
    "moirai_res":  "moirai_1_1_small__residual__hijri__L336__seed0.parquet",
    "timesfm_res": "timesfm_2_5__residual__hijri__L168__seed0.parquet",
    "chronos_bare": "chronos_bolt_base__nohijri__L720__seed0.parquet",
    "lgbm_h_bare":  "lgbm__hijri__seed44.parquet",
}


def build_stacked_lgbm(seed: int = 0) -> pd.DataFrame:
    """LightGBM trained on (member predictions + hour features).

    Regime-stratified routing: train on Normal+Ramadan only; route
    Heatwave τ to Chronos-Bolt-Base bare (the best Heatwave single
    model). Without this, the meta-learner blows up Heatwave.
    """
    dfs = {name: _load(fn) for name, fn in MEMBERS.items()}
    common = None
    for d in dfs.values():
        common = d.index if common is None else common.intersection(d.index)
    for name in dfs:
        dfs[name] = dfs[name].loc[common]

    ref = next(iter(dfs.values()))
    chronos_bare = dfs["chronos_bare"]  # for Heatwave routing

    X = pd.DataFrame({
        name: dfs[name]["y_pred"].values for name in MEMBERS
    }, index=common)
    X["hour_sin"] = np.sin(2 * np.pi * common.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * common.hour / 24)
    X["dow_sin"]  = np.sin(2 * np.pi * common.dayofweek / 7)
    X["dow_cos"]  = np.cos(2 * np.pi * common.dayofweek / 7)
    X["is_ramadan_regime"] = (ref.regime == "Ramadan").astype(int).values
    y = ref["y_true"].values
    regime_arr = ref["regime"].values

    out_pieces = []
    for i in range(3):
        lo, hi = FOLD_BOUNDARIES[i], FOLD_BOUNDARIES[i + 1]
        fold_mask = (common >= lo) & (common < hi)
        fold_idx = common[fold_mask]
        other_idx = common[~fold_mask]

        # Restrict training to Normal+Ramadan rows.
        other_regimes = ref.loc[other_idx, "regime"].values
        other_keep = (other_regimes == "Normal") | (other_regimes == "Ramadan")
        other_idx = other_idx[other_keep]
        cutoff = int(len(other_idx) * 0.9)
        train_idx = other_idx[:cutoff]
        val_idx = other_idx[cutoff:]
        X_tr = X.loc[train_idx]
        y_tr = pd.Series(y, index=common).loc[train_idx].values
        X_va = X.loc[val_idx]
        y_va = pd.Series(y, index=common).loc[val_idx].values
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        params = {
            "objective": "regression_l1", "metric": "mae",
            "learning_rate": 0.03, "num_leaves": 63, "max_depth": -1,
            "min_data_in_leaf": 30, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "seed": seed, "verbose": -1,
        }
        booster = lgb.train(
            params, dtrain, num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(60, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        # Stacked predictions for held-out fold
        stacked_held = booster.predict(X.loc[fold_idx])
        held_regimes = ref.loc[fold_idx, "regime"].values
        held_y_true = pd.Series(y, index=common).loc[fold_idx].values

        # Route Heatwave to Chronos-bare; keep Normal+Ramadan as stacked
        new_pred = stacked_held.copy()
        heat_mask = held_regimes == "Heatwave"
        new_pred[heat_mask] = chronos_bare.loc[fold_idx[heat_mask], "y_pred"].values

        piece = pd.DataFrame({
            "y_true": held_y_true,
            "y_pred": new_pred,
            "regime": held_regimes,
        }, index=fold_idx)
        out_pieces.append(piece)

    return pd.concat(out_pieces).sort_index()


def main() -> None:
    print("[1/2] Meta-router (ensemble Normal / LGBM Ramadan / Chronos Heatwave) ...")
    router = build_meta_router()
    out_path = PRED_DIR / "meta_router__seed0.parquet"
    router.to_parquet(out_path)
    print(summary("meta-router", router))
    print(f"      -> {out_path.name}")

    print("\n[2/2] Stacked LightGBM meta-learner (8 members + regime + hour) ...")
    t0 = time.time()
    stacked = build_stacked_lgbm(seed=0)
    out_path = PRED_DIR / "stacked_lgbm__seed0.parquet"
    stacked.to_parquet(out_path)
    print(summary("stacked-lgbm", stacked))
    print(f"      ({time.time()-t0:.1f}s) -> {out_path.name}")

    print("\n=== Comparison vs prior champions ===")
    for fname, label in [
        ("ensemble__top4_residual__seed0.parquet", "ensemble-top4-residual"),
        ("ensemble__top4__seed0.parquet",         "ensemble-top4-mixed"),
        ("routed__best_per_regime__seed0.parquet","routed-best-per-regime"),
        ("meta_router__seed0.parquet",            "meta-router (NEW)"),
        ("stacked_lgbm__seed0.parquet",           "stacked-lgbm (NEW)"),
    ]:
        d = _load(fname)
        print(summary(label, d))


if __name__ == "__main__":
    main()
