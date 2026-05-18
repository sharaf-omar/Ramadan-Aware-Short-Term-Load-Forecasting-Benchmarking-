"""Tier 2 quick wins on top of Tier 1.

Three families of new prediction parquets:

  (A) Per-regime separate residual heads on the 4 TSFMs:
      one head trained on Normal-only rows, another on Ramadan-only,
      Heatwave routed to bare. Compare vs the current Tier-1 single
      Normal+Ramadan head.

  (B) Apply the regime-stratified residual head to LightGBM-hijri and
      LightGBM-nohijri. LGBM was trained on these same features, so a
      gain here would be a "residual rescues even the incumbent" finding.

  (C) Apply the regime-stratified residual head to MSTL+ETS hijri and
      SARIMAX hijri. Classical baselines are weak and have lots of
      learnable structure remaining.

Outputs new parquets under canonical names with a `_per_regime` or
`_lgbm_res` / `_classical_res` suffix where helpful.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tune_residual import (
    FEATURES, FOLD_BOUNDARIES, enrich_features, load_v2,
)
from src.evaluation.predictions_io import predictions_path


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


def fit_residual_cv_per_regime(
    tsfm_df: pd.DataFrame, feat_df: pd.DataFrame, seed: int = 0,
) -> pd.DataFrame:
    """Two separate residual heads, one per regime.

    Trains a Normal-only head on Normal rows in train-folds, an
    Ramadan-only head on Ramadan rows. Heatwave routes to bare.
    """
    feat = enrich_features(feat_df)
    common = tsfm_df.index.intersection(feat.index)
    tsfm_df = tsfm_df.loc[common]
    feat = feat.loc[common]
    out_pieces = []
    for i in range(3):
        lo, hi = FOLD_BOUNDARIES[i], FOLD_BOUNDARIES[i + 1]
        fold_mask = (tsfm_df.index >= lo) & (tsfm_df.index < hi)
        fold_idx = tsfm_df.index[fold_mask]
        other_idx = tsfm_df.index[~fold_mask]

        boosters: dict[str, lgb.Booster] = {}
        for regime in ("Normal", "Ramadan"):
            keep = tsfm_df.loc[other_idx, "regime"] == regime
            r_idx = other_idx[keep.values]
            if len(r_idx) < 50:
                continue
            cutoff = int(len(r_idx) * 0.9)
            train_idx = r_idx[:cutoff]
            val_idx = r_idx[cutoff:]
            X_tr = feat.loc[train_idx, FEATURES]
            y_tr = (tsfm_df.loc[train_idx, "y_true"] - tsfm_df.loc[train_idx, "y_pred"]).values
            X_va = feat.loc[val_idx, FEATURES]
            y_va = (tsfm_df.loc[val_idx, "y_true"] - tsfm_df.loc[val_idx, "y_pred"]).values
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
            params = {
                "objective": "regression_l1", "metric": "mae",
                "learning_rate": 0.05, "num_leaves": 127, "max_depth": -1,
                "min_data_in_leaf": 20, "feature_fraction": 0.9,
                "bagging_fraction": 0.9, "bagging_freq": 5,
                "seed": seed, "verbose": -1,
            }
            boosters[regime] = lgb.train(
                params, dtrain, num_boost_round=2000,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

        held = tsfm_df.loc[fold_idx].copy()
        new_pred = held["y_pred"].values.copy()
        for regime, booster in boosters.items():
            mask = (held["regime"] == regime).values
            if not mask.any():
                continue
            X = feat.loc[fold_idx[mask], FEATURES]
            new_pred[mask] = held.loc[fold_idx[mask], "y_pred"].values + booster.predict(X)
        held["y_pred"] = new_pred
        out_pieces.append(held)
    return pd.concat(out_pieces).sort_index()


def fit_residual_cv_for_existing(
    base_df: pd.DataFrame, feat_df: pd.DataFrame, seed: int = 0,
    only_regimes: list[str] | None = None,
) -> pd.DataFrame:
    """Generic regime-stratified residual head — works for any base parquet
    (LGBM, classical, etc.), not just TSFM. Same recipe as Tier 1."""
    feat = enrich_features(feat_df)
    common = base_df.index.intersection(feat.index)
    base_df = base_df.loc[common]
    feat = feat.loc[common]
    out_pieces = []
    for i in range(3):
        lo, hi = FOLD_BOUNDARIES[i], FOLD_BOUNDARIES[i + 1]
        fold_mask = (base_df.index >= lo) & (base_df.index < hi)
        fold_idx = base_df.index[fold_mask]
        other_idx = base_df.index[~fold_mask]
        cutoff = int(len(other_idx) * 0.9)
        train_idx = other_idx[:cutoff]
        val_idx = other_idx[cutoff:]
        if only_regimes is not None:
            train_keep = base_df.loc[train_idx, "regime"].isin(only_regimes)
            val_keep   = base_df.loc[val_idx,   "regime"].isin(only_regimes)
            train_idx = train_idx[train_keep.values]
            val_idx   = val_idx[val_keep.values]
        X_tr = feat.loc[train_idx, FEATURES]
        y_tr = (base_df.loc[train_idx, "y_true"] - base_df.loc[train_idx, "y_pred"]).values
        X_va = feat.loc[val_idx, FEATURES]
        y_va = (base_df.loc[val_idx, "y_true"] - base_df.loc[val_idx, "y_pred"]).values
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        params = {
            "objective": "regression_l1", "metric": "mae",
            "learning_rate": 0.05, "num_leaves": 127, "max_depth": -1,
            "min_data_in_leaf": 20, "feature_fraction": 0.9,
            "bagging_fraction": 0.9, "bagging_freq": 5,
            "seed": seed, "verbose": -1,
        }
        booster = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        held = base_df.loc[fold_idx].copy()
        X = feat.loc[fold_idx, FEATURES]
        residual_hat = booster.predict(X)
        new_pred = held["y_pred"].values.copy()
        if only_regimes is not None:
            apply_mask = held["regime"].isin(only_regimes).values
            new_pred[apply_mask] += residual_hat[apply_mask]
        else:
            new_pred += residual_hat
        held["y_pred"] = new_pred
        out_pieces.append(held)
    return pd.concat(out_pieces).sort_index()


def summary(label: str, df: pd.DataFrame) -> str:
    return (
        f"  {label:45s} agg={mae(df):7.1f}  "
        f"N={mae(df,'Normal'):7.1f}  R={mae(df,'Ramadan'):7.1f}  "
        f"H={mae(df,'Heatwave'):7.1f}"
    )


def main() -> None:
    v2 = load_v2()
    print(f"v2 rows: {len(v2):,}")

    # ===== (A) Per-regime separate residual heads on 4 TSFMs =====
    print("\n=== (A) Per-regime separate residual heads on TSFMs ===")
    for tsfm, L in [("chronos_bolt_base", 720), ("moirai_1_1_small", 336),
                    ("timesfm_2_5", 168), ("time_moe_200m", 720)]:
        bare = _load(f"{tsfm}__nohijri__L{L}__seed0.parquet")
        existing = _load(f"{tsfm}__residual__hijri__L{L}__seed0.parquet")
        print(summary(f"{tsfm}-L{L} bare", bare))
        print(summary(f"{tsfm}-L{L} +residual-h (Tier1)", existing))
        t0 = time.time()
        per_reg = fit_residual_cv_per_regime(bare, v2, seed=0)
        out_path = PRED_DIR / f"{tsfm}__residual_per_regime__hijri__L{L}__seed0.parquet"
        per_reg.to_parquet(out_path)
        print(summary(f"{tsfm}-L{L} +residual-h-per-regime", per_reg))
        print(f"      ({time.time()-t0:.1f}s) -> {out_path.name}")

    # ===== (B) Apply residual head to LightGBM =====
    print("\n=== (B) Residual head on top of LightGBM ===")
    for variant_lgbm in ("nohijri", "hijri"):
        bare = _load(f"lgbm__{variant_lgbm}__seed44.parquet")
        print(summary(f"lgbm-{variant_lgbm} bare", bare))
        t0 = time.time()
        out = fit_residual_cv_for_existing(bare, v2, seed=0, only_regimes=["Normal", "Ramadan"])
        out_path = PRED_DIR / f"lgbm__{variant_lgbm}__residual_h__seed44.parquet"
        out.to_parquet(out_path)
        print(summary(f"lgbm-{variant_lgbm} + residual-h", out))
        print(f"      ({time.time()-t0:.1f}s) -> {out_path.name}")

    # ===== (C) Apply residual head to classical baselines =====
    print("\n=== (C) Residual head on classical baselines ===")
    for classical_name, fname in [
        ("mstl_ets-hijri", "mstl_ets__hijri__seed0.parquet"),
        ("sarimax-hijri", "sarimax__hijri__seed0.parquet"),
    ]:
        bare = _load(fname)
        print(summary(f"{classical_name} bare", bare))
        t0 = time.time()
        out = fit_residual_cv_for_existing(bare, v2, seed=0, only_regimes=["Normal", "Ramadan"])
        prefix = fname.split("__")[0]
        variant = fname.split("__")[1]
        out_path = PRED_DIR / f"{prefix}__{variant}__residual_h__seed0.parquet"
        out.to_parquet(out_path)
        print(summary(f"{classical_name} + residual-h", out))
        print(f"      ({time.time()-t0:.1f}s) -> {out_path.name}")


if __name__ == "__main__":
    main()
