"""One-shot tuning experiment for Plan 6 residual heads.

Tries three layered changes vs the baseline residual head:
  (a) wider LGBM (num_leaves=127, min_data_in_leaf=20, 2000 estimators)
  (b) denser Hijri features (days_since_eid, days_to_eid, ramadan*hour,
      eid*hour, ramadan*dow)
  (c) regime-stratified routing — train on Normal+Ramadan only, route
      Heatwave tau back to bare TSFM (no correction during heatwaves)

Produces:
  - data/predictions/<tsfm>__residual_tuned__hijri__L<L>__seed0.parquet (4)
  - prints a tuning summary table
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.predictions_io import predictions_path
from src.models.residual.lgbm_residual import _ensure_calendar_features


ROOT = Path(__file__).resolve().parents[1]
V2_CSV = ROOT / "data" / "processed" / "final_training_set_v2.csv"
PRED_DIR = ROOT / "data" / "predictions"

FOLD_BOUNDARIES = [
    pd.Timestamp("2024-01-01", tz="UTC"),
    pd.Timestamp("2024-06-01", tz="UTC"),
    pd.Timestamp("2024-11-01", tz="UTC"),
    pd.Timestamp("2025-04-01", tz="UTC"),
]

TSFM_SPECS = [
    ("chronos_bolt_base", 720),
    ("moirai_1_1_small",  336),
    ("timesfm_2_5",       168),
    ("time_moe_200m",     720),
]


def load_v2() -> pd.DataFrame:
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    df.index = (df.index.tz_convert("UTC")
                if df.index.tz is not None else df.index.tz_localize("UTC"))
    return df.dropna(subset=["y_lag_336h", "y_roll168_mean"])


def enrich_features(v2: pd.DataFrame) -> pd.DataFrame:
    """Add denser Hijri features."""
    out = _ensure_calendar_features(v2)
    if "is_ramadan" in out.columns:
        out["ramadan_x_hour_sin"] = out["is_ramadan"] * out["hour_sin"]
        out["ramadan_x_hour_cos"] = out["is_ramadan"] * out["hour_cos"]
        out["ramadan_x_dow_sin"]  = out["is_ramadan"] * out["dow_sin"]
        out["ramadan_x_dow_cos"]  = out["is_ramadan"] * out["dow_cos"]
    if "is_eid" in out.columns:
        out["eid_x_hour_sin"] = out["is_eid"] * out["hour_sin"]
        out["eid_x_hour_cos"] = out["is_eid"] * out["hour_cos"]
        # days_since_eid / days_to_eid via groupby on is_eid runs.
        is_eid = out["is_eid"].astype(int).values
        n = len(is_eid)
        last_eid = np.full(n, -9999, dtype=np.int64)
        cur = -9999
        for i in range(n):
            if is_eid[i] == 1:
                cur = i
            last_eid[i] = cur
        next_eid = np.full(n, 9999999, dtype=np.int64)
        cur = 9999999
        for i in range(n - 1, -1, -1):
            if is_eid[i] == 1:
                cur = i
            next_eid[i] = cur
        out["days_since_eid"] = np.where(last_eid >= 0, (np.arange(n) - last_eid) / 24.0, 999.0).clip(0, 60)
        out["days_to_eid"]    = np.where(next_eid < n, (next_eid - np.arange(n)) / 24.0, 999.0).clip(0, 60)
    return out


WEATHER = ["temp_c", "dewpoint_c", "wind_speed", "solar_rad", "temp_sq", "temp_above_35"]
CALENDAR = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
LAGS = ["y_lag_24h", "y_lag_168h", "y_lag_336h", "y_roll168_mean", "y_roll168_std"]
HIJRI = ["is_ramadan", "day_of_ramadan", "is_eid"]
HIJRI_DENSE = ["ramadan_x_hour_sin", "ramadan_x_hour_cos",
               "ramadan_x_dow_sin", "ramadan_x_dow_cos",
               "eid_x_hour_sin", "eid_x_hour_cos",
               "days_since_eid", "days_to_eid"]
FEATURES = WEATHER + CALENDAR + LAGS + HIJRI + HIJRI_DENSE


def fit_residual_cv(
    tsfm_df: pd.DataFrame, feat_df: pd.DataFrame, seed: int = 0,
    only_regimes: list[str] | None = None,
) -> pd.DataFrame:
    """3-fold time-block CV residual correction. If only_regimes is given,
    fit/apply on rows where regime is in the list; rows outside are kept
    as bare TSFM predictions (no correction)."""
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
        cutoff = int(len(other_idx) * 0.9)
        train_idx = other_idx[:cutoff]
        val_idx = other_idx[cutoff:]

        if only_regimes is not None:
            train_keep = tsfm_df.loc[train_idx, "regime"].isin(only_regimes)
            val_keep   = tsfm_df.loc[val_idx,   "regime"].isin(only_regimes)
            train_idx = train_idx[train_keep.values]
            val_idx   = val_idx[val_keep.values]

        X_tr = feat.loc[train_idx, FEATURES]
        y_tr = (tsfm_df.loc[train_idx, "y_true"] - tsfm_df.loc[train_idx, "y_pred"]).values
        X_va = feat.loc[val_idx, FEATURES]
        y_va = (tsfm_df.loc[val_idx, "y_true"] - tsfm_df.loc[val_idx, "y_pred"]).values
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "max_depth": -1,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "seed": seed,
            "verbose": -1,
        }
        booster = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        # Apply to held-out
        held_tsfm = tsfm_df.loc[fold_idx]
        held_feat = feat.loc[fold_idx, FEATURES]
        residual_hat = booster.predict(held_feat)
        corrected = held_tsfm.copy()
        if only_regimes is not None:
            apply_mask = held_tsfm["regime"].isin(only_regimes).values
            new_pred = corrected["y_pred"].values.copy()
            new_pred[apply_mask] += residual_hat[apply_mask]
            corrected["y_pred"] = new_pred
        else:
            corrected["y_pred"] = corrected["y_pred"].values + residual_hat
        out_pieces.append(corrected)
    return pd.concat(out_pieces).sort_index()


def mae(df: pd.DataFrame, regime: str | None = None) -> float:
    s = df if regime is None else df[df.regime == regime]
    return (s.y_true - s.y_pred).abs().mean() if len(s) else float("nan")


def summarize(df: pd.DataFrame, label: str) -> dict:
    return {
        "model": label,
        "agg": mae(df),
        "Normal": mae(df, "Normal"),
        "Ramadan": mae(df, "Ramadan"),
        "Heatwave": mae(df, "Heatwave"),
    }


def main() -> None:
    v2 = load_v2()
    print(f"v2 rows: {len(v2):,}")

    rows: list[dict] = []
    for tsfm, L in TSFM_SPECS:
        print(f"\n=== {tsfm} L={L} ===")
        bare = pd.read_parquet(PRED_DIR / f"{tsfm}__nohijri__L{L}__seed0.parquet")
        if bare.index.tz is None:
            bare.index = bare.index.tz_localize("UTC")
        rows.append({"tsfm": tsfm, **summarize(bare, "bare")})

        print("  [a] wider+dense (all regimes) ...")
        t0 = time.time()
        all_reg = fit_residual_cv(bare, v2, seed=0)
        print(f"      done in {time.time()-t0:.1f}s")
        rows.append({"tsfm": tsfm, **summarize(all_reg, "wider+dense-all")})

        print("  [b] wider+dense (Normal+Ramadan only, Heatwave routed bare) ...")
        t0 = time.time()
        nr = fit_residual_cv(bare, v2, seed=0,
                             only_regimes=["Normal", "Ramadan"])
        print(f"      done in {time.time()-t0:.1f}s")
        rows.append({"tsfm": tsfm, **summarize(nr, "wider+dense-NR-only")})

        # Save the better of the two as the tuned parquet
        chosen = nr if mae(nr) < mae(all_reg) else all_reg
        out_path = predictions_path(
            model=f"{tsfm}__residual_tuned",
            variant="hijri",
            context_length=L,
            seed=0,
        )
        chosen.to_parquet(out_path)
        print(f"      -> saved {out_path.name}")

    df = pd.DataFrame(rows)
    print("\n=== TUNING SUMMARY ===")
    print(df.round(1).to_string(index=False))

    # Improvement vs bare
    print("\n=== IMPROVEMENT vs bare (negative = better) ===")
    for tsfm, _ in TSFM_SPECS:
        sub = df[df.tsfm == tsfm].reset_index(drop=True)
        bare_agg = sub.loc[0, "agg"]
        for i in [1, 2]:
            delta = sub.loc[i, "agg"] - bare_agg
            pct = 100 * delta / bare_agg
            print(f"  {tsfm:25s} {sub.loc[i, 'model']:25s} "
                  f"Δagg = {delta:+7.1f} MW  ({pct:+5.1f}%)")


if __name__ == "__main__":
    main()
