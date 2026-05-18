"""Build docs/deep_analysis.md + data/analysis/*.csv

Two analyses on top of the existing prediction parquets:

1. **Per-horizon MAE decomposition** (5 block-forecaster models with `y_block`):
   for each model and each horizon h in {1..24} from issuance time
   t = tau - 24, compute MAE of `block[h-1]` against the true value at
   `tau - 24 + h`. Shows how forecast error grows with horizon.

2. **Diurnal MAE** (all 12 headline models): for each model, MAE bucketed
   by hour-of-day of tau (UTC). Both aggregate and per-regime.

Outputs:
  - data/analysis/horizon_mae.csv      (long: model, horizon, mae, n)
  - data/analysis/diurnal_mae.csv      (long: model, hour, regime, mae, n)
  - docs/deep_analysis.md              human-readable summary

Reused infrastructure: build_statistical_appendix.MODELS for the canonical
model list, predictions_io schema (y_true, y_pred, regime, y_block).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow direct CLI run: register the project root on sys.path so the
# sibling-script import below works without making scripts/ a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_statistical_appendix import MODELS, PRED_DIR


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data" / "analysis"
DOC_PATH = ROOT / "docs" / "deep_analysis.md"

# Subset of MODELS that have y_block (TSFM family + PatchTSMixer).
BLOCK_MODELS = [
    (n, f) for (n, f) in MODELS
    if any(prefix in f for prefix in (
        "chronos_bolt_base", "timesfm_2_5", "moirai", "time_moe", "patchtsmixer"
    ))
]


def _load_one(fname: str) -> pd.DataFrame:
    df = pd.read_parquet(PRED_DIR / fname)
    # Normalize tz
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def compute_horizon_mae(spec: list[tuple[str, str]] = BLOCK_MODELS) -> pd.DataFrame:
    """For each block-forecaster, MAE per horizon h in {1..24} from issuance.

    horizon h refers to: block[h-1] predicts y at (tau - 24 + h).
    h=24 (block[23]) is the canonical y_pred that we already report.

    Vectorized: at each h we shift the tau index by (24 - h) hours back to
    get the target timestamp, then reindex y_true onto those targets and
    take elementwise |pred - true|. For tau values where the target falls
    before the test window starts, the reindex returns NaN and that pair
    is dropped.
    """
    rows = []
    for name, fname in spec:
        df = _load_one(fname)
        y_true_series = df["y_true"]
        # Stack y_block into (N, 24) array for vectorized indexing.
        block_arr = np.stack([np.asarray(b, dtype=float) for b in df["y_block"].values])
        tau_index = df.index
        for h in range(1, 25):
            target_ts = tau_index - pd.Timedelta(hours=24 - h)
            y_true_at_target = y_true_series.reindex(target_ts).values
            y_pred_at_h = block_arr[:, h - 1]
            valid = ~np.isnan(y_true_at_target)
            if not valid.any():
                continue
            errs = np.abs(y_pred_at_h[valid] - y_true_at_target[valid])
            rows.append({
                "model": name, "horizon": h,
                "mae": float(errs.mean()),
                "n": int(valid.sum()),
            })
    return pd.DataFrame(rows, columns=["model", "horizon", "mae", "n"])


def compute_diurnal_mae(spec: list[tuple[str, str]] = MODELS) -> pd.DataFrame:
    """For each model, MAE per (regime, hour-of-day of tau).

    regime='aggregate' rows are the all-regime MAE per hour.
    """
    rows = []
    for name, fname in spec:
        df = _load_one(fname).copy()
        df["abs_err"] = np.abs(df["y_true"] - df["y_pred"])
        df["hour"] = df.index.hour
        # aggregate per hour
        for hour, sub in df.groupby("hour"):
            rows.append({
                "model": name, "hour": int(hour), "regime": "aggregate",
                "mae": float(sub["abs_err"].mean()),
                "n": int(len(sub)),
            })
        # per-regime per hour
        for (regime, hour), sub in df.groupby(["regime", "hour"]):
            if len(sub) == 0:
                continue
            rows.append({
                "model": name, "hour": int(hour), "regime": str(regime),
                "mae": float(sub["abs_err"].mean()),
                "n": int(len(sub)),
            })
    return pd.DataFrame(rows, columns=["model", "hour", "regime", "mae", "n"])


def render_markdown(
    horizon_df: pd.DataFrame,
    diurnal_df: pd.DataFrame,
    n_tau: int,
) -> str:
    """Render the deep-analysis doc."""
    lines: list[str] = []
    lines.append("# Deep Analysis: Per-Horizon and Diurnal Error Decomposition")
    lines.append("")
    lines.append(
        "Two analyses on top of the prediction parquets, using the saved "
        "`y_block` columns (5 block-forecaster models) and per-tau "
        "predictions (all 12 models)."
    )
    lines.append("")
    lines.append(f"**Test rows per model:** {n_tau:,}")
    lines.append("")

    # ---- Per-horizon section ----
    lines.append("## 1. Per-horizon MAE decomposition")
    lines.append("")
    lines.append(
        "For each block-forecaster model, MAE at each horizon h in {1..24} "
        "from the issuance time t = tau - 24. Horizon 24 is the canonical "
        "y_pred reported in the headline tables (block[23])."
    )
    lines.append("")

    # Wide table: model × horizon in {1, 6, 12, 18, 24} summary
    SUMMARY_H = [1, 6, 12, 18, 24]
    lines.append("**MAE at selected horizons** (rest in `data/analysis/horizon_mae.csv`):")
    lines.append("")
    header = "| Model | " + " | ".join(f"h={h}" for h in SUMMARY_H) + " | h=24/h=1 ratio |"
    sep = "|" + "---|" * (len(SUMMARY_H) + 2)
    lines.append(header)
    lines.append(sep)
    for model in horizon_df["model"].unique():
        sub = horizon_df[horizon_df["model"] == model].set_index("horizon")
        cells = []
        for h in SUMMARY_H:
            if h in sub.index:
                cells.append(f"{sub.loc[h, 'mae']:.0f}")
            else:
                cells.append("—")
        # ratio: MAE at h=24 / MAE at h=1
        if 1 in sub.index and 24 in sub.index:
            ratio = sub.loc[24, "mae"] / sub.loc[1, "mae"]
            cells.append(f"{ratio:.2f}x")
        else:
            cells.append("—")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "A ratio close to 1.00 means the model's forecast quality is "
        "approximately constant across the 24-hour horizon — a sign of a "
        "good direct-prediction architecture. A ratio >> 1.00 means error "
        "compounds at long horizon (typical of autoregressive models)."
    )
    lines.append("")

    # ---- Diurnal section ----
    lines.append("## 2. Diurnal MAE (by hour-of-day of tau, UTC)")
    lines.append("")
    lines.append(
        "MAE bucketed by hour-of-day. Hours are UTC; local Turkish time = "
        "UTC + 3. Local morning peak is UTC 04-06; local evening peak is "
        "UTC 16-19."
    )
    lines.append("")

    # Aggregate diurnal: model × hour-bucket
    diurnal_agg = diurnal_df[diurnal_df["regime"] == "aggregate"]
    # Pick a 4-hour summary: {0-3, 4-7, 8-11, 12-15, 16-19, 20-23} -> 6 bins
    BIN_LABELS = ["00-03 (UTC)", "04-07", "08-11", "12-15", "16-19", "20-23"]
    BIN_HOURS = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19), (20, 23)]
    lines.append("**Aggregate-regime MAE by 4-hour UTC bin** (full hourly in `data/analysis/diurnal_mae.csv`):")
    lines.append("")
    header = "| Model | " + " | ".join(BIN_LABELS) + " |"
    sep = "|" + "---|" * (len(BIN_LABELS) + 1)
    lines.append(header)
    lines.append(sep)
    for model in diurnal_agg["model"].unique():
        sub = diurnal_agg[diurnal_agg["model"] == model].set_index("hour")
        cells = []
        for (lo, hi) in BIN_HOURS:
            mask = sub.index.isin(range(lo, hi + 1))
            if mask.any():
                cells.append(f"{sub.loc[mask, 'mae'].mean():.0f}")
            else:
                cells.append("—")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")

    # Peak-hour callout
    lines.append("### Peak hours per model")
    lines.append("")
    lines.append(
        "The UTC hour where each model's aggregate MAE is highest, with "
        "the value. Highlights where each model fails most."
    )
    lines.append("")
    lines.append("| Model | Worst hour (UTC) | Worst-hour MAE | Best hour | Best-hour MAE |")
    lines.append("|---|---|---|---|---|")
    for model in diurnal_agg["model"].unique():
        sub = diurnal_agg[diurnal_agg["model"] == model].set_index("hour").sort_index()
        worst_h = int(sub["mae"].idxmax())
        best_h = int(sub["mae"].idxmin())
        lines.append(
            f"| {model} | {worst_h:02d} | {sub.loc[worst_h, 'mae']:.0f} | "
            f"{best_h:02d} | {sub.loc[best_h, 'mae']:.0f} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    print("[1/4] Loading + computing per-horizon MAE (5 block models, ~3 min)...")
    horizon_df = compute_horizon_mae()
    print(f"      {len(horizon_df)} rows")

    print("[2/4] Computing diurnal MAE (12 models)...")
    diurnal_df = compute_diurnal_mae()
    print(f"      {len(diurnal_df)} rows")

    print("[3/4] Writing CSVs ...")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    horizon_df.to_csv(ANALYSIS_DIR / "horizon_mae.csv", index=False)
    diurnal_df.to_csv(ANALYSIS_DIR / "diurnal_mae.csv", index=False)

    # n_tau = test rows per model (use the first model in MODELS as reference)
    ref_name, ref_fname = MODELS[0]
    n_tau = len(_load_one(ref_fname))

    print("[4/4] Rendering markdown ...")
    md = render_markdown(horizon_df, diurnal_df, n_tau=n_tau)
    DOC_PATH.write_text(md, encoding="utf-8")
    print(f"      -> {DOC_PATH}")
    print(f"      -> {ANALYSIS_DIR / 'horizon_mae.csv'}")
    print(f"      -> {ANALYSIS_DIR / 'diurnal_mae.csv'}")


if __name__ == "__main__":
    main()
