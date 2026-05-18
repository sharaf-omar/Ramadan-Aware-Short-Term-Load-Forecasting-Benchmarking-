"""Build docs/failure_modes.md + data/analysis/failure_modes_*.csv

For each headline model, identify the worst-MAE *days* in the test
window and surface their conditions (regime, mean temp, max temp,
Ramadan day count, weekend flag). Two artifacts:

  - data/analysis/failure_modes_per_model.csv
        long format: model, date, day_mae, regime_dominant, mean_temp_c,
        max_temp_c, ramadan_hours, weekend
  - data/analysis/failure_modes_common.csv
        the top N worst-MAE days *across all models* (intersection-on-day),
        showing which days break everyone

  - docs/failure_modes.md     human-readable summary
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_statistical_appendix import MODELS, PRED_DIR


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data" / "analysis"
DOC_PATH = ROOT / "docs" / "failure_modes.md"
V2_CSV = ROOT / "data" / "processed" / "final_training_set_v2.csv"

TOP_N = 10


def _load_v2() -> pd.DataFrame:
    df = pd.read_csv(V2_CSV, parse_dates=["timestamp"]).set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.loc["2024-01-01":"2025-03-31"]


def compute_per_model_worst_days(
    spec: list[tuple[str, str]] = MODELS,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """For each model, top-N worst test days by mean daily MAE."""
    v2 = _load_v2()
    rows = []
    for name, fname in spec:
        df = pd.read_parquet(PRED_DIR / fname)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.copy()
        df["abs_err"] = np.abs(df["y_true"] - df["y_pred"])
        df["date"] = df.index.date
        daily = df.groupby("date").agg(
            day_mae=("abs_err", "mean"),
            n_hours=("abs_err", "size"),
            regime_dominant=("regime",
                             lambda s: s.value_counts().index[0]),
        )
        worst = daily.nlargest(top_n, "day_mae").reset_index()

        # Enrich with v2 conditions.
        v2_daily = v2.groupby(v2.index.date).agg(
            mean_temp_c=("temp_c", "mean"),
            max_temp_c=("temp_c", "max"),
            ramadan_hours=("is_ramadan", "sum") if "is_ramadan" in v2.columns else ("temp_c", "size"),
        )
        if "is_ramadan" not in v2.columns:
            v2_daily["ramadan_hours"] = 0
        weekend_flag = pd.Series(
            {d: pd.Timestamp(d).weekday() >= 5 for d in v2_daily.index},
            name="weekend",
        )
        v2_daily["weekend"] = weekend_flag

        for _, r in worst.iterrows():
            cond = v2_daily.loc[r["date"]] if r["date"] in v2_daily.index else None
            rows.append({
                "model": name,
                "date": r["date"],
                "day_mae": float(r["day_mae"]),
                "n_hours": int(r["n_hours"]),
                "regime_dominant": str(r["regime_dominant"]),
                "mean_temp_c": float(cond["mean_temp_c"]) if cond is not None else np.nan,
                "max_temp_c": float(cond["max_temp_c"]) if cond is not None else np.nan,
                "ramadan_hours": int(cond["ramadan_hours"]) if cond is not None else 0,
                "weekend": bool(cond["weekend"]) if cond is not None else False,
            })
    return pd.DataFrame(rows)


def compute_universal_worst_days(
    spec: list[tuple[str, str]] = MODELS,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Days where the AVERAGE MAE across all models is highest."""
    v2 = _load_v2()
    per_model_daily: list[pd.DataFrame] = []
    for name, fname in spec:
        df = pd.read_parquet(PRED_DIR / fname)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.copy()
        df["abs_err"] = np.abs(df["y_true"] - df["y_pred"])
        df["date"] = df.index.date
        daily = df.groupby("date")["abs_err"].mean().rename(name)
        per_model_daily.append(daily)
    matrix = pd.concat(per_model_daily, axis=1)
    matrix["mean_mae_across_models"] = matrix.mean(axis=1)
    worst = matrix.nlargest(top_n, "mean_mae_across_models").reset_index()

    # Enrich with v2 conditions.
    v2_daily = v2.groupby(v2.index.date).agg(
        mean_temp_c=("temp_c", "mean"),
        max_temp_c=("temp_c", "max"),
    )
    if "is_ramadan" in v2.columns:
        v2_daily["ramadan_hours"] = v2.groupby(v2.index.date)["is_ramadan"].sum()
    else:
        v2_daily["ramadan_hours"] = 0
    if "regime" in v2.columns:
        v2_daily["regime"] = v2.groupby(v2.index.date)["regime"].agg(
            lambda s: s.value_counts().index[0]
        )
    else:
        v2_daily["regime"] = "unknown"
    worst = worst.merge(v2_daily, left_on="date", right_index=True, how="left")
    worst["weekend"] = worst["date"].apply(lambda d: pd.Timestamp(d).weekday() >= 5)
    return worst


def render_markdown(
    per_model_df: pd.DataFrame,
    universal_df: pd.DataFrame,
    top_n: int,
) -> str:
    lines: list[str] = []
    lines.append("# Failure-Mode Analysis")
    lines.append("")
    lines.append(
        f"For each of the 12 headline models, the top-{top_n} worst test "
        f"days (by mean daily MAE) plus the {top_n} days where every "
        f"model fails together (mean MAE across all models). Conditions "
        f"are enriched from the v2 dataset (mean/max temp, Ramadan hour "
        f"count, dominant regime, weekend flag).")
    lines.append("")
    lines.append(
        f"Test window: 2024-01-01 to 2025-03-31. N test days = "
        f"{(pd.Timestamp('2025-03-31') - pd.Timestamp('2024-01-01')).days + 1}.")
    lines.append("")

    # ---- Universal worst days ----
    lines.append("## Days where everyone fails (mean MAE across 12 models)")
    lines.append("")
    lines.append(
        "These days are hard for the dataset, not just for one model — "
        "useful for sanity-checking 'is the headline model bad, or is the "
        "data bad here?'")
    lines.append("")
    lines.append("| Date | Mean MAE | Regime | Mean temp (°C) | Max temp (°C) | Ramadan hours | Weekend |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in universal_df.iterrows():
        regime = r.get("regime", "—")
        mt = r.get("mean_temp_c", np.nan)
        xt = r.get("max_temp_c", np.nan)
        rh = int(r.get("ramadan_hours", 0))
        lines.append(
            f"| {r['date']} | {r['mean_mae_across_models']:.0f} | {regime} | "
            f"{mt:.1f} | {xt:.1f} | {rh} | {'Y' if r['weekend'] else 'N'} |"
        )
    lines.append("")

    # ---- Per-model worst-day summary ----
    lines.append(f"## Top-{top_n} worst days per model — regime / weekday summary")
    lines.append("")
    lines.append(
        "For each model, what fraction of its top-N worst days fall in "
        "each regime, and what fraction are weekends? Highlights "
        "systematic failure modes.")
    lines.append("")
    lines.append("| Model | Worst day MAE | Median worst-day MAE | Top-N regimes (count) | Weekend % |")
    lines.append("|---|---|---|---|---|")
    for model, sub in per_model_df.groupby("model"):
        regimes_str = ", ".join(
            f"{r}:{c}" for r, c in sub["regime_dominant"].value_counts().items()
        )
        weekend_pct = 100 * sub["weekend"].mean()
        lines.append(
            f"| {model} | {sub['day_mae'].max():.0f} | "
            f"{sub['day_mae'].median():.0f} | {regimes_str} | {weekend_pct:.0f}% |"
        )
    lines.append("")

    # ---- Worst day per model ----
    lines.append("## Single worst day per model (with conditions)")
    lines.append("")
    lines.append("| Model | Worst day | MAE | Regime | Mean temp | Max temp | Ramadan hrs | Wknd |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for model, sub in per_model_df.groupby("model"):
        w = sub.loc[sub["day_mae"].idxmax()]
        lines.append(
            f"| {model} | {w['date']} | {w['day_mae']:.0f} | "
            f"{w['regime_dominant']} | {w['mean_temp_c']:.1f} | "
            f"{w['max_temp_c']:.1f} | {w['ramadan_hours']} | "
            f"{'Y' if w['weekend'] else 'N'} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    print(f"[1/4] Computing per-model top-{TOP_N} worst days (12 models)...")
    per_model = compute_per_model_worst_days()
    print(f"      {len(per_model)} rows")

    print(f"[2/4] Computing universal worst {TOP_N} days...")
    universal = compute_universal_worst_days()
    print(f"      {len(universal)} rows")

    print("[3/4] Writing CSVs ...")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    per_model.to_csv(ANALYSIS_DIR / "failure_modes_per_model.csv", index=False)
    universal.to_csv(ANALYSIS_DIR / "failure_modes_common.csv", index=False)

    print("[4/4] Rendering markdown ...")
    md = render_markdown(per_model, universal, top_n=TOP_N)
    DOC_PATH.write_text(md, encoding="utf-8")
    print(f"      -> {DOC_PATH}")
    print(f"      -> {ANALYSIS_DIR / 'failure_modes_per_model.csv'}")
    print(f"      -> {ANALYSIS_DIR / 'failure_modes_common.csv'}")


if __name__ == "__main__":
    main()
