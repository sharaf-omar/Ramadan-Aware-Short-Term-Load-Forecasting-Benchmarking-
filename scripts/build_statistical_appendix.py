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
    # Plan 6: post-hoc LGBM residual heads on the 4 TSFMs, 2 variants each.
    ("chronos-bolt-L720+res-nh",  "chronos_bolt_base__residual__nohijri__L720__seed0.parquet"),
    ("chronos-bolt-L720+res-h",   "chronos_bolt_base__residual__hijri__L720__seed0.parquet"),
    ("moirai-L336+res-nh",        "moirai_1_1_small__residual__nohijri__L336__seed0.parquet"),
    ("moirai-L336+res-h",         "moirai_1_1_small__residual__hijri__L336__seed0.parquet"),
    ("timesfm-L168+res-nh",       "timesfm_2_5__residual__nohijri__L168__seed0.parquet"),
    ("timesfm-L168+res-h",        "timesfm_2_5__residual__hijri__L168__seed0.parquet"),
    ("time-moe-L720+res-nh",      "time_moe_200m__residual__nohijri__L720__seed0.parquet"),
    ("time-moe-L720+res-h",       "time_moe_200m__residual__hijri__L720__seed0.parquet"),
    # Tier-1 quick-win artifacts: ensemble + PatchTSMixer+residual + regime-routed best-of.
    ("patchtsmixer-L168+res-h",   "patchtsmixer__residual__hijri__L168__seed42.parquet"),
    ("ensemble-top4-median",      "ensemble__top4__seed0.parquet"),
    ("routed-best-per-regime",    "routed__best_per_regime__seed0.parquet"),
    # Tier-2 quick-win artifacts: residual heads on LGBM and classical baselines + improved ensemble.
    ("lgbm-nohijri+res-h",        "lgbm__nohijri__residual_h__seed44.parquet"),
    ("lgbm-hijri+res-h",          "lgbm__hijri__residual_h__seed44.parquet"),
    ("mstl_ets-hijri+res-h",      "mstl_ets__hijri__residual_h__seed0.parquet"),
    ("sarimax-hijri+res-h",       "sarimax__hijri__residual_h__seed0.parquet"),
    ("ensemble-top4-residual-median", "ensemble__top4_residual__seed0.parquet"),
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


def _sig_marker(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _format_ci_cell(mae: float, lo: float, hi: float) -> str:
    if pd.isna(mae):
        return "—"
    return f"{mae:.1f} [{lo:.1f}, {hi:.1f}]"


def _format_dm_cell(stat: float, p_holm: float) -> str:
    if pd.isna(stat):
        return "—"
    marker = _sig_marker(p_holm)
    if marker == "ns":
        return f"{stat:+.1f} ns"
    return f"{stat:+.1f} {marker}"


def render_markdown(
    ci_df: pd.DataFrame,
    dm_by_regime: dict[str, pd.DataFrame],
    n_tau: int,
) -> str:
    """Render the full appendix as markdown."""
    lines: list[str] = []
    lines.append("# Statistical Appendix")
    lines.append("")
    lines.append(
        "Canonical statistical-rigor artifact for the benchmark. Block-"
        "bootstrap 95% CIs around MAE for every headline model × regime, "
        "plus full pairwise Diebold-Mariano matrices (Holm-Bonferroni "
        "adjusted within each regime).")
    lines.append("")
    lines.append(f"**Intersection set size (n=τ rows across all models):** {n_tau:,}")
    lines.append("")
    lines.append(
        "**Bootstrap:** stationary block bootstrap (Politis & Romano 1994), "
        "block_size=24h, 1000 resamples, alpha=0.05, seed=0.")
    lines.append("")
    lines.append(
        "**DM test:** MAE loss, HAC h=24, two-sided. Holm-Bonferroni applied "
        "within each regime's pairwise family. Significance markers: "
        "`***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` otherwise. "
        "DM stat sign convention (from `src.evaluation.dm_test`): positive "
        "means model_j (column) has lower loss; negative means model_i (row).")
    lines.append("")

    lines.append("## Bootstrap MAE confidence intervals")
    lines.append("")
    regimes = list(dm_by_regime.keys())
    header = "| Model | " + " | ".join(regimes) + " |"
    sep = "|" + "---|" * (len(regimes) + 1)
    lines.append(header)
    lines.append(sep)
    pivot = ci_df.pivot(index="model", columns="regime")
    for model in ci_df["model"].unique():
        cells = []
        for r in regimes:
            try:
                mae = pivot.loc[model, ("mae", r)]
                lo = pivot.loc[model, ("ci_lo", r)]
                hi = pivot.loc[model, ("ci_hi", r)]
                cells.append(_format_ci_cell(mae, lo, hi))
            except KeyError:
                cells.append("—")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Pairwise Diebold-Mariano tests")
    lines.append("")
    model_order = list(ci_df["model"].unique())
    for regime in regimes:
        lines.append(f"### DM matrix — {regime}")
        lines.append("")
        dm = dm_by_regime[regime]
        cells: dict[tuple[str, str], tuple[float, float]] = {}
        for _, row in dm.iterrows():
            cells[(row["model_i"], row["model_j"])] = (row["dm_stat"], row["p_holm"])
        header_cells = ["row \\ col"] + model_order
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "---|" * len(header_cells))
        for i, ri in enumerate(model_order):
            row_cells: list[str] = [ri]
            for j, rj in enumerate(model_order):
                if j <= i:
                    row_cells.append("")
                else:
                    stat, p_holm = cells.get((ri, rj), (np.nan, np.nan))
                    row_cells.append(_format_dm_cell(stat, p_holm))
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    print("[1/4] Loading 12 prediction parquets ...")
    preds = load_predictions(MODELS)
    n_tau = len(next(iter(preds.values())))
    print(f"      intersection-on-tau: {n_tau:,} rows")

    print("[2/4] Computing bootstrap MAE CIs (12 × 4 regimes = 48 cells, ~5 min) ...")
    ci_df = compute_ci_table(preds, regimes=REGIMES)

    print("[3/4] Computing pairwise DM matrices (4 regimes × 66 pairs each, ~5 min) ...")
    dm_by_regime: dict[str, pd.DataFrame] = {}
    for regime in REGIMES:
        print(f"      DM regime={regime}")
        dm_by_regime[regime] = compute_dm_matrix(preds, regime)

    print("[4/4] Writing outputs ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ci_df.to_csv(OUT_DIR / "ci_table.csv", index=False)
    for regime, dm in dm_by_regime.items():
        dm.to_csv(OUT_DIR / f"dm_{regime}.csv", index=False)
    md = render_markdown(ci_df, dm_by_regime, n_tau=n_tau)
    DOC_PATH.write_text(md, encoding="utf-8")
    print(f"      -> {DOC_PATH}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"      -> {f}")


if __name__ == "__main__":
    main()
