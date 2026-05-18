"""Generate report-quality figures from the existing analysis CSVs.

Produces PNG figures at 180 dpi under docs/figures/:

  fig1_leaderboard_forest.png        — top 15 systems with 95% CI bars
  fig2_per_horizon.png               — MAE vs horizon for 5 block models
  fig3_diurnal_heatmap.png           — model x hour-of-day MAE heatmap
  fig4_per_regime_bars.png           — top 8 systems x 4 regime bars
  fig5_residual_impact.png           — bare vs +residual delta per model
  fig6_failure_days.png              — top 10 universal-failure days

The script reads:
  data/statistical_appendix/ci_table.csv
  data/analysis/horizon_mae.csv
  data/analysis/diurnal_mae.csv
  data/analysis/failure_modes_common.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Cohesive minimalist style.
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
})


# Color palette: cool blue family for composites, warm orange family for
# single models, gray for baselines/classical.
PALETTE = {
    "composite": "#2E5C8A",      # navy blue
    "composite_light": "#5B8AB8",
    "tsfm": "#D97700",            # warm orange
    "tsfm_light": "#F0A540",
    "tabular": "#2E7D32",         # green
    "deep": "#7B1FA2",            # purple
    "classical": "#6B6B6B",       # gray
    "highlight": "#C62828",       # red highlight
}


def _classify_model(name: str) -> str:
    if any(k in name for k in ("ensemble", "meta-router", "routed-best", "stacked")):
        return "composite"
    if any(k in name for k in ("chronos", "time-moe", "timesfm", "moirai")):
        return "tsfm"
    if "lgbm" in name:
        return "tabular"
    if "patchtsmixer" in name:
        return "deep"
    if "mstl_ets" in name or "sarimax" in name:
        return "classical"
    return "tabular"


# ====================================================================
# Fig 1: Leaderboard forest plot
# ====================================================================
def fig1_leaderboard_forest() -> None:
    ci = pd.read_csv(ROOT / "data" / "statistical_appendix" / "ci_table.csv")
    agg = ci[ci.regime == "aggregate"].sort_values("mae").head(15).reset_index(drop=True)

    colors = [PALETTE[_classify_model(m)] for m in agg.model]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    y = np.arange(len(agg))[::-1]
    ax.errorbar(
        agg.mae, y,
        xerr=[agg.mae - agg.ci_lo, agg.ci_hi - agg.mae],
        fmt="o", color="black", ecolor="#888", elinewidth=1.2,
        capsize=3, markersize=0,
    )
    ax.scatter(agg.mae, y, c=colors, s=60, zorder=3, edgecolor="white", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(agg.model)
    ax.set_xlabel("Aggregate test MAE (MW)  —  95% block-bootstrap CI")
    ax.set_title("Top 15 forecasting systems — 14-month Turkish STLF benchmark", pad=14)
    # Baseline reference line + annotation placed above the topmost row
    # so it never collides with data.
    ax.axvline(979, color="#aaa", linestyle="--", linewidth=0.8, zorder=1)
    ax.annotate(
        "LGBM-hijri baseline (979)",
        xy=(979, len(agg) - 0.5),
        xytext=(984, len(agg) - 0.5),
        color="#555", fontsize=8, va="center", ha="left",
    )

    # Legend OUTSIDE the axes (right side) so it can't overlap data.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=PALETTE["composite"], label="Composite (ensemble / router)"),
        Patch(facecolor=PALETTE["tsfm"], label="Time-Series Foundation Model"),
        Patch(facecolor=PALETTE["tabular"], label="LightGBM (tabular)"),
        Patch(facecolor=PALETTE["deep"], label="Deep learning baseline"),
        Patch(facecolor=PALETTE["classical"], label="Classical baseline"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.savefig(FIG_DIR / "fig1_leaderboard_forest.png")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'fig1_leaderboard_forest.png'}")


# ====================================================================
# Fig 2: Per-horizon MAE lines
# ====================================================================
def fig2_per_horizon() -> None:
    df = pd.read_csv(ROOT / "data" / "analysis" / "horizon_mae.csv")
    # Distinct color per model — six models, six visually-distinct hues.
    model_colors = {
        "chronos-bolt-L720":         "#1976D2",   # blue
        "time-moe-200m-L720":        "#D97700",   # orange
        "timesfm-2.5-L168":          "#2E7D32",   # green
        "moirai-1.1-small-L336":     "#C62828",   # red
        "patchtsmixer-nohijri-L168": "#7B1FA2",   # purple solid
        "patchtsmixer-hijri-L168":   "#7B1FA2",   # purple dashed
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for model in df.model.unique():
        sub = df[df.model == model].sort_values("horizon")
        c = model_colors.get(model, "#444")
        ls = "--" if model == "patchtsmixer-hijri-L168" else "-"
        ax.plot(sub.horizon, sub.mae, marker="o", markersize=4,
                color=c, linestyle=ls, label=model, alpha=0.9, linewidth=1.8)
    ax.set_xlabel("Forecast horizon h (hours from issuance)")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Per-horizon MAE decomposition — block-forecasting models", pad=12)
    ax.set_xticks([1, 6, 12, 18, 24])
    # Legend outside the axes on the right.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.savefig(FIG_DIR / "fig2_per_horizon.png")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'fig2_per_horizon.png'}")


# ====================================================================
# Fig 3: Diurnal MAE heatmap (aggregate regime)
# ====================================================================
def fig3_diurnal_heatmap() -> None:
    df = pd.read_csv(ROOT / "data" / "analysis" / "diurnal_mae.csv")
    agg = df[df.regime == "aggregate"]
    pivot = agg.pivot(index="model", columns="hour", values="mae")
    # Sort by overall mean (best on top)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Hour of day (UTC; local = UTC + 3)", labelpad=22)
    # Title gets extra padding so the annotations placed below the
    # x-axis don't collide.
    ax.set_title("Diurnal MAE heatmap — 12 models × hour-of-day", pad=14)
    fig.colorbar(im, ax=ax, label="MAE (MW)", shrink=0.85)
    # Vertical reference lines for the two failure clusters.
    ax.axvline(5.5, color="navy", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axvline(10.5, color="darkred", linestyle=":", linewidth=1.2, alpha=0.7)
    # Annotations BELOW the x-axis (no overlap with title or heatmap).
    ax.annotate("morning ramp\n(local 8-9 AM)",
                xy=(5.5, len(pivot.index) - 0.5),
                xytext=(5.5, len(pivot.index) + 1.8),
                color="navy", fontsize=8, ha="center", va="top",
                annotation_clip=False)
    ax.annotate("afternoon peak\n(local 13-14)",
                xy=(10.5, len(pivot.index) - 0.5),
                xytext=(10.5, len(pivot.index) + 1.8),
                color="darkred", fontsize=8, ha="center", va="top",
                annotation_clip=False)
    fig.savefig(FIG_DIR / "fig3_diurnal_heatmap.png")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'fig3_diurnal_heatmap.png'}")


# ====================================================================
# Fig 4: Per-regime grouped bar chart
# ====================================================================
def fig4_per_regime_bars() -> None:
    ci = pd.read_csv(ROOT / "data" / "statistical_appendix" / "ci_table.csv")
    # Pick the 8 top aggregate systems
    top_models = ci[ci.regime == "aggregate"].sort_values("mae").head(8).model.tolist()
    pivot = ci[ci.model.isin(top_models)].pivot(
        index="model", columns="regime", values="mae"
    ).loc[top_models]
    regimes = ["Normal", "Ramadan", "Heatwave"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(pivot.index))
    width = 0.27
    for i, regime in enumerate(regimes):
        vals = pivot[regime].values
        ax.bar(x + (i - 1) * width, vals, width=width, label=regime,
               color=["#4A6FA5", "#2EA887", "#D97700"][i], alpha=0.9,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Per-regime MAE — top 8 systems by aggregate")
    ax.legend(loc="upper left", title="Regime")
    ax.set_axisbelow(True)
    fig.savefig(FIG_DIR / "fig4_per_regime_bars.png")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'fig4_per_regime_bars.png'}")


# ====================================================================
# Fig 5: Residual-correction impact bars (bare vs +residual)
# ====================================================================
def fig5_residual_impact() -> None:
    # Hardcoded from documented per-TSFM table; double-checked against ci_table.
    rows = [
        ("sarimax-hijri",          2485.9, 1299.3),
        ("patchtsmixer-L168",      1552.7, 1045.8),
        ("moirai-L336",            1727.1, 1317.2),
        ("mstl_ets-hijri",         1527.5, 1364.9),
        ("timesfm-L168",           1173.2, 1057.5),
        ("lgbm-nohijri",           1003.3,  950.7),
        ("lgbm-hijri",              979.0,  940.4),
        ("time-moe-L720",           985.9,  954.5),
        ("chronos-bolt-L720",       968.9,  948.5),
    ]
    df = pd.DataFrame(rows, columns=["model", "bare", "corrected"])
    df["delta_pct"] = 100 * (df.corrected - df.bare) / df.bare
    df = df.sort_values("delta_pct")  # most-improved first (most negative)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [1.0, 1.0]})

    # Left: bare vs corrected paired bars
    y = np.arange(len(df))[::-1]
    width = 0.4
    ax1.barh(y + width / 2, df.bare, height=width, color="#9e9e9e",
             label="Bare", edgecolor="white", linewidth=0.5)
    ax1.barh(y - width / 2, df.corrected, height=width, color="#2E5C8A",
             label="+ LGBM residual head", edgecolor="white", linewidth=0.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(df.model)
    ax1.set_xlabel("Aggregate test MAE (MW)")
    ax1.set_title("Bare vs +residual-corrected")
    ax1.legend(loc="lower right")
    ax1.set_axisbelow(True)

    # Right: improvement % bars
    colors = ["#C62828" if d < -15 else "#2E7D32" if d < 0 else "#9e9e9e" for d in df.delta_pct]
    ax2.barh(y, df.delta_pct, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(df))
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Δ aggregate MAE  (%)")
    ax2.set_title("Improvement from residual correction")
    for i, (_, row) in enumerate(df.iterrows()):
        ax2.text(row.delta_pct, y[i], f" {row.delta_pct:+.1f}%",
                 va="center", ha="left" if row.delta_pct > 0 else "right",
                 fontsize=9, color="black")
    ax2.set_axisbelow(True)

    fig.suptitle(
        "Post-hoc LightGBM residual correction — weaker bare models benefit more",
        fontsize=12, fontweight="semibold", y=1.02,
    )
    fig.savefig(FIG_DIR / "fig5_residual_impact.png")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'fig5_residual_impact.png'}")


# ====================================================================
# Fig 6: Top failure days (universal)
# ====================================================================
def fig6_failure_days() -> None:
    df = pd.read_csv(ROOT / "data" / "analysis" / "failure_modes_common.csv")
    df = df.sort_values("mean_mae_across_models", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(df))[::-1]
    # Color by anomaly type via heuristic on date+ramadan_hours
    colors = []
    labels = []
    for _, r in df.iterrows():
        if r.ramadan_hours > 0:
            colors.append("#D97700")
            labels.append(f"{r.date} — Ramadan / Eid")
        elif r.date in ("2024-01-01", "2025-01-01"):
            colors.append("#7B1FA2")
            labels.append(f"{r.date} — New Year's Day")
        elif r.date == "2024-06-15":
            colors.append("#C62828")
            labels.append(f"{r.date} — Eid al-Adha + heatwave + weekend")
        elif r.max_temp_c > 28:
            colors.append("#1976D2")
            labels.append(f"{r.date} — heatwave-period day")
        else:
            colors.append("#6B6B6B")
            labels.append(f"{r.date}")

    ax.barh(y, df.mean_mae_across_models, color=colors,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean MAE across 12 single models (MW)")
    ax.set_title("Top-10 hardest days in the test window — universal failure modes")
    for i, mae in enumerate(df.mean_mae_across_models):
        ax.text(mae, y[i], f" {mae:.0f}", va="center", fontsize=9)
    ax.set_axisbelow(True)
    fig.savefig(FIG_DIR / "fig6_failure_days.png")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'fig6_failure_days.png'}")


# ====================================================================
# Fig 7: L-sweep curves per TSFM (Ablation C)
# ====================================================================
def fig7_l_sweep() -> None:
    PRED_DIR = ROOT / "data" / "predictions"
    tsfms = [
        ("chronos_bolt_base", "chronos-bolt-base", "#1976D2"),
        ("timesfm_2_5",       "timesfm-2.5-200m",  "#2E7D32"),
        ("moirai_1_1_small",  "moirai-1.1-small",  "#C62828"),
        ("time_moe_200m",     "time-moe-200m",     "#D97700"),
    ]
    Ls = [96, 168, 336, 720]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for fname_prefix, label, color in tsfms:
        maes = []
        for L in Ls:
            f = PRED_DIR / f"{fname_prefix}__nohijri__L{L}__seed0.parquet"
            df = pd.read_parquet(f)
            maes.append((df.y_true - df.y_pred).abs().mean())
        ax.plot(Ls, maes, marker="o", markersize=7, color=color,
                label=label, linewidth=2)
        # Mark the best L per model
        best_i = int(np.argmin(maes))
        ax.scatter(Ls[best_i], maes[best_i], s=140, facecolor="white",
                   edgecolor=color, linewidth=2.4, zorder=5)
    ax.set_xticks(Ls)
    ax.set_xlabel("Context length L (hours)")
    ax.set_ylabel("Aggregate test MAE (MW)")
    ax.set_title("Ablation C — TSFM aggregate MAE vs context length\n"
                 "(open circle = per-model best L)", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.savefig(FIG_DIR / "fig7_l_sweep.png")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'fig7_l_sweep.png'}")


# ====================================================================
# Fig 8: Hijri-delta bar chart (Ablation A summary)
# ====================================================================
def fig8_hijri_delta() -> None:
    PRED_DIR = ROOT / "data" / "predictions"
    pairs = [
        ("LightGBM",                "lgbm__nohijri__seed44.parquet",
                                    "lgbm__hijri__seed44.parquet"),
        ("MSTL+ETS",                "mstl_ets__nohijri__seed0.parquet",
                                    "mstl_ets__hijri__seed0.parquet"),
        ("SARIMAX",                 "sarimax__nohijri__seed0.parquet",
                                    "sarimax__hijri__seed0.parquet"),
        ("TimesFM L=336 (HF covariate)",  "timesfm_2_5__nohijri__L336__seed0.parquet",
                                          "timesfm_2_5__hijri__L336__seed0.parquet"),
        ("Moirai L=336 (HF covariate)",   "moirai_1_1_small__nohijri__L336__seed0.parquet",
                                          "moirai_1_1_small__hijri__L336__seed0.parquet"),
        ("PatchTSMixer L=168 (X-channel)", "patchtsmixer__nohijri__L168__seed42.parquet",
                                            "patchtsmixer__hijri__L168__seed42.parquet"),
        ("Chronos L=720 (residual head)", "chronos_bolt_base__residual__nohijri__L720__seed0.parquet",
                                           "chronos_bolt_base__residual__hijri__L720__seed0.parquet"),
        ("Time-MoE L=720 (residual head)", "time_moe_200m__residual__nohijri__L720__seed0.parquet",
                                           "time_moe_200m__residual__hijri__L720__seed0.parquet"),
        ("TimesFM L=168 (residual head)", "timesfm_2_5__residual__nohijri__L168__seed0.parquet",
                                          "timesfm_2_5__residual__hijri__L168__seed0.parquet"),
        ("Moirai L=336 (residual head)",  "moirai_1_1_small__residual__nohijri__L336__seed0.parquet",
                                          "moirai_1_1_small__residual__hijri__L336__seed0.parquet"),
    ]

    rows = []
    for name, nh, h in pairs:
        d_nh = pd.read_parquet(PRED_DIR / nh)
        d_h  = pd.read_parquet(PRED_DIR / h)
        # Use Ramadan-only MAE since this is where Hijri features should matter most
        ram_nh = d_nh[d_nh.regime == "Ramadan"]
        ram_h  = d_h [d_h.regime  == "Ramadan"]
        mae_nh = (ram_nh.y_true - ram_nh.y_pred).abs().mean()
        mae_h  = (ram_h.y_true  - ram_h.y_pred).abs().mean()
        rows.append({"model": name, "nohijri": mae_nh, "hijri": mae_h,
                     "delta": mae_h - mae_nh})
    df = pd.DataFrame(rows).sort_values("delta")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ["#2E7D32" if d < 0 else "#C62828" if d > 0 else "#6B6B6B" for d in df.delta]
    y = np.arange(len(df))[::-1]
    ax.barh(y, df.delta, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df.model, fontsize=9)
    ax.set_xlabel("Δ Ramadan MAE  (hijri − nohijri, MW)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Ablation A — Ramadan MAE delta from Hijri features\n"
                 "(green = Hijri helps,  red = Hijri hurts)", pad=12)
    for i, (_, r) in enumerate(df.iterrows()):
        label = f"{r.delta:+.0f} MW  ({100*r.delta/r.nohijri:+.1f}%)"
        # Always place label OUTSIDE the bar on the side closer to zero
        # so it never crosses the y-axis label area.
        if r.delta > 0:
            # Bar goes right of 0; put label to the RIGHT of the bar's tip.
            ax.text(r.delta + 5, y[i], label,
                    va="center", ha="left", fontsize=9, color="black")
        else:
            # Bar goes left of 0; put label to the RIGHT of the bar's tip
            # (i.e., closer to zero), inside the empty space between the
            # bar tip and the zero axis.
            ax.text(r.delta + 5, y[i], label,
                    va="center", ha="left", fontsize=9, color="black")
    ax.set_axisbelow(True)
    fig.savefig(FIG_DIR / "fig8_hijri_delta.png")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'fig8_hijri_delta.png'}")


# ====================================================================
# Fig 9: DM matrix heatmap (aggregate regime)
# ====================================================================
def fig9_dm_heatmap() -> None:
    dm = pd.read_csv(ROOT / "data" / "statistical_appendix" / "dm_aggregate.csv")
    ci = pd.read_csv(ROOT / "data" / "statistical_appendix" / "ci_table.csv")
    # Order models by aggregate MAE (best first)
    order = ci[ci.regime == "aggregate"].sort_values("mae").model.tolist()
    n = len(order)
    idx = {m: i for i, m in enumerate(order)}

    # Build a (n, n) matrix of dm_stats, lower-triangular only
    M = np.full((n, n), np.nan)
    SIG = np.full((n, n), np.nan)  # 1 if p_holm<0.05 else 0
    for _, row in dm.iterrows():
        i, j = idx.get(row.model_i), idx.get(row.model_j)
        if i is None or j is None or np.isnan(row.dm_stat):
            continue
        # CSV has (i, j) where i<j. We want a directional matrix where
        # M[a, b] > 0 means column model is better than row model.
        # dm_test returns positive when model_j is better than model_i.
        # Place the value at [row=model_i, col=model_j].
        M[i, j] = row.dm_stat
        SIG[i, j] = 1.0 if (not np.isnan(row.p_holm)) and row.p_holm < 0.05 else 0.0

    fig, ax = plt.subplots(figsize=(14, 12.5))
    # Diverging colormap centered at 0
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(order, rotation=75, ha="right", fontsize=7.5)
    ax.set_yticklabels(order, fontsize=7.5)

    # Mark significant cells with a small dot
    for i in range(n):
        for j in range(n):
            if not np.isnan(SIG[i, j]) and SIG[i, j] > 0:
                ax.plot(j, i, marker=".", color="black", markersize=4)

    ax.set_title("Diebold-Mariano statistic — aggregate regime, Holm-adjusted\n"
                 "Cell color: green = column model significantly better;  "
                 "red = row model better.   Black dot = p_holm < 0.05.",
                 pad=22, fontsize=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="DM statistic", pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(FIG_DIR / "fig9_dm_heatmap.png")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'fig9_dm_heatmap.png'}")


# ====================================================================
# Fig 10: Pipeline / architecture diagram
# ====================================================================
def fig10_pipeline() -> None:
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 64)
    ax.axis("off")

    def box(x, y, w, h, text, color, edge="black", text_color="white", fs=10):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.4,rounding_size=1.2",
                           facecolor=color, edgecolor=edge, linewidth=1.4)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=text_color, fontsize=fs, fontweight="semibold")

    def arrow(x1, y1, x2, y2, color="#444"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=color, linewidth=1.4))

    # ---- Top row: data sources
    box(2, 50, 22, 7, "EPIAS load\n(2018-2025 hourly)", "#37474F")
    box(28, 50, 22, 7, "ERA5 weather\n(7 southern cities)", "#37474F")
    box(54, 50, 22, 7, "Hijri calendar\n(hijridate library)", "#37474F")

    # ---- Feature builder
    box(28, 38, 44, 6, "src/data/build_v2_dataset.py — feature engineering",
        "#2C3E50", fs=10)
    for x in (13, 39, 65):
        arrow(x, 50, x + 0.5, 44)

    # ---- v2 dataset
    box(36, 30, 28, 5, "final_training_set_v2.csv  (60k rows × 25 features)",
        "#1565C0", fs=9.5)
    arrow(50, 38, 50, 35)

    # ---- Model families (parallel layer)
    box(2,  18, 18, 8, "LightGBM\n(Plan 1)\n3 var × 5 seeds", "#2E7D32", fs=9)
    box(22, 18, 18, 8, "TSFMs ×4\n(Plans 2-3)\nL ∈ {96,168,336,720}",
        "#D97700", fs=9)
    box(42, 18, 18, 8, "Classical\n(Plan 4)\nMSTL+ETS, SARIMAX", "#6B6B6B", fs=9)
    box(62, 18, 18, 8, "PatchTSMixer\n(Plan 5)\nL=168, seed 42", "#7B1FA2", fs=9)
    box(82, 18, 14, 8, "Residual\nheads\n(Plan 6)", "#5B8AB8", fs=9)
    for x in (11, 31, 51, 71, 89):
        arrow(50, 30, x, 26)

    # ---- Composite layer
    box(15, 7, 60, 6, "Tier 1-3 composites: ensembles, regime router, meta-router, stacked LGBM",
        "#2E5C8A", fs=10)
    for x in (11, 31, 51, 71, 89):
        arrow(x, 18, x, 13)

    # ---- Eval block
    box(15, -1, 60, 5, "Evaluation: regime metrics, DM (HAC), block bootstrap, smoke tests",
        "#1A237E", fs=9.5)
    arrow(45, 7, 45, 4)

    ax.text(50, 62.5,
            "Pipeline overview — data → 31 forecasting systems → unified evaluation",
            ha="center", va="center", fontsize=13, fontweight="semibold")

    fig.savefig(FIG_DIR / "fig10_pipeline.png")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'fig10_pipeline.png'}")


# ====================================================================
# Fig 11: Sample-week forecast vs actual
# ====================================================================
def fig11_sample_weeks() -> None:
    PRED_DIR = ROOT / "data" / "predictions"
    router = pd.read_parquet(PRED_DIR / "meta_router_v2__seed0.parquet")
    chronos = pd.read_parquet(PRED_DIR / "chronos_bolt_base__nohijri__L720__seed0.parquet")
    if router.index.tz is None:
        router.index = router.index.tz_localize("UTC")
    if chronos.index.tz is None:
        chronos.index = chronos.index.tz_localize("UTC")

    # Pick one representative week per regime
    weeks = {
        "Normal":   ("2024-02-12", "2024-02-18"),
        "Ramadan":  ("2024-03-15", "2024-03-21"),  # mid-Ramadan 2024
        "Heatwave": ("2024-08-12", "2024-08-18"),  # mid heatwave 2024
    }

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.5), sharex=False)
    handles_global = labels_global = None
    for i, (ax, (label, (start, end))) in enumerate(zip(axes, weeks.items())):
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        mask = (router.index >= start_ts) & (router.index < end_ts)
        r_sub = router[mask]
        c_sub = chronos[(chronos.index >= start_ts) & (chronos.index < end_ts)]
        if len(r_sub) == 0:
            ax.text(0.5, 0.5, f"No {label} data in {start}..{end}",
                    transform=ax.transAxes, ha="center")
            continue
        h1, = ax.plot(r_sub.index, r_sub.y_true, color="black",
                      label="actual load", linewidth=1.7)
        h2, = ax.plot(c_sub.index, c_sub.y_pred, color="#D97700",
                      label="Chronos-L720 bare", linewidth=1.4, alpha=0.85)
        h3, = ax.plot(r_sub.index, r_sub.y_pred, color="#1976D2",
                      label="meta-router-v2 (composite)", linewidth=1.4, alpha=0.95)
        ax.set_title(f"{label} week  ({start} → {end})", fontsize=11, pad=6)
        ax.set_ylabel("Load (MW)")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%a %m-%d"))
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        if handles_global is None:
            handles_global = [h1, h2, h3]
            labels_global = ["actual load", "Chronos-L720 bare",
                             "meta-router-v2 (composite)"]
    fig.suptitle("Sample-week forecasts: meta-router-v2 vs Chronos-bare vs actuals",
                 fontsize=13, fontweight="semibold", y=0.995)
    # Single legend at the top, below the suptitle.
    fig.legend(handles_global, labels_global,
               loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig11_sample_weeks.png")
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'fig11_sample_weeks.png'}")


def main() -> None:
    print("Building figures ...")
    fig1_leaderboard_forest()
    fig2_per_horizon()
    fig3_diurnal_heatmap()
    fig4_per_regime_bars()
    fig5_residual_impact()
    fig6_failure_days()
    fig7_l_sweep()
    fig8_hijri_delta()
    fig9_dm_heatmap()
    fig10_pipeline()
    fig11_sample_weeks()
    print(f"\nAll figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
