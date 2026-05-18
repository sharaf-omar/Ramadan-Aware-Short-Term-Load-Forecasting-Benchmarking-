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


def main() -> None:
    print("Building figures ...")
    fig1_leaderboard_forest()
    fig2_per_horizon()
    fig3_diurnal_heatmap()
    fig4_per_regime_bars()
    fig5_residual_impact()
    fig6_failure_days()
    print(f"\nAll figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
