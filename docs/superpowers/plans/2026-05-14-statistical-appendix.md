# Statistical Appendix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One-shot script that produces `docs/statistical_appendix.md` and its supporting CSVs from the 12 headline prediction parquets — 95% block-bootstrap MAE CIs plus full pairwise DM matrices (Holm-Bonferroni adjusted) across 4 regime slices.

**Architecture:** Single script `scripts/build_statistical_appendix.py` with four pure-function building blocks (`load_predictions`, `compute_ci_table`, `compute_dm_matrix`, `render_markdown`) and a `main()` that wires them. All statistics come from the existing `src.evaluation.bootstrap.block_bootstrap_ci`, `src.evaluation.dm_test.dm_test`, and `src.evaluation.dm_test.holm_bonferroni`. No changes to `src/evaluation/*`.

**Tech Stack:** Python 3.12, pandas 2.1.4, numpy 1.26.4, existing project eval utilities. CPU-only, ~10-15 min runtime.

**Spec:** [`docs/superpowers/specs/2026-05-14-statistical-appendix-design.md`](../specs/2026-05-14-statistical-appendix-design.md)

---

## File structure (locked in)

| File | Responsibility |
|------|----------------|
| `scripts/build_statistical_appendix.py` | Generator: 4 helpers + `main()`. Idempotent — re-running overwrites outputs. |
| `tests/test_statistical_appendix.py` | Unit tests for the four helpers using synthetic mini-parquets. |
| `tests/test_smoke_pipeline.py` | Add one smoke check that the appendix doc + 5 CSVs exist. |
| `docs/statistical_appendix.md` | Generated artifact. |
| `data/statistical_appendix/ci_table.csv` | 12×4 CI rows (long format). |
| `data/statistical_appendix/dm_{aggregate,Normal,Ramadan,Heatwave}.csv` | 4 DM matrices (long format: model_i, model_j, dm_stat, p_raw, p_holm). |

The script lives in `scripts/` because it's a one-shot regenerator, not library code. Tests live in `tests/` per the repo convention.

---

## Task 1: `load_predictions` (intersection-on-τ loader)

**Files:**
- Create: `scripts/build_statistical_appendix.py`
- Create: `tests/test_statistical_appendix.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_statistical_appendix.py`:

```python
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.build_statistical_appendix import load_predictions, MODELS


def _write_parquet(tmp_path, fname, rows):
    p = tmp_path / fname
    df = pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(rows["timestamp"], tz="UTC", name="timestamp"),
    ).drop(columns="timestamp")
    df.to_parquet(p)
    return p


def test_load_predictions_intersects_on_tau(tmp_path, monkeypatch):
    # Two model parquets with overlapping but not identical timestamps.
    ts = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    df_a = pd.DataFrame({
        "timestamp": ts[:8],
        "y_true": np.arange(8, dtype=float),
        "y_pred": np.arange(8, dtype=float) + 0.5,
        "regime": ["Normal"] * 8,
    })
    df_b = pd.DataFrame({
        "timestamp": ts[2:],
        "y_true": np.arange(8, dtype=float) + 2,
        "y_pred": np.arange(8, dtype=float) + 2.5,
        "regime": ["Normal"] * 8,
    })
    _write_parquet(tmp_path, "a.parquet", df_a)
    _write_parquet(tmp_path, "b.parquet", df_b)

    spec = [
        ("model_a", "a.parquet"),
        ("model_b", "b.parquet"),
    ]
    monkeypatch.setattr("scripts.build_statistical_appendix.PRED_DIR", tmp_path)
    out = load_predictions(spec)

    assert set(out.keys()) == {"model_a", "model_b"}
    # Intersection is ts[2:8] = 6 rows.
    assert len(out["model_a"]) == 6
    assert len(out["model_b"]) == 6
    assert out["model_a"].index.equals(out["model_b"].index)


def test_models_constant_has_12_entries():
    assert len(MODELS) == 12
    # Each entry is (display_name, parquet_filename).
    for entry in MODELS:
        assert len(entry) == 2
        assert entry[1].endswith(".parquet")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_statistical_appendix.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.build_statistical_appendix'` (or import error on `load_predictions` / `MODELS`).

- [ ] **Step 3: Implement skeleton + loader**

Create `scripts/build_statistical_appendix.py`:

```python
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

    # Intersection of all τ indices.
    common = None
    for df in dfs.values():
        idx = df.index
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) == 0:
        raise ValueError("No τ overlap across model parquets — aborting.")

    return {name: df.loc[common].sort_index() for name, df in dfs.items()}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_statistical_appendix.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_statistical_appendix.py tests/test_statistical_appendix.py
git commit -m "feat(stats): load_predictions with intersection-on-tau"
```

---

## Task 2: `compute_ci_table` (bootstrap CIs per model × regime)

**Files:**
- Modify: `scripts/build_statistical_appendix.py`
- Modify: `tests/test_statistical_appendix.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_statistical_appendix.py`:

```python
def _two_model_synthetic(n=240):
    """24h × 10d of synthetic predictions with regime labels."""
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    y_true = np.sin(np.arange(n) * 2 * np.pi / 24) + 10
    rng = np.random.default_rng(0)
    return {
        "good": pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_true + rng.normal(scale=0.1, size=n),
            "regime": ["Normal"] * (n // 2) + ["Ramadan"] * (n // 2),
        }, index=ts),
        "bad": pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_true + rng.normal(scale=1.0, size=n),
            "regime": ["Normal"] * (n // 2) + ["Ramadan"] * (n // 2),
        }, index=ts),
    }


def test_compute_ci_table_shape_and_ordering():
    from scripts.build_statistical_appendix import compute_ci_table
    preds = _two_model_synthetic()
    df = compute_ci_table(preds, regimes=["aggregate", "Normal", "Ramadan"])
    # Long format: model × regime rows
    assert list(df.columns) == ["model", "regime", "mae", "ci_lo", "ci_hi"]
    assert len(df) == 2 * 3
    # bad's MAE should be greater than good's MAE everywhere
    g_agg = df[(df.model == "good") & (df.regime == "aggregate")].mae.iloc[0]
    b_agg = df[(df.model == "bad")  & (df.regime == "aggregate")].mae.iloc[0]
    assert b_agg > g_agg
    # CI brackets the point estimate
    row = df.iloc[0]
    assert row.ci_lo <= row.mae <= row.ci_hi


def test_compute_ci_table_handles_empty_regime():
    from scripts.build_statistical_appendix import compute_ci_table
    preds = _two_model_synthetic()
    # No Heatwave rows exist in synthetic
    df = compute_ci_table(preds, regimes=["aggregate", "Heatwave"])
    # aggregate rows present, Heatwave rows present but NaN
    heat = df[df.regime == "Heatwave"]
    assert len(heat) == 2
    assert heat.mae.isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_statistical_appendix.py::test_compute_ci_table_shape_and_ordering tests/test_statistical_appendix.py::test_compute_ci_table_handles_empty_regime -v
```

Expected: `ImportError: cannot import name 'compute_ci_table'`.

- [ ] **Step 3: Implement `compute_ci_table`**

Append to `scripts/build_statistical_appendix.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_statistical_appendix.py -v
```

Expected: all 4 tests pass (the two new ones + the two from Task 1).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_statistical_appendix.py tests/test_statistical_appendix.py
git commit -m "feat(stats): compute_ci_table with block-bootstrap 95% CIs"
```

---

## Task 3: `compute_dm_matrix` (pairwise DM + Holm)

**Files:**
- Modify: `scripts/build_statistical_appendix.py`
- Modify: `tests/test_statistical_appendix.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_statistical_appendix.py`:

```python
def test_compute_dm_matrix_pairs_holm_adjusted():
    from scripts.build_statistical_appendix import compute_dm_matrix
    # Three models with monotonically worse predictions
    ts = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
    y_true = np.sin(np.arange(500) * 2 * np.pi / 24) * 100 + 1000
    rng = np.random.default_rng(0)
    preds = {
        "best":  pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_true + rng.normal(scale=10.0, size=500),
            "regime": ["Normal"] * 500,
        }, index=ts),
        "mid":   pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_true + rng.normal(scale=50.0, size=500),
            "regime": ["Normal"] * 500,
        }, index=ts),
        "worst": pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_true + rng.normal(scale=200.0, size=500),
            "regime": ["Normal"] * 500,
        }, index=ts),
    }
    df = compute_dm_matrix(preds, regime="aggregate")
    # 3 models -> 3 pairs (i < j on a list-index basis)
    assert len(df) == 3
    assert list(df.columns) == ["model_i", "model_j", "dm_stat", "p_raw", "p_holm"]
    # holm-adjusted p must be >= raw p
    assert (df.p_holm >= df.p_raw - 1e-12).all()
    # best vs worst should be highly significant
    bw = df[(df.model_i == "best") & (df.model_j == "worst")]
    assert len(bw) == 1
    assert bw.p_holm.iloc[0] < 0.01


def test_compute_dm_matrix_skips_empty_regime():
    from scripts.build_statistical_appendix import compute_dm_matrix
    preds = _two_model_synthetic()
    df = compute_dm_matrix(preds, regime="Heatwave")  # no Heatwave rows
    assert len(df) == 1
    assert pd.isna(df.dm_stat.iloc[0])
    assert pd.isna(df.p_raw.iloc[0])
    assert pd.isna(df.p_holm.iloc[0])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_statistical_appendix.py::test_compute_dm_matrix_pairs_holm_adjusted tests/test_statistical_appendix.py::test_compute_dm_matrix_skips_empty_regime -v
```

Expected: `ImportError: cannot import name 'compute_dm_matrix'`.

- [ ] **Step 3: Implement `compute_dm_matrix`**

Append to `scripts/build_statistical_appendix.py`:

```python
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

    # Holm on the non-NaN raw p-values only.
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_statistical_appendix.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_statistical_appendix.py tests/test_statistical_appendix.py
git commit -m "feat(stats): compute_dm_matrix pairwise + Holm-Bonferroni per regime"
```

---

## Task 4: `render_markdown` (the doc)

**Files:**
- Modify: `scripts/build_statistical_appendix.py`
- Modify: `tests/test_statistical_appendix.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_statistical_appendix.py`:

```python
def test_render_markdown_includes_required_sections():
    from scripts.build_statistical_appendix import (
        compute_ci_table, compute_dm_matrix, render_markdown,
    )
    preds = _two_model_synthetic()
    ci = compute_ci_table(preds, regimes=["aggregate", "Normal", "Ramadan"])
    dm_by_regime = {
        r: compute_dm_matrix(preds, r) for r in ["aggregate", "Normal", "Ramadan"]
    }
    md = render_markdown(ci, dm_by_regime, n_tau=240)
    # Required headings
    assert "# Statistical Appendix" in md
    assert "## Bootstrap MAE confidence intervals" in md
    assert "## Pairwise Diebold-Mariano tests" in md
    # Each regime gets a sub-section
    for r in ["aggregate", "Normal", "Ramadan"]:
        assert f"### DM matrix — {r}" in md
    # Model names appear
    assert "good" in md and "bad" in md
    # Sig markers present somewhere (either *** or ns)
    assert ("***" in md) or ("**" in md) or ("ns" in md)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_statistical_appendix.py::test_render_markdown_includes_required_sections -v
```

Expected: `ImportError: cannot import name 'render_markdown'`.

- [ ] **Step 3: Implement `render_markdown`**

Append to `scripts/build_statistical_appendix.py`:

```python
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
    """Render the full appendix as markdown.

    Sections:
      1. Header + setup notes
      2. Bootstrap MAE CI table (regimes as columns, models as rows)
      3. One DM matrix sub-section per regime (lower-triangular)
    """
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

    # CI table — wide format (model × regime)
    lines.append("## Bootstrap MAE confidence intervals")
    lines.append("")
    regimes = list(dm_by_regime.keys())
    header = "| Model | " + " | ".join(regimes) + " |"
    sep = "|" + "---|" * (len(regimes) + 1)
    lines.append(header)
    lines.append(sep)
    pivot = ci_df.pivot(index="model", columns="regime")
    # Preserve original model order
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

    # DM matrices — one per regime, lower-triangular
    lines.append("## Pairwise Diebold-Mariano tests")
    lines.append("")
    # Order of models: preserve insertion order from ci_df
    model_order = list(ci_df["model"].unique())
    for regime in regimes:
        lines.append(f"### DM matrix — {regime}")
        lines.append("")
        dm = dm_by_regime[regime]
        # Build a {(i,j): (stat, p_holm)} dict
        cells: dict[tuple[str, str], tuple[float, float]] = {}
        for _, row in dm.iterrows():
            cells[(row["model_i"], row["model_j"])] = (row["dm_stat"], row["p_holm"])
        # Header: column model names
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_statistical_appendix.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_statistical_appendix.py tests/test_statistical_appendix.py
git commit -m "feat(stats): render_markdown produces CI table + DM matrices per regime"
```

---

## Task 5: `main()` + run on real data

**Files:**
- Modify: `scripts/build_statistical_appendix.py`

- [ ] **Step 1: Implement `main()`**

Append to `scripts/build_statistical_appendix.py`:

```python
def main() -> None:
    print("[1/4] Loading 12 prediction parquets ...")
    preds = load_predictions(MODELS)
    n_tau = len(next(iter(preds.values())))
    print(f"      intersection-on-τ: {n_tau:,} rows")

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
```

- [ ] **Step 2: Run the script end-to-end**

```bash
.venv/Scripts/python.exe scripts/build_statistical_appendix.py
```

Expected:
- Prints 4 stages.
- Wall-clock 10-15 min.
- Produces `docs/statistical_appendix.md` (non-empty).
- Produces 5 CSVs in `data/statistical_appendix/`.

If a parquet is missing, the script raises `FileNotFoundError` with the
specific filename — fix by running the missing model first (per
spec §Models in scope).

- [ ] **Step 3: Sanity check the doc**

```bash
.venv/Scripts/python.exe -c "
from pathlib import Path
md = Path('docs/statistical_appendix.md').read_text(encoding='utf-8')
print(f'length: {len(md):,} chars')
print(f'has CI table heading: {\"## Bootstrap MAE confidence intervals\" in md}')
print(f'has DM section: {\"## Pairwise Diebold-Mariano tests\" in md}')
for r in ['aggregate','Normal','Ramadan','Heatwave']:
    print(f'has DM matrix for {r}: {f\"### DM matrix — {r}\" in md}')
# Spot-check LGBM-hijri aggregate CI from CSV
import pandas as pd
ci = pd.read_csv('data/statistical_appendix/ci_table.csv')
row = ci[(ci.model == 'lgbm-hijri') & (ci.regime == 'aggregate')].iloc[0]
print(f'lgbm-hijri aggregate: MAE {row.mae:.1f} [{row.ci_lo:.1f}, {row.ci_hi:.1f}]')
"
```

Expected: all booleans `True`; LGBM-hijri agg MAE ≈ 979 with CI bracketing it.

- [ ] **Step 4: Commit outputs**

```bash
git add scripts/build_statistical_appendix.py docs/statistical_appendix.md data/statistical_appendix/
git commit -m "feat(stats): generate statistical appendix (CIs + 4 DM matrices)"
```

---

## Task 6: Smoke test + final pytest

**Files:**
- Modify: `tests/test_smoke_pipeline.py`

- [ ] **Step 1: Add appendix existence checks**

Append to `tests/test_smoke_pipeline.py`:

```python
# Plan 7 sub-task: statistical appendix.
STATS_DIR = ROOT / "data" / "statistical_appendix"
STATS_DOC = ROOT / "docs" / "statistical_appendix.md"
STATS_CSVS = [
    "ci_table.csv",
    "dm_aggregate.csv",
    "dm_Normal.csv",
    "dm_Ramadan.csv",
    "dm_Heatwave.csv",
]


def test_statistical_appendix_doc_exists():
    assert STATS_DOC.exists(), (
        f"Missing {STATS_DOC}. Re-run scripts/build_statistical_appendix.py."
    )
    txt = STATS_DOC.read_text(encoding="utf-8")
    assert "# Statistical Appendix" in txt
    assert "## Bootstrap MAE confidence intervals" in txt
    assert "## Pairwise Diebold-Mariano tests" in txt


@pytest.mark.parametrize("name", STATS_CSVS)
def test_statistical_appendix_csv_exists(name):
    p = STATS_DIR / name
    assert p.exists(), (
        f"Missing {p}. Re-run scripts/build_statistical_appendix.py."
    )
    df = pd.read_csv(p)
    assert len(df) > 0
```

- [ ] **Step 2: Run smoke tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_smoke_pipeline.py -v -k "statistical_appendix"
```

Expected: 6 tests pass (1 doc + 5 CSVs).

- [ ] **Step 3: Run full pytest**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: all green (prior 127 PatchTSMixer/Plan-5 tests + 7 new statistical_appendix unit tests + 6 new smoke tests).

- [ ] **Step 4: Commit smoke tests**

```bash
git add tests/test_smoke_pipeline.py
git commit -m "test(stats): smoke checks for statistical_appendix doc + 5 CSVs"
```

---

## Self-check before merging

- All tests green (pytest -q)
- `docs/statistical_appendix.md` exists and contains both major sections
- `data/statistical_appendix/` contains 5 CSVs
- LGBM-hijri aggregate MAE in CI table matches the value already in `tsfm_zero_shot_baseline.md` (≈979)
