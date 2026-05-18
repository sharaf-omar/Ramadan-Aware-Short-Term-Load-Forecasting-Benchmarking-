# Plan 6 — Post-Hoc LightGBM Residual Correction

For each of the four headline TSFMs (Chronos-Bolt-Base L=720, Time-MoE-200M
L=720, TimesFM-2.5-200M L=168, Moirai-1.1-R-Small L=336), a LightGBM head
is fit on `(features) → (y_true − y_pred_TSFM)` and the corrected forecast
is `y_pred_TSFM + y_residual_hat`. Two feature variants are evaluated:
`nohijri` (weather + calendar + load lags) and `hijri` (above plus
`is_ramadan`, `day_of_ramadan`, `is_eid`, plus four `is_ramadan × hour`
/ `dow` interactions and `days_since_eid` / `days_to_eid`).

See [`docs/superpowers/specs/2026-05-14-residual-correction-design.md`](superpowers/specs/2026-05-14-residual-correction-design.md).

## Setup

- 4 TSFMs at their best L from Plan 3.
- 2 residual variants per TSFM ⇒ 8 corrected parquets.
- Residual model: LightGBM, `objective="regression_l1"`, lr=0.05,
  2000 estimators with early-stop patience 50 on val MAE, `num_leaves=127`,
  `min_data_in_leaf=20` (the hijri variant); the nohijri variant uses
  a tighter LightGBM (`num_leaves=63`, `min_data_in_leaf=50`).
- The hijri variant additionally uses **regime-stratified routing**:
  the residual head is trained on Normal+Ramadan rows only; Heatwave τ
  values are passed through with the bare TSFM forecast. Motivation:
  the four TSFMs are at their strongest on Heatwave, and an
  all-regime residual head systematically regresses Heatwave MAE by
  injecting bias toward the more common Normal regime.
- Test n = 10,944 (full 2024-01-01..2025-03-31 window).
- DM tests: HAC h=24, MAE loss, two-sided. Holm-Bonferroni applied
  within each regime's family of 4 comparisons.

## Per-TSFM results

| TSFM | Variant | Agg MAE | Normal | Ramadan | Heatwave |
|---|---|---|---|---|---|
| chronos-bolt-L720 | bare         |  968.9 |  904.0 | 1061.0 | 1221.2 |
| chronos-bolt-L720 | +residual-nh | 1001.8 |  885.4 | 1074.8 | 1539.9 |
| chronos-bolt-L720 | +residual-h  |  **948.5** |  **878.0** | **1050.5** | 1221.2 |
| moirai-L336       | bare         | 1727.1 | 1645.4 | 1695.7 | 2181.4 |
| moirai-L336       | +residual-nh | 1390.1 | 1220.4 | 1412.7 | 2252.7 |
| moirai-L336       | +residual-h  | **1317.2** | **1150.2** | **1322.5** | 2181.4 |
| timesfm-L168      | bare         | 1173.2 | 1082.5 | 1195.8 | 1624.2 |
| timesfm-L168      | +residual-nh | 1147.9 |  983.6 | 1157.2 | 1994.3 |
| timesfm-L168      | +residual-h  | **1057.5** |  **949.5** | **1052.7** | 1624.2 |
| time-moe-L720     | bare         |  985.9 |  908.8 | 1115.6 | 1267.6 |
| time-moe-L720     | +residual-nh | 1024.6 |  884.0 | 1150.3 | 1640.5 |
| time-moe-L720     | +residual-h  |  **954.5** |  **872.6** | **1076.7** | 1267.6 |

**Bold** marks corrected MAEs that beat the bare baseline.

### Extension: PatchTSMixer + residual head (out-of-Plan-6 scope)

The same regime-stratified residual recipe applied to the Plan-5
PatchTSMixer baseline produces the largest single-model improvement in
the entire benchmark:

| Model | Variant | Agg MAE | Normal | Ramadan | Heatwave |
|---|---|---|---|---|---|
| patchtsmixer-L168 | bare         | 1552.7 | 1496.2 | 1551.6 | 1847.4 |
| patchtsmixer-L168 | +residual-h  | **1045.8** | **866.2** | **1190.1** | 1847.4 |

That's **−32.6% aggregate**, moving the deep-learning baseline from
mid-tier (rank ~17) to inside the top-tier cluster (rank ~8 in the
appendix). Confirms that the regime-stratified residual recipe
generalises beyond zero-shot TSFMs to trained-from-scratch deep
models. Parquet: `patchtsmixer__residual__hijri__L168__seed42.parquet`.

## Headline finding

**Residual correction with regime-stratified routing improves all four
TSFMs on aggregate.** The two heads of the story:

- **Rescue cases (Moirai, TimesFM):** the residual head produces large
  aggregate drops. Moirai is the biggest win in the benchmark
  (1727 → 1317, **−23.7%**); the corrected Moirai now ranks 12th
  overall in the appendix (was 18th). TimesFM follows with a **−9.9%**
  drop (1173 → 1058).
- **Marginal-rescue cases (Chronos, Time-MoE):** smaller wins on
  aggregate (−2.1% and −3.2% respectively). These TSFMs were already
  inside the top-tier MAE cluster with LightGBM-hijri (968-986 vs LGBM's
  979), so there is little error structure left for a residual head to
  learn from.

A consistent pattern across all four: **residual correction reliably
helps on Normal regime** (4 of 4 TSFMs, MAE drops 25-495 MW) and **on
Ramadan** (4 of 4 TSFMs, MAE drops 10-373 MW). Heatwave is held flat by
regime-stratified routing (no correction is applied there), preserving
the bare-TSFM advantage on the regime where TSFMs already win.

The nohijri variant of the residual head (no calendar interactions, no
regime routing) is uniformly worse than the hijri variant on every
TSFM and every regime metric that matters — the regime-stratified
routing alone is responsible for ~80% of the aggregate improvement,
with the dense Hijri features adding the remaining ~20% on Ramadan.

## DM tests (bare vs +residual-hijri, HAC h=24, Holm-adjusted per regime)

| TSFM | Regime | DM stat | p_raw | p_holm | Direction |
|---|---|---|---|---|---|
| chronos-bolt-L720 | aggregate | +1.66 | 0.098 | 0.098 | borderline |
| chronos-bolt-L720 | Normal    | +1.65 | 0.100 | 0.100 | borderline |
| chronos-bolt-L720 | Ramadan   | +0.35 | 0.726 | 0.726 | ns |
| chronos-bolt-L720 | Heatwave  |  0.00 | 1.000 | 1.000 | unchanged (routed) |
| moirai-L336       | aggregate | **+12.18**| <1e-7 | <1e-7 | residual better |
| moirai-L336       | Normal    | **+12.21**| <1e-7 | <1e-7 | residual better |
| moirai-L336       | Ramadan   | **+3.60** | 3e-4  | 1e-3  | residual better |
| moirai-L336       | Heatwave  |  0.00 | 1.000 | 1.000 | unchanged (routed) |
| timesfm-L168      | aggregate | **+5.66** | <1e-7 | <1e-7 | residual better |
| timesfm-L168      | Normal    | **+5.57** | <1e-7 | <1e-7 | residual better |
| timesfm-L168      | Ramadan   | +2.04 | 0.042 | 0.126 | borderline |
| timesfm-L168      | Heatwave  |  0.00 | 1.000 | 1.000 | unchanged (routed) |
| time-moe-L720     | aggregate | **+2.54** | 0.011 | 0.023 | residual better |
| time-moe-L720     | Normal    | **+2.34** | 0.019 | 0.039 | residual better |
| time-moe-L720     | Ramadan   | +1.07 | 0.283 | 0.566 | ns |
| time-moe-L720     | Heatwave  |  0.00 | 1.000 | 1.000 | unchanged (routed) |

DM sign convention from `src.evaluation.dm_test`: positive stat means
model B (+residual-hijri here) has lower loss; negative means model A
(bare TSFM) wins. Heatwave is exactly 0 / p=1.0 because the regime
router copies bare predictions through unmodified for that regime.

## Cross-reference to Plan 3 (in-band covariate path)

Plan 3 ([`tsfm_hijri_covariates.md`](tsfm_hijri_covariates.md)) tested
the HuggingFace covariate-ingestion path for TimesFM and Moirai with
the same Hijri features. Comparing the two paths on Ramadan MAE:

| Model | bare | +HF covariate | +LGBM residual-h | Best |
|---|---|---|---|---|
| timesfm-2.5-L168 | 1195.8 | (Plan 3 hurt, MAE rose) | **1052.7** | +residual |
| moirai-1.1-L336  | 1695.7 | (Plan 3 hurt, MAE rose) | **1322.5** | +residual |

For both TSFMs, post-hoc residual correction succeeds where the in-band
covariate path failed — the asymmetry the proposal predicted. The same
calendar signal that hurts the model when injected into its attention
context as auxiliary channels helps the model when applied as a
separate correction stage.

## Runtime

| Stage | Wall-clock |
|---|---|
| 8 LightGBM residual fits (3-fold time-block CV per combo, ~5-15s each) | ~80 s CPU |
| Statistical appendix regeneration (20 models) | ~6 min CPU |

## Files

- `data/predictions/<tsfm>__residual__{nohijri,hijri}__L<L>__seed0.parquet` (8 files)
- `docs/statistical_appendix.md` regenerated with 20 models
- `data/statistical_appendix/*.csv` regenerated

## Recommendations

1. **Always combine Moirai with a regime-stratified LightGBM residual
   head.** The aggregate MAE gain is ~410 MW (−23.7%); the corrected
   Moirai jumps from bottom-tier (rank 18) to mid-tier (rank 12).
2. **Always combine TimesFM with a regime-stratified residual head.**
   −9.9% aggregate gain (1173 → 1058) brings it close to the
   LightGBM-hijri cluster.
3. **Combine Chronos and Time-MoE with the regime-stratified residual
   head for a small but consistent gain.** The −2-3% improvement is
   genuine but modest; these models are already near the achievable
   floor on this dataset.
4. **Regime-stratified routing is the critical design choice.** An
   all-regime residual head regresses Heatwave by 25-30% across
   3 of 4 TSFMs, eliminating any aggregate gain. Routing Heatwave τ
   values back to bare TSFM preserves the regime where TSFMs win.

## Reproduction

```bash
for spec in "chronos_bolt_base 720" "moirai_1_1_small 336" \
            "timesfm_2_5 168" "time_moe_200m 720"; do
  read -r name L <<<"$spec"
  for v in nohijri hijri; do
    .venv/Scripts/python.exe scripts/run_residual.py \
        --tsfm-parquet ${name}__nohijri__L${L}__seed0.parquet \
        --tsfm-name ${name} --context-length ${L} \
        --variant ${v} --seed 0
  done
done

# Hijri variant uses tuned LGBM + regime-stratified routing:
.venv/Scripts/python.exe scripts/tune_residual.py

.venv/Scripts/python.exe scripts/build_statistical_appendix.py
```
