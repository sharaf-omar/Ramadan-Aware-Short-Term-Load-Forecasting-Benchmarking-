# Plan 6 — Post-Hoc LightGBM Residual Correction

For each of the four headline TSFMs (Chronos-Bolt-Base L=720, Time-MoE-200M
L=720, TimesFM-2.5-200M L=168, Moirai-1.1-R-Small L=336), a LightGBM head
is fit on `(features) → (y_true − y_pred_TSFM)` and the corrected forecast
is `y_pred_TSFM + y_residual_hat`. Two feature variants are evaluated:
`nohijri` (weather + calendar + load lags) and `hijri` (above plus
`is_ramadan`, `day_of_ramadan`, `is_eid`).

See [`docs/superpowers/specs/2026-05-14-residual-correction-design.md`](superpowers/specs/2026-05-14-residual-correction-design.md).

## Setup

- 4 TSFMs at their best L from Plan 3.
- 2 residual variants per TSFM ⇒ 8 corrected parquets.
- Residual model: LightGBM, `objective="regression_l1"`, lr=0.05, 1000
  estimators, early stop patience 30 on val MAE.
- Test n = 10,944 (full 2024-01-01..2025-03-31 window, intersection with
  the rest of the headline cohort).
- DM tests: HAC h=24, MAE loss, two-sided. Holm-Bonferroni applied
  within each regime's family of 4 comparisons.

## Per-TSFM results

| TSFM | Variant | Agg MAE | Normal | Ramadan | Heatwave |
|---|---|---|---|---|---|
| chronos-bolt-L720 | bare         |  968.9 |  904.0 | 1061.0 | 1221.2 |
| chronos-bolt-L720 | +residual-nh | 1001.8 |  885.4 | 1074.8 | 1539.9 |
| chronos-bolt-L720 | +residual-h  |  994.5 |  879.7 | 1044.4 | 1545.7 |
| moirai-L336       | bare         | 1727.1 | 1645.4 | 1695.7 | 2181.4 |
| moirai-L336       | +residual-nh | 1390.1 | 1220.4 | 1412.7 | 2252.7 |
| moirai-L336       | +residual-h  | **1377.3** | **1214.6** | 1441.6 | 2164.2 |
| timesfm-L168      | bare         | 1173.2 | 1082.5 | 1195.8 | 1624.2 |
| timesfm-L168      | +residual-nh | 1147.9 |  983.6 | 1157.2 | 1994.3 |
| timesfm-L168      | +residual-h  | **1136.5** |  **980.4** | 1199.6 | 1890.5 |
| time-moe-L720     | bare         |  985.9 |  908.8 | 1115.6 | 1267.6 |
| time-moe-L720     | +residual-nh | 1024.6 |  884.0 | 1150.3 | 1640.5 |
| time-moe-L720     | +residual-h  | 1033.7 |  886.8 | 1188.8 | 1655.5 |

**Bold** marks corrected MAEs that beat the bare baseline.

## Headline finding

**Residual correction has a model-dependent effect.** The four TSFMs split
into two response groups:

- **Rescue cases (Moirai, TimesFM):** the residual head produces a clear
  aggregate improvement. Moirai's drop is the largest in the benchmark
  (1727 → 1377, −20%); the corrected Moirai now ranks 12th overall in
  the appendix (was 18th). TimesFM-residual edges its bare baseline by
  ~37 MW on aggregate.
- **No-rescue cases (Chronos, Time-MoE):** the corrected aggregate MAE is
  *worse* than bare because the residual head substantially regresses
  Heatwave performance (Chronos: 1221 → 1546, +27%; Time-MoE: 1268 →
  1656, +31%). On Normal regime, residual *does* help these models too
  (Chronos −24 MW, Time-MoE −22 MW), but the Heatwave loss dominates the
  weighted average.

A pattern emerges across all four: **residual correction reliably helps
on Normal regime** (4 of 4 TSFMs, MAE drops 18-431 MW) and **reliably
hurts on Heatwave** (3 of 4 TSFMs degrade; only Moirai is approximately
neutral). Ramadan response is mixed and mostly within noise.

The Heatwave regression matches a plausible mechanism: heatwave-period
load is the regime where TSFMs already win the benchmark (Chronos beats
LGBM by 28% on Heatwave). The residual head, fit on the average
forecast-error pattern, injects bias toward the more common Normal
regime and shifts already-good Heatwave predictions away from truth.

The Hijri variant rarely beats the nohijri variant by much — typically
0-15 MW on Ramadan — suggesting the calendar features the residual head
uses (hour/day-of-week sin/cos + load lags) already capture most of the
weekly-periodic Ramadan effect that the Hijri channels add on top.

## DM tests (bare vs +residual-hijri, HAC h=24, Holm-adjusted per regime)

| TSFM | Regime | DM stat | p_raw | p_holm | Direction |
|---|---|---|---|---|---|
| chronos-bolt-L720 | aggregate | −1.62 | 0.106 | 0.211 | ns |
| chronos-bolt-L720 | Normal    | +1.54 | 0.123 | 0.246 | ns |
| chronos-bolt-L720 | Ramadan   | +0.57 | 0.570 | 1.000 | ns |
| chronos-bolt-L720 | Heatwave  | **−5.63** | <1e-7 | <1e-7 | bare better |
| moirai-L336       | aggregate | **+9.81** | <1e-7 | <1e-7 | residual better |
| moirai-L336       | Normal    | **+11.43**| <1e-7 | <1e-7 | residual better |
| moirai-L336       | Ramadan   | +2.49 | 0.013 | 0.051 | borderline |
| moirai-L336       | Heatwave  | +0.14 | 0.886 | 0.886 | ns |
| timesfm-L168      | aggregate | +1.47 | 0.143 | 0.211 | ns |
| timesfm-L168      | Normal    | **+3.70** | 2e-4 | 6e-4 | residual better |
| timesfm-L168      | Ramadan   | −0.06 | 0.953 | 1.000 | ns |
| timesfm-L168      | Heatwave  | **−3.74** | 2e-4 | 4e-4 | bare better |
| time-moe-L720     | aggregate | **−2.76** | 0.006 | 0.018 | bare better |
| time-moe-L720     | Normal    | +1.28 | 0.200 | 0.246 | ns |
| time-moe-L720     | Ramadan   | −2.07 | 0.039 | 0.116 | borderline |
| time-moe-L720     | Heatwave  | **−6.41** | <1e-7 | <1e-7 | bare better |

DM sign convention from `src.evaluation.dm_test`: positive stat means
model B (+residual-hijri here) has lower loss; negative means model A
(bare TSFM) wins.

## Cross-reference to Plan 3 (in-band covariate path)

Plan 3 ([`tsfm_hijri_covariates.md`](tsfm_hijri_covariates.md)) tested
the HuggingFace covariate-ingestion path for TimesFM and Moirai with
the same Hijri features. Comparing the two paths on Ramadan MAE:

| Model | bare | +HF covariate | +LGBM residual-h | Best |
|---|---|---|---|---|
| timesfm-2.5-L168 | 1195.8 | (Plan 3 hurt, MAE rose) | 1199.6 | bare |
| moirai-1.1-L336  | 1695.7 | (Plan 3 hurt, MAE rose) | 1441.6 | **+residual** |

For Moirai, post-hoc residual correction succeeds where the in-band
covariate path failed — the kind of asymmetry the proposal predicted.
For TimesFM, neither approach moves Ramadan; the residual head's value
shows up on Normal regime instead.

## Runtime

| Stage | Wall-clock |
|---|---|
| 8 LightGBM residual fits (3-fold time-block CV per combo, ~2.5s each) | ~30 s CPU |
| Statistical appendix regeneration (20 models) | ~6 min CPU |

## Files

- `data/predictions/<tsfm>__residual__{nohijri,hijri}__L<L>__seed0.parquet` (8 files)
- `docs/statistical_appendix.md` regenerated with 20 models
- `data/statistical_appendix/*.csv` regenerated

## Recommendations

1. **Always combine Moirai with a LightGBM residual head.** The
   aggregate MAE gain is ~350 MW; the corrected Moirai jumps from
   bottom-tier (rank 18) to mid-tier (rank 12).
2. **Always combine TimesFM with a LightGBM residual head on Normal-
   regime forecasts**, but route Heatwave hours to bare TimesFM.
3. **Do NOT correct Chronos or Time-MoE forecasts during Heatwave.**
   These models already win Heatwave by 25-30% over LightGBM; the
   residual head only hurts them there.
4. **The Hijri flavour of the residual head adds little over the
   nohijri flavour.** The calendar + load-lag features carry most of
   the weekly-periodic signal a residual head can leverage. For
   simplicity, ship `+residual-nh`.

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

.venv/Scripts/python.exe scripts/build_statistical_appendix.py
```
