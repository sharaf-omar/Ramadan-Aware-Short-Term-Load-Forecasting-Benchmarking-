# Statistical Appendix

Canonical statistical-rigor artifact for the benchmark. Block-bootstrap 95% CIs around MAE for every headline model × regime, plus full pairwise Diebold-Mariano matrices (Holm-Bonferroni adjusted within each regime).

**Intersection set size (n=τ rows across all models):** 10,944

**Bootstrap:** stationary block bootstrap (Politis & Romano 1994), block_size=24h, 1000 resamples, alpha=0.05, seed=0.

**DM test:** MAE loss, HAC h=24, two-sided. Holm-Bonferroni applied within each regime's pairwise family. Significance markers: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` otherwise. DM stat sign convention (from `src.evaluation.dm_test`): positive means model_j (column) has lower loss; negative means model_i (row).

## Bootstrap MAE confidence intervals

| Model | aggregate | Normal | Ramadan | Heatwave |
|---|---|---|---|---|
| lgbm-nohijri | 1003.3 [907.5, 1113.3] | 889.2 [786.5, 995.2] | 897.7 [722.2, 1106.5] | 1693.9 [1433.2, 1992.4] |
| lgbm-hijri | 979.0 [890.2, 1079.9] | 873.5 [784.1, 967.7] | 799.9 [681.9, 936.8] | 1693.0 [1441.2, 1989.1] |
| chronos-bolt-L720 | 968.9 [868.8, 1097.9] | 904.0 [790.9, 1026.3] | 1061.0 [838.7, 1336.8] | 1221.2 [915.5, 1596.6] |
| timesfm-2.5-L168 | 1173.2 [1057.2, 1313.2] | 1082.5 [964.4, 1217.3] | 1195.8 [972.1, 1480.5] | 1624.2 [1201.9, 2183.0] |
| moirai-1.1-small-L336 | 1727.1 [1620.3, 1853.5] | 1645.4 [1515.9, 1773.3] | 1695.7 [1450.8, 1968.9] | 2181.4 [1844.3, 2554.3] |
| time-moe-200m-L720 | 985.9 [878.2, 1119.7] | 908.8 [795.8, 1029.0] | 1115.6 [855.4, 1447.9] | 1267.6 [967.2, 1601.8] |
| mstl_ets-nohijri | 1593.3 [1448.6, 1759.8] | 1370.6 [1231.9, 1528.6] | 1842.7 [1578.4, 2192.2] | 2522.3 [2194.7, 2934.4] |
| mstl_ets-hijri | 1527.5 [1379.5, 1692.9] | 1371.9 [1232.5, 1530.4] | 1327.0 [1043.8, 1711.8] | 2522.3 [2194.7, 2934.4] |
| sarimax-nohijri | 2525.8 [2373.9, 2708.9] | 2477.9 [2295.2, 2665.9] | 2255.5 [1786.6, 2722.2] | 3024.1 [2636.5, 3395.8] |
| sarimax-hijri | 2485.9 [2344.8, 2648.8] | 2422.8 [2255.6, 2598.1] | 2250.6 [1801.3, 2703.6] | 3030.9 [2639.6, 3404.6] |
| patchtsmixer-nohijri-L168 | 1550.6 [1489.2, 1625.1] | 1492.0 [1428.2, 1555.3] | 1492.1 [1332.5, 1697.2] | 1909.4 [1736.5, 2126.3] |
| patchtsmixer-hijri-L168 | 1552.7 [1491.9, 1620.8] | 1496.2 [1437.2, 1555.9] | 1551.6 [1400.5, 1736.9] | 1847.4 [1667.3, 2085.4] |

## Pairwise Diebold-Mariano tests

### DM matrix — aggregate

| row \ col | lgbm-nohijri | lgbm-hijri | chronos-bolt-L720 | timesfm-2.5-L168 | moirai-1.1-small-L336 | time-moe-200m-L720 | mstl_ets-nohijri | mstl_ets-hijri | sarimax-nohijri | sarimax-hijri | patchtsmixer-nohijri-L168 | patchtsmixer-hijri-L168 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm-nohijri |  | +1.6 ns | +0.9 ns | -4.0 ** | -18.9 *** | +0.5 ns | -13.6 *** | -12.2 *** | -20.9 *** | -21.0 *** | -19.0 *** | -17.5 *** |
| lgbm-hijri |  |  | +0.2 ns | -4.1 *** | -18.2 *** | -0.2 ns | -12.7 *** | -11.4 *** | -20.3 *** | -21.0 *** | -19.7 *** | -19.3 *** |
| chronos-bolt-L720 |  |  |  | -6.7 *** | -17.7 *** | -0.7 ns | -15.3 *** | -13.6 *** | -22.2 *** | -21.9 *** | -16.8 *** | -15.7 *** |
| timesfm-2.5-L168 |  |  |  |  | -13.3 *** | +5.5 *** | -8.8 *** | -7.4 *** | -20.9 *** | -20.3 *** | -9.0 *** | -8.6 *** |
| moirai-1.1-small-L336 |  |  |  |  |  | +16.6 *** | +2.4 ns | +3.5 ** | -11.5 *** | -11.2 *** | +4.1 *** | +3.9 ** |
| time-moe-200m-L720 |  |  |  |  |  |  | -15.0 *** | -13.3 *** | -22.2 *** | -21.9 *** | -16.4 *** | -15.4 *** |
| mstl_ets-nohijri |  |  |  |  |  |  |  | +6.4 *** | -12.0 *** | -11.8 *** | +0.9 ns | +0.8 ns |
| mstl_ets-hijri |  |  |  |  |  |  |  |  | -12.9 *** | -12.7 *** | -0.5 ns | -0.5 ns |
| sarimax-nohijri |  |  |  |  |  |  |  |  |  | +2.4 ns | +12.9 *** | +12.6 *** |
| sarimax-hijri |  |  |  |  |  |  |  |  |  |  | +13.0 *** | +12.8 *** |
| patchtsmixer-nohijri-L168 |  |  |  |  |  |  |  |  |  |  |  | -0.2 ns |
| patchtsmixer-hijri-L168 |  |  |  |  |  |  |  |  |  |  |  |  |

### DM matrix — Normal

| row \ col | lgbm-nohijri | lgbm-hijri | chronos-bolt-L720 | timesfm-2.5-L168 | moirai-1.1-small-L336 | time-moe-200m-L720 | mstl_ets-nohijri | mstl_ets-hijri | sarimax-nohijri | sarimax-hijri | patchtsmixer-nohijri-L168 | patchtsmixer-hijri-L168 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm-nohijri |  | +0.9 ns | -0.4 ns | -5.0 *** | -18.6 *** | -0.6 ns | -10.2 *** | -10.2 *** | -19.2 *** | -19.1 *** | -19.2 *** | -17.6 *** |
| lgbm-hijri |  |  | -0.7 ns | -4.6 *** | -17.4 *** | -0.8 ns | -9.5 *** | -9.5 *** | -18.4 *** | -19.1 *** | -20.4 *** | -20.0 *** |
| chronos-bolt-L720 |  |  |  | -5.9 *** | -15.2 *** | -0.2 ns | -10.2 *** | -10.2 *** | -19.4 *** | -19.0 *** | -15.1 *** | -13.9 *** |
| timesfm-2.5-L168 |  |  |  |  | -12.3 *** | +5.5 *** | -5.9 *** | -5.9 *** | -19.0 *** | -18.1 *** | -9.5 *** | -8.8 *** |
| moirai-1.1-small-L336 |  |  |  |  |  | +15.4 *** | +4.3 *** | +4.3 *** | -10.5 *** | -10.1 *** | +3.3 * | +3.0 * |
| time-moe-200m-L720 |  |  |  |  |  |  | -10.6 *** | -10.6 *** | -19.6 *** | -19.2 *** | -15.0 *** | -14.0 *** |
| mstl_ets-nohijri |  |  |  |  |  |  |  | -1.4 ns | -12.5 *** | -12.3 *** | -2.3 ns | -2.2 ns |
| mstl_ets-hijri |  |  |  |  |  |  |  |  | -12.5 *** | -12.3 *** | -2.2 ns | -2.2 ns |
| sarimax-nohijri |  |  |  |  |  |  |  |  |  | +2.4 ns | +11.3 *** | +10.9 *** |
| sarimax-hijri |  |  |  |  |  |  |  |  |  |  | +11.3 *** | +11.0 *** |
| patchtsmixer-nohijri-L168 |  |  |  |  |  |  |  |  |  |  |  | -0.4 ns |
| patchtsmixer-hijri-L168 |  |  |  |  |  |  |  |  |  |  |  |  |

### DM matrix — Ramadan

| row \ col | lgbm-nohijri | lgbm-hijri | chronos-bolt-L720 | timesfm-2.5-L168 | moirai-1.1-small-L336 | time-moe-200m-L720 | mstl_ets-nohijri | mstl_ets-hijri | sarimax-nohijri | sarimax-hijri | patchtsmixer-nohijri-L168 | patchtsmixer-hijri-L168 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm-nohijri |  | +2.1 ns | -3.1 * | -5.2 *** | -8.2 *** | -3.1 * | -10.8 *** | -4.7 *** | -6.7 *** | -6.9 *** | -13.3 *** | -13.3 *** |
| lgbm-hijri |  |  | -3.2 * | -4.7 *** | -9.2 *** | -3.0 * | -8.6 *** | -4.2 *** | -6.9 *** | -7.2 *** | -11.0 *** | -13.7 *** |
| chronos-bolt-L720 |  |  |  | -3.1 * | -6.4 *** | -1.1 ns | -11.1 *** | -3.8 ** | -6.1 *** | -6.2 *** | -7.5 *** | -7.5 *** |
| timesfm-2.5-L168 |  |  |  |  | -5.7 *** | +1.5 ns | -8.4 *** | -1.7 ns | -6.0 *** | -6.2 *** | -4.7 *** | -4.8 *** |
| moirai-1.1-small-L336 |  |  |  |  |  | +5.3 *** | -1.2 ns | +2.9 ns | -2.9 ns | -3.0 * | +2.2 ns | +1.4 ns |
| time-moe-200m-L720 |  |  |  |  |  |  | -11.9 *** | -3.5 * | -5.9 *** | -6.1 *** | -5.1 *** | -5.1 *** |
| mstl_ets-nohijri |  |  |  |  |  |  |  | +10.5 *** | -1.9 ns | -1.9 ns | +4.0 ** | +2.9 ns |
| mstl_ets-hijri |  |  |  |  |  |  |  |  | -4.3 *** | -4.4 *** | -1.8 ns | -2.2 ns |
| sarimax-nohijri |  |  |  |  |  |  |  |  |  | +0.2 ns | +3.6 ** | +3.2 * |
| sarimax-hijri |  |  |  |  |  |  |  |  |  |  | +3.7 ** | +3.3 * |
| patchtsmixer-nohijri-L168 |  |  |  |  |  |  |  |  |  |  |  | -1.6 ns |
| patchtsmixer-hijri-L168 |  |  |  |  |  |  |  |  |  |  |  |  |

### DM matrix — Heatwave

| row \ col | lgbm-nohijri | lgbm-hijri | chronos-bolt-L720 | timesfm-2.5-L168 | moirai-1.1-small-L336 | time-moe-200m-L720 | mstl_ets-nohijri | mstl_ets-hijri | sarimax-nohijri | sarimax-hijri | patchtsmixer-nohijri-L168 | patchtsmixer-hijri-L168 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm-nohijri |  | +0.1 ns | +3.2 * | +0.3 ns | -3.4 * | +2.8 ns | -5.5 *** | -5.5 *** | -6.3 *** | -6.3 *** | -2.0 ns | -1.4 ns |
| lgbm-hijri |  |  | +3.2 * | +0.3 ns | -3.4 * | +2.8 ns | -5.5 *** | -5.5 *** | -6.3 *** | -6.2 *** | -2.0 ns | -1.4 ns |
| chronos-bolt-L720 |  |  |  | -2.9 ns | -6.8 *** | -0.6 ns | -11.7 *** | -11.7 *** | -9.8 *** | -9.8 *** | -5.5 *** | -5.1 *** |
| timesfm-2.5-L168 |  |  |  |  | -3.5 * | +2.2 ns | -4.6 *** | -4.6 *** | -7.3 *** | -7.4 *** | -1.6 ns | -1.2 ns |
| moirai-1.1-small-L336 |  |  |  |  |  | +5.5 *** | -1.9 ns | -1.9 ns | -4.3 *** | -4.3 *** | +1.7 ns | +2.1 ns |
| time-moe-200m-L720 |  |  |  |  |  |  | -9.2 *** | -9.2 *** | -9.7 *** | -9.7 *** | -5.5 *** | -4.9 *** |
| mstl_ets-nohijri |  |  |  |  |  |  |  | +0.0 ns | -2.4 ns | -2.4 ns | +4.0 ** | +4.4 *** |
| mstl_ets-hijri |  |  |  |  |  |  |  |  | -2.4 ns | -2.4 ns | +4.0 ** | +4.4 *** |
| sarimax-nohijri |  |  |  |  |  |  |  |  |  | -0.9 ns | +5.7 *** | +6.0 *** |
| sarimax-hijri |  |  |  |  |  |  |  |  |  |  | +5.7 *** | +6.1 *** |
| patchtsmixer-nohijri-L168 |  |  |  |  |  |  |  |  |  |  |  | +2.8 ns |
| patchtsmixer-hijri-L168 |  |  |  |  |  |  |  |  |  |  |  |  |
