# Deep Analysis: Per-Horizon and Diurnal Error Decomposition

Two analyses on top of the prediction parquets, using the saved `y_block` columns (5 block-forecaster models) and per-tau predictions (all 12 models).

**Test rows per model:** 10,944

## 1. Per-horizon MAE decomposition

For each block-forecaster model, MAE at each horizon h in {1..24} from the issuance time t = tau - 24. Horizon 24 is the canonical y_pred reported in the headline tables (block[23]).

**MAE at selected horizons** (rest in `data/analysis/horizon_mae.csv`):

| Model | h=1 | h=6 | h=12 | h=18 | h=24 | h=24/h=1 ratio |
|---|---|---|---|---|---|---|
| chronos-bolt-L720 | 369 | 671 | 808 | 893 | 969 | 2.62x |
| timesfm-2.5-L168 | 413 | 865 | 1038 | 1112 | 1173 | 2.84x |
| moirai-1.1-small-L336 | 871 | 1377 | 1602 | 1627 | 1727 | 1.98x |
| time-moe-200m-L720 | 262 | 641 | 812 | 908 | 986 | 3.76x |
| patchtsmixer-nohijri-L168 | 1258 | 1351 | 1407 | 1493 | 1551 | 1.23x |
| patchtsmixer-hijri-L168 | 1270 | 1351 | 1428 | 1471 | 1553 | 1.22x |

A ratio close to 1.00 means the model's forecast quality is approximately constant across the 24-hour horizon — a sign of a good direct-prediction architecture. A ratio >> 1.00 means error compounds at long horizon (typical of autoregressive models).

## 2. Diurnal MAE (by hour-of-day of tau, UTC)

MAE bucketed by hour-of-day. Hours are UTC; local Turkish time = UTC + 3. Local morning peak is UTC 04-06; local evening peak is UTC 16-19.

**Aggregate-regime MAE by 4-hour UTC bin** (full hourly in `data/analysis/diurnal_mae.csv`):

| Model | 00-03 (UTC) | 04-07 | 08-11 | 12-15 | 16-19 | 20-23 |
|---|---|---|---|---|---|---|
| lgbm-nohijri | 676 | 1060 | 1381 | 1356 | 921 | 625 |
| lgbm-hijri | 661 | 1023 | 1353 | 1321 | 905 | 611 |
| chronos-bolt-L720 | 652 | 1050 | 1325 | 1197 | 900 | 689 |
| timesfm-2.5-L168 | 870 | 1366 | 1549 | 1381 | 1022 | 851 |
| moirai-1.1-small-L336 | 1033 | 2533 | 2339 | 1932 | 1440 | 1087 |
| time-moe-200m-L720 | 669 | 1104 | 1363 | 1205 | 876 | 699 |
| mstl_ets-nohijri | 1465 | 1661 | 1897 | 1869 | 1405 | 1263 |
| mstl_ets-hijri | 1364 | 1530 | 1866 | 1777 | 1389 | 1240 |
| sarimax-nohijri | 1279 | 3476 | 3384 | 3256 | 2107 | 1653 |
| sarimax-hijri | 1262 | 3442 | 3339 | 3206 | 2053 | 1613 |
| patchtsmixer-nohijri-L168 | 827 | 2724 | 1487 | 1248 | 1203 | 1814 |
| patchtsmixer-hijri-L168 | 824 | 2749 | 1485 | 1251 | 1189 | 1818 |

### Peak hours per model

The UTC hour where each model's aggregate MAE is highest, with the value. Highlights where each model fails most.

| Model | Worst hour (UTC) | Worst-hour MAE | Best hour | Best-hour MAE |
|---|---|---|---|---|
| lgbm-nohijri | 11 | 1494 | 23 | 558 |
| lgbm-hijri | 11 | 1468 | 23 | 563 |
| chronos-bolt-L720 | 10 | 1416 | 00 | 620 |
| timesfm-2.5-L168 | 10 | 1681 | 21 | 815 |
| moirai-1.1-small-L336 | 06 | 2966 | 22 | 968 |
| time-moe-200m-L720 | 11 | 1415 | 00 | 651 |
| mstl_ets-nohijri | 11 | 2101 | 00 | 1211 |
| mstl_ets-hijri | 11 | 2074 | 00 | 1162 |
| sarimax-nohijri | 06 | 4099 | 00 | 1157 |
| sarimax-hijri | 06 | 4064 | 00 | 1148 |
| patchtsmixer-nohijri-L168 | 05 | 4464 | 01 | 647 |
| patchtsmixer-hijri-L168 | 05 | 4498 | 01 | 631 |
