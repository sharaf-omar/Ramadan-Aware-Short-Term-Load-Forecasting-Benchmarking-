# Failure-Mode Analysis

For each of the 12 headline models, the top-10 worst test days (by mean daily MAE) plus the 10 days where every model fails together (mean MAE across all models). Conditions are enriched from the v2 dataset (mean/max temp, Ramadan hour count, dominant regime, weekend flag).

Test window: 2024-01-01 to 2025-03-31. N test days = 456.

## Days where everyone fails (mean MAE across 12 models)

These days are hard for the dataset, not just for one model — useful for sanity-checking 'is the headline model bad, or is the data bad here?'

| Date | Mean MAE | Regime | Mean temp (°C) | Max temp (°C) | Ramadan hours | Weekend |
|---|---|---|---|---|---|---|
| 2024-06-15 | 7708 | Heatwave | 23.9 | 28.1 | 0 | Y |
| 2024-04-10 | 6400 | Normal | 13.8 | 18.2 | 0 | N |
| 2024-01-01 | 5791 | Normal | 9.0 | 12.8 | 0 | N |
| 2025-01-01 | 5747 | Normal | 5.3 | 10.4 | 0 | N |
| 2024-04-09 | 5715 | Ramadan | 13.4 | 17.4 | 21 | N |
| 2025-03-29 | 5499 | Ramadan | 14.6 | 18.6 | 21 | Y |
| 2024-06-16 | 5496 | Normal | 24.0 | 29.2 | 0 | Y |
| 2024-06-17 | 5367 | Normal | 25.0 | 30.6 | 0 | N |
| 2025-03-30 | 4781 | Normal | 12.4 | 15.9 | 0 | Y |
| 2024-04-15 | 4763 | Normal | 16.7 | 22.7 | 0 | N |

## Top-10 worst days per model — regime / weekday summary

For each model, what fraction of its top-N worst days fall in each regime, and what fraction are weekends? Highlights systematic failure modes.

| Model | Worst day MAE | Median worst-day MAE | Top-N regimes (count) | Weekend % |
|---|---|---|---|---|
| chronos-bolt-L720 | 7960 | 6162 | Normal:7, Ramadan:2, Heatwave:1 | 20% |
| lgbm-hijri | 7370 | 4369 | Heatwave:5, Normal:5 | 10% |
| lgbm-nohijri | 7186 | 4825 | Normal:5, Heatwave:4, Ramadan:1 | 20% |
| moirai-1.1-small-L336 | 7761 | 6129 | Heatwave:4, Normal:4, Ramadan:2 | 20% |
| mstl_ets-hijri | 9833 | 7922 | Normal:6, Heatwave:3, Ramadan:1 | 30% |
| mstl_ets-nohijri | 9833 | 7922 | Normal:6, Heatwave:3, Ramadan:1 | 30% |
| patchtsmixer-hijri-L168 | 7079 | 3904 | Normal:6, Ramadan:3, Heatwave:1 | 20% |
| patchtsmixer-nohijri-L168 | 6652 | 4179 | Normal:5, Ramadan:3, Heatwave:2 | 20% |
| sarimax-hijri | 11287 | 7524 | Normal:6, Ramadan:3, Heatwave:1 | 90% |
| sarimax-nohijri | 11747 | 8079 | Normal:6, Ramadan:3, Heatwave:1 | 70% |
| time-moe-200m-L720 | 8316 | 5779 | Normal:6, Heatwave:2, Ramadan:2 | 20% |
| timesfm-2.5-L168 | 9618 | 5970 | Normal:7, Heatwave:2, Ramadan:1 | 20% |

## Single worst day per model (with conditions)

| Model | Worst day | MAE | Regime | Mean temp | Max temp | Ramadan hrs | Wknd |
|---|---|---|---|---|---|---|---|
| chronos-bolt-L720 | 2024-04-10 | 7960 | Normal | 13.8 | 18.2 | 0 | N |
| lgbm-hijri | 2024-06-15 | 7370 | Heatwave | 23.9 | 28.1 | 0 | Y |
| lgbm-nohijri | 2024-06-15 | 7186 | Heatwave | 23.9 | 28.1 | 0 | Y |
| moirai-1.1-small-L336 | 2024-06-24 | 7761 | Heatwave | 26.1 | 31.6 | 0 | N |
| mstl_ets-hijri | 2024-06-15 | 9833 | Heatwave | 23.9 | 28.1 | 0 | Y |
| mstl_ets-nohijri | 2024-06-15 | 9833 | Heatwave | 23.9 | 28.1 | 0 | Y |
| patchtsmixer-hijri-L168 | 2024-06-15 | 7079 | Heatwave | 23.9 | 28.1 | 0 | Y |
| patchtsmixer-nohijri-L168 | 2024-06-15 | 6652 | Heatwave | 23.9 | 28.1 | 0 | Y |
| sarimax-hijri | 2024-06-16 | 11287 | Normal | 24.0 | 29.2 | 0 | Y |
| sarimax-nohijri | 2025-03-30 | 11747 | Normal | 12.4 | 15.9 | 0 | Y |
| time-moe-200m-L720 | 2024-04-10 | 8316 | Normal | 13.8 | 18.2 | 0 | N |
| timesfm-2.5-L168 | 2024-06-24 | 9618 | Heatwave | 26.1 | 31.6 | 0 | N |
