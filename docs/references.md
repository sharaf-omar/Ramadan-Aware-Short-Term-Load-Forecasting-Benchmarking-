# References

Bibliography for the benchmark, organised by topic. Citation strings
are in a copy-pasteable format suitable for both APA-style and
inline-numeric citation systems.

---

## Time-series foundation models (TSFMs)

The four TSFMs benchmarked in Plans 2-3, plus the deep-learning
sibling used in Plan 5.

**Chronos / Chronos-Bolt**
- Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P.,
  Shen, H., Shchur, O., Rangapuram, S. S., Pineda Arango, S., Kapoor,
  S., Zschiegner, J., Maddix, D. C., Wang, H., Mahoney, M. W., Torkkola,
  K., Wilson, A. G., Bohlke-Schneider, M., & Wang, Y. (2024). *Chronos:
  Learning the Language of Time Series.* arXiv:2403.07815.
- Model checkpoint: `amazon/chronos-bolt-base` (HuggingFace Hub).
  Inference library: `chronos-forecasting==1.5.2`.

**TimesFM**
- Das, A., Kong, W., Sen, R., & Zhou, Y. (2024). *A decoder-only
  foundation model for time-series forecasting.* In Proceedings of the
  41st International Conference on Machine Learning (ICML 2024).
  arXiv:2310.10688.
- Checkpoint: `google/timesfm-2.5-200m-pytorch`. Library: `timesfm`
  installed from GitHub HEAD (PyPI 1.x ships an incompatible Lingvo
  dependency).

**Moirai**
- Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., & Sahoo, D.
  (2024). *Unified Training of Universal Time Series Forecasting
  Transformers.* In Proceedings of the 41st International Conference on
  Machine Learning (ICML 2024). arXiv:2402.02592.
- Checkpoint: `Salesforce/moirai-1.1-R-small`. Library: `uni2ts==2.0.0`.

**Time-MoE**
- Shi, X., Wang, S., Nie, Y., Li, D., Ye, Z., Wen, Q., & Jin, M.
  (2024). *Time-MoE: Billion-Scale Time Series Foundation Models with
  Mixture of Experts.* arXiv:2409.16040.
- Checkpoint: `Maple728/TimeMoE-200M`. Inference uses a direct call to
  the model's 32-step forecasting head (the bundled `generate()` is
  broken at the time of this benchmark).

**PatchTSMixer (Plan 5 substitute for vanilla PatchTST)**
- Ekambaram, V., Jati, A., Nguyen, N., Sinthong, P., & Kalagnanam, J.
  (2023). *TSMixer: Lightweight MLP-Mixer Model for Multivariate Time
  Series Forecasting.* In Proceedings of the 29th ACM SIGKDD Conference
  on Knowledge Discovery and Data Mining (KDD 2023). arXiv:2306.09364.
- Implementation: `transformers.models.patchtsmixer.PatchTSMixerForPrediction`,
  cross-channel mixing mode (`mode="mix_channel"`).

**PatchTST (referenced; substituted)**
- Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). *A
  Time Series is Worth 64 Words: Long-term Forecasting with
  Transformers.* In Proceedings of the 11th International Conference
  on Learning Representations (ICLR 2023). arXiv:2211.14730.
- Substituted in Plan 5 because HuggingFace's
  `PatchTSTForPrediction` is channel-independent by design (no
  cross-channel attention), so the Hijri-ablation would be a no-op.
  See [`docs/superpowers/specs/2026-05-14-patchtsmixer-baseline-design.md`](superpowers/specs/2026-05-14-patchtsmixer-baseline-design.md).

---

## Classical time-series baselines (Plan 4)

**MSTL — multiple seasonal-trend decomposition**
- Bandara, K., Hyndman, R. J., & Bergmeir, C. (2021). *MSTL: A
  Seasonal-Trend Decomposition Algorithm for Time Series with Multiple
  Seasonal Patterns.* International Journal of Operational Research
  (forthcoming). arXiv:2107.13462.
- Implementation: `statsmodels.tsa.seasonal.MSTL` with periods
  (24, 168). Trend forecast via Holt's linear method (additive).

**ETS — exponential smoothing**
- Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008).
  *Forecasting with Exponential Smoothing: The State Space Approach.*
  Springer-Verlag.
- Implementation: `statsmodels.tsa.holtwinters.ExponentialSmoothing`
  with `trend="add"` (Holt's linear trend) for the trajectory across
  the 24-hour horizon.

**SARIMAX — seasonal ARIMA with exogenous regressors**
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
  *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting:
  Principles and Practice* (3rd ed.), §9.10 (hourly electricity demand).
  https://otexts.com/fpp3/
- Implementation: `statsmodels.tsa.statespace.sarimax.SARIMAX` with
  order (1, 0, 1) and seasonal_order (0, 1, 1, 24) — the Hyndman &
  Athanasopoulos default for hourly electricity load.

---

## Tree-based machine learning (Plan 1)

**LightGBM**
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., &
  Liu, T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting
  Decision Tree.* In Advances in Neural Information Processing Systems
  30 (NeurIPS 2017).
- Implementation: `lightgbm==4.x`. Hyperparameters tuned via Optuna
  (50 trials) on the val-2023 set; final ensemble of 5 seeds (42-46).

**Optuna**
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).
  *Optuna: A Next-generation Hyperparameter Optimization Framework.*
  In Proceedings of the 25th ACM SIGKDD Conference on Knowledge
  Discovery and Data Mining (KDD 2019).

---

## Statistical-test methodology (Plan 7)

**Diebold-Mariano test**
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive
  Accuracy.* Journal of Business & Economic Statistics, 13(3),
  253-263.
- HAC (Heteroscedasticity- and Autocorrelation-Consistent) variance:
  Newey, W. K., & West, K. D. (1987). *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.*
  Econometrica, 55(3), 703-708.
- Truncation lag set to the forecast horizon h=24 (Newey-West rule for
  h-step-ahead forecasts).

**Holm-Bonferroni multiple-comparison adjustment**
- Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test
  Procedure.* Scandinavian Journal of Statistics, 6(2), 65-70.
- Applied within each regime's pairwise DM family (n=66 for the
  12-model headline cohort; n=190 for the full 20-model + tier-1/2/3
  cohort).

**Stationary block bootstrap**
- Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap.*
  Journal of the American Statistical Association, 89(428), 1303-1313.
- Block length set to 24 (one day) for the hourly time series. 1000
  resamples per CI, α = 0.05.

---

## Forecasting domain references

**Short-term electricity load forecasting (STLF) reviews**
- Hong, T., & Fan, S. (2016). *Probabilistic Electric Load Forecasting:
  A Tutorial Review.* International Journal of Forecasting, 32(3),
  914-938.
- Hong, T., Pinson, P., Wang, Y., Weron, R., Yang, D., & Zareipour, H.
  (2020). *Energy Forecasting: A Review and Outlook.* IEEE Open Access
  Journal of Power and Energy, 7, 376-388.

**Hourly load forecasting reference**
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting:
  Principles and Practice* (3rd ed.), Chapter 9 (ARIMA models).

---

## Data sources

**Turkish electricity load (EPIAS — Energy Exchange Istanbul)**
- EPIAS Transparency Platform. https://seffaflik.epias.com.tr/
- Real-time consumption series, hourly granularity, 2017-present.

**Weather (ERA5 reanalysis)**
- Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A.,
  Muñoz‐Sabater, J., et al. (2020). *The ERA5 Global Reanalysis.*
  Quarterly Journal of the Royal Meteorological Society, 146(730),
  1999-2049.
- Population-weighted national temperature aggregation: this project,
  using 2024 TÜİK province population weights.

**Hijri calendar**
- `hijridate` Python library (Umm al-Qura calendar variant).
  https://github.com/mhalshehri/hijridate

---

## Software stack

The full pinned environment is in [`pyproject.toml`](../pyproject.toml).
Key dependencies for citation:

- Python 3.12.12
- `torch==2.4.1+cu124` (CUDA 12.4 build for the 4070 Laptop)
- `transformers==4.48.3`
- `pandas==2.1.4`, `numpy==1.26.4`, `scipy`
- `statsmodels` (MSTL, ExponentialSmoothing, SARIMAX, OLS, HAC)
- `lightgbm` (≥ 4)
- `chronos-forecasting==1.5.2`, `uni2ts==2.0.0`, `gluonts`
- `timesfm` (GitHub HEAD; PyPI 1.0 broken on Python 3.12)
- `matplotlib==3.10.9` (figures)
- `optuna` (LightGBM hyperparameter tuning)
- `pytest==9.0.3` (152 automated tests)

---

## Within-project cross-references

- Headline narrative: [`capstone_synthesis.md`](capstone_synthesis.md)
- Full 31-model statistical appendix: [`statistical_appendix.md`](statistical_appendix.md)
- Per-plan technical detail: see the docs sidebar in
  [`docs/`](.)
- All design specs and implementation plans:
  [`docs/superpowers/`](superpowers/)
