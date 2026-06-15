# Model Overview — Duckstradamus XGBoost Suite

All files located in `Model_XGBoost/Dave/`

---

## 1. `xgboost_lasso_adapt.ipynb`
**Status:** Early prototype — superseded by v2/v3

First adaptation of the base XGBoost model. Introduced the core preprocessing pipeline and the Lasso (L1) regularisation concept.

**Key contents:**
- Hourly data, OTA2201 target price
- 24h lag on production, demand, other price nodes
- Cyclical time features (sin/cos encoding)
- Target lags: 24h, 168h (1 week), 8760h (1 year)
- Rolling mean/std: 24h, 168h, 8760h
- XGBRegressor with `reg_alpha` (Lasso penalty)
- Feature importance threshold pruning
- 5-fold TimeSeriesSplit CV
- Basic actual vs predicted plot

---

## 2. `xgboost_lasso_v2.ipynb`
**Status:** Stable reference model

Extended version of the adapt notebook. Added walk-forward CV visualisation, seasonal forecast plots, and the naive baseline comparison.

**Key contents:**
- All preprocessing from adapt (ffill, lags, rolling, time features)
- Naive baseline: predicts today = yesterday same hour (`target_lag_24h`)
- Naive vs model comparison: MAE, RMSE, MASE
- 5-fold large-window CV (metrics)
- 9-fold 24h walk-forward CV with per-fold plots
- Rolling 14-day forecast function (actual lag features, no feedback loop)
- Seasonal 14-day plots: Summer / Autumn / Winter / Spring (NZ 2024)
- August 2024 extended window (25 Jul – 7 Sep)
- Feature importance bar chart + distribution plot
- Importance threshold pruning + pruned model comparison
- Predictions clipped to ≥ 0

**Notable:** MAPE dropped — unreliable for electricity prices (near-zero/negative prices blow up the metric). Use MAE + RMSE + MASE instead.

---

## 3. `xgboost_lasso_v3.ipynb`
**Status:** ✅ Current main hourly model

Clean consolidated version. Single CV loop replaces all previous separate CV runs. Fair model vs naive comparison across 12 months.

**Key contents:**
- Same preprocessing pipeline as v2
- **12-fold expanding-window CV** (`test_size=720h ≈ 1 month`)
  - Covers full year 2024, one month per fold
  - XGBoost trained on all data before each test window (expanding)
  - Naive baseline evaluated on **same window** → fair MASE per fold
  - Live fold-by-fold metrics table during training
- **Summary table:** MAE / RMSE / MASE per month + averages + beats naive ✓/✗
- **Monthly visualisation:** 12 figures, one per fold
  - Best week (lowest MAE) + Worst week (highest MAE) per month
  - Actual vs XGBoost vs Naive in each subplot
  - y-axis floored at 0
- **Final model:** trained on 80% for feature importance + seasonal plots
- Feature importance ranking
- Seasonal 14-day forecast plots (all four NZ seasons, naive shown alongside)

**CV design rationale:** Expanding window chosen over rolling because electricity seasonality requires multiple years of historical data. Rolling window would discard useful seasonal patterns.

---

## 4. `xgboost_daily.ipynb`
**Status:** ✅ Current daily model

Daily-resolution version of v3. Hourly data aggregated to daily before modelling. Loses intraday patterns but gives a cleaner trend signal.

**Aggregation rules:**

| Type | Columns | Method |
|---|---|---|
| Price | all el_price nodes | mean |
| Generation | Coal, Diesel, Ele, Gas, Geo, Hydro, Solar, Wind, Wood | sum (MW×24h ≈ MWh) |
| Demand | demand_GWh_* | sum |
| Flow | avg_flow_MW | mean |
| Flow | peak_flow_MW | max |
| Flow | Direction | mean |
| Outages | outage_*_MW | mean (capacity rate) |
| Weather | wind, temp, solar, sunshine | mean |
| Hydro | lake_level, active_storage | mean |
| Holiday | holiday | max |

**Key differences from v3:**
- Lags: 1d, 7d, 365d (replacing 24h, 168h, 8760h)
- Rolling windows: 7d, 30d, 365d
- No hour/hour_sin/cos features (irrelevant at daily resolution)
- CV `test_size=30` days
- Plots have dot markers per day
- Naive = yesterday's daily average price (`target_lag_1d`)

**Important:** MAE here is in daily average NZD/MWh — lower than hourly MAE by design (daily averaging smooths spikes). Do not compare MAE directly between hourly and daily models.

---

## 5. `xgboost_classifier_spikes.ipynb`
**Status:** ✅ Current spike classifier --> failed: prediction not good enough to use in LSTM

Multiclass classification model predicting whether each hour is a price spike. Designed to feed spike labels into the downstream LSTM.

**Spike definition:**
- Centered rolling percentile window (±2 days = 96h total)
- Class 0: Normal — price ≤ rolling 95th percentile
- Class 1: Weak spike — rolling 95th < price ≤ rolling 99th
- Class 2: Strong spike — price > rolling 99th percentile
- Centered window prevents trending prices from being mislabelled as spikes
- Model input features still use shift(1) — no leakage

**Key contents:**
- Same preprocessing pipeline as v3 (hourly, same features)
- `XGBClassifier` with `objective='multi:softmax'`, `num_class=3`
- `compute_sample_weight('balanced')` + `SPIKE_WEIGHT_MULTIPLIER` for recall bias
- Probability threshold prediction (`predict_proba`) instead of argmax:
  - `SPIKE_PROB_THRESHOLD` — lower = more spikes flagged (higher recall)
  - Designed for asymmetric cost: missing a spike > false alarm
- Classification report (precision / recall / F1 per class)
- Confusion matrix
- Feature importance
- Importance threshold pruning
- **4-month winter forecast plot** (Jul–Oct 2024): price line + rolling thresholds
- **Seasonal 2-week plots** (4 seasons × 2024): coloured price line by predicted class
- **August extended plot** (25 Jul – 7 Sep 2024)
- Visualization style: price line coloured by prediction, actual spikes marked with triangles above the line

**Key tuning parameters:**
| Parameter | Effect |
|---|---|
| `WEAK_QUANTILE` | Percentile threshold for weak spike (default 0.95) |
| `STRONG_QUANTILE` | Percentile threshold for strong spike (default 0.99) |
| `ROLLING_WINDOW` | Window size for spike definition (default 96h = ±2 days) |
| `SPIKE_WEIGHT_MULTIPLIER` | Extra weight on spike classes during training |
| `SPIKE_PROB_THRESHOLD` | Combined spike probability to flag an hour (default 0.2) |

---

## 6. `logreg_spike_check.ipynb`
**Status:** Diagnostic tool

Logistic regression sanity check. Tests whether any features are linearly informative for spike detection before committing to the XGBoost classifier.

**Key contents:**
- Same preprocessing + spike definition as classifier
- `LogisticRegression(multi_class='multinomial', class_weight='balanced')`
- `StandardScaler` applied (required for logistic regression, not for XGBoost)
- Classification report + confusion matrix
- Coefficient table ranked by max absolute coefficient across classes
- Bar chart: top 20 coefficients for weak spike and strong spike classes

**How to read results:**
- Recall on spike classes clearly above random (~30%+) → features are informative → XGBoost has something to learn
- Near-zero recall → spikes may be random w.r.t. current features, or spike definition needs rethinking
- Coefficient chart → shows which features push toward spike prediction linearly

---

## Metric reference

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | mean(\|actual − pred\|) | Average error in NZD/MWh |
| RMSE | √mean((actual − pred)²) | Penalises large errors more |
| R² | 1 − SS_res/SS_tot | 1 = perfect, 0 = mean predictor, <0 = worse than mean |
| MASE | MAE_model / MAE_naive | <1 = beats naive, >1 = worse than naive |
| MAPE | — | **Not used** — breaks with near-zero/negative electricity prices |

---

## Pipeline position

These models sit between data preprocessing and the downstream LSTM:

```
Data_Processing/preprocessed_data.csv
        │
        ├── xgboost_lasso_v3.ipynb     → hourly price predictions
        ├── xgboost_daily.ipynb         → daily price predictions
        └── xgboost_classifier_spikes.ipynb → is_spike labels (0/1/2)
                                                    │
                                              → LSTM input features
```
