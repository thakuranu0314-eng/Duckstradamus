# LSTM Hyperparameter Optimisation — File Overview

Based on `LSTM_dave_adaption1`. The key difference is that this version assumes the data pipeline has already done all heavy preprocessing (lagging, rolling averages, cleaning). This file only adds cyclical time features and scales the data, then runs a hyperparameter search before the main CV.

---

## Overall Flow

```
Load data
  └── Filter to time window (DATE_START / DATE_END)
        └── Add cyclical time features
              └── Define X / y
                    └── 80/20 train/test split
                          └── Random hyperparameter search (train set only)
                                └── 12-fold expanding-window CV with best params
                                      └── Summary table + monthly plots
                                            └── Final model on full 80%
```

---

## Central Control Cell — Window Configuration

```python
DATE_START = None   # e.g. '2022-01-01'
DATE_END   = None   # e.g. '2024-12-31'
CV_FOLDS   = 12     # reduce to 6 for short windows
```

This is the only cell you need to change when switching between a short optimisation trial and a full run. Everything downstream adapts automatically.

**Rule of thumb for CV_FOLDS:** `CV_FOLDS × 720 ≤ 60% of total rows`

| Use case | DATE_START | DATE_END | CV_FOLDS |
|---|---|---|---|
| Quick optimisation trial | `'2022-01-01'` | `'2024-12-31'` | `6` |
| Full production run | `None` | `None` | `12` |

---

## Preprocessing — What This File Does and Does Not Do

| Step | Done upstream | Done here |
|---|---|---|
| Lag features | ✓ | |
| Rolling averages | ✓ | |
| Forward-fill / back-fill | ✓ | |
| NaN check | | ✓ |
| Cyclical time features | | ✓ |
| StandardScaler | | ✓ (per fold, no leakage) |

### Cyclical Time Features

Raw integers are misleading for a neural network — hour 23 and hour 0 look far apart but are actually neighbours. Converting to sin/cos puts them on a circle so the distance is correct.

Applied to: hour-of-day (period 24), day-of-week (period 7), day-of-year (period 365).

### StandardScaler

Fitted on the training split of each fold independently. Never fitted on test data. Applied before building sequences.

---

## Sequence Structure

| Parameter | Value | Meaning |
|---|---|---|
| `INPUT_LENGTH` | 168 | 1 week of hourly history as input |
| `OUTPUT_LENGTH` | 24 | Predict the next 24 hours at once |
| `HORIZON` | 1 | 1-step gap — model never sees the first hour it predicts |
| `STRIDE` | 12 | Consecutive windows overlap by half a day |

The sequence builder converts the flat feature table into 3D blocks of shape `(samples, 168, n_features)` for X and `(samples, 24)` for y.

---

## Model Architecture

```
Input (168 timesteps × n_features)
  └── LSTM layer 1  [lstm1_units, return_sequences=True, dropout]
        └── LSTM layer 2  [lstm2_units, return_sequences=False, dropout]
              └── Dense hidden layer  [dense_units, ReLU]
                    └── Dropout
                          └── Output layer  [24 neurons, linear]
```

- **Loss:** Huber loss — less sensitive to electricity price spikes than MSE
- **Optimizer:** Adam with fixed learning rate (tuned by random search)
- **Early stopping:** monitors validation loss, restores best weights

---

## Hyperparameter Random Search

Searches `N_ITER` random combinations from the parameter grid. Uses a **3-fold inner CV on the training set only** — the test set is never touched here.

### Parameters Tuned

| Parameter | Options | What it controls |
|---|---|---|
| `lstm1_units` | 32, 64, 128, 256 | Capacity of the first recurrent layer |
| `lstm2_units` | 16, 32, 64 | Capacity of the second recurrent layer |
| `dropout` | 0.1, 0.2, 0.3, 0.4 | Regularisation strength |
| `dense_units` | 16, 32, 64 | Head capacity before output |
| `learning_rate` | 0.001, 0.005, 0.01 | Adam step size |
| `batch_size` | 16, 32, 64 | Sequences per gradient update |

### Search Process

1. Draw `N_ITER` random combinations from the grid
2. For each combination: run 3-fold CV on train set, average the MAEs
3. Pick the combination with the lowest average MAE → `BEST_PARAMS`
4. `BEST_PARAMS` flows into the CV loop and final model automatically

`N_ITER = 10` is fast. Raise to 20–30 for a more thorough search.

---

## 12-Fold Cross-Validation

Uses `TimeSeriesSplit(n_splits=CV_FOLDS, test_size=720)`:
- Each test window ≈ 1 month (720 hours)
- Training window **expands** each fold — more data available as time progresses
- Folds are time-ordered — no future data leaks into training

### Per-fold Steps

1. Fit `StandardScaler` on fold's training rows only
2. Scale train and test
3. Build sliding-window sequences
4. Train LSTM with early stopping (`patience=8`)
5. Predict on test sequences
6. Score against model and naive baseline

### Naive Baseline

"Tomorrow looks like today same hour" — the `target_lag_24h` column.
**MASE < 1** means the LSTM beats this baseline for that month.

### Metrics Reported

| Metric | Description |
|---|---|
| MAE model | Mean absolute error of LSTM predictions |
| MAE naive | Mean absolute error of naive baseline |
| RMSE model / naive | Root mean squared error |
| MASE | MAE model ÷ MAE naive — below 1 = beats naive |

---

## Outputs

| Output | What it shows |
|---|---|
| Summary table | MAE / RMSE / MASE per month + average row |
| Monthly plots | Best and worst week per fold (actual vs LSTM vs naive) |
| Final model metrics | MAE / RMSE on the held-out 20% test set |

---

## Key Design Decisions

- **StandardScaler over baked-in Normalization layer** — scaling happens per fold outside the model so it can be correctly isolated from test data in the CV loop
- **Huber loss** — robust to the extreme price spikes common in NZ electricity markets
- **Expanding window CV** — mirrors how the model would be used in production (more data over time)
- **Lookback prepended to test sequences** — each fold's test window is prefixed with `INPUT_LENGTH` rows from the end of the training set so the first test sequence has a full input window without peeking into the future
