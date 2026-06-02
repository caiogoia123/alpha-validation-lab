# Model Card — XGBoost (controlled across all studies)

## Overview
A single gradient-boosted tree classifier (**XGBoost**) is reused across every
experiment. The model is held **constant on purpose**: the independent variable in
this research is the **prediction target** (direction, magnitude, volatility,
regime), not the model. This isolates *what is predictable* from *modeling tricks*.

## Inputs (features) — fixed
`ema9`, `ema21`, `ema_diff` (normalized EMA spread), `rsi`, `macd`, `macd_signal`,
`macd_hist`, `ret_1` (1-bar return), `vol_change` (volume change). All causal, all
price/volume-derived. **No volatility-native feature (ATR, realized vol)** — a
deliberate scope choice that the validation later exposes as a limitation.

## Hyperparameters
`n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`,
`colsample_bytree=0.8`. Objective is set automatically by task: `binary:logistic`
(2-class) or `multi:softprob` (k-class). Identical otherwise across targets.

## Training & evaluation protocol
- **Temporal split**, no shuffling. In-sample → train, out-of-sample → evaluate.
- **Purged & embargoed K-fold** for the volatility study (leakage-safe AUC).
- **Non-overlapping sampling** (every `H` candles) for significance tests — avoids
  autocorrelation-inflated p-values.
- Metrics: AUC, accuracy vs. majority baseline, Wilson CIs, profit factor,
  expectancy, **Deflated Sharpe Ratio**.

## Performance summary
| Target | OOS AUC | Note |
|---|---|---|
| Direction | ≈ 0.50 | indistinguishable from chance |
| Magnitude (terciles) | ≈ 0.57 | weak |
| Regime (5-class) | ≈ 0.63 | moderate |
| Absolute move | ≈ 0.64 | moderate |
| **Volatility** | **≈ 0.79** | strong — **but beaten by a persistence baseline under purged CV** |

## Intended use & limitations
- **Intended use:** research into market predictability and validation methodology.
- **Out of scope:** live trading, financial advice. The economics show no
  profitable strategy after costs.
- **Key limitation:** for volatility, a trivial `vol_{t+1} ≈ vol_t` baseline
  outperforms this model — the feature set is direction-oriented, not vol-native.

## Ethical / risk notes
Markets are adversarial and non-stationary. A model that validates today can decay
without warning. Negative results here are a feature, not a failure.
