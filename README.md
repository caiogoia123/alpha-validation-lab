# Predictability ≠ Profitability
### A statistical & economic validation framework for crypto market hypotheses — BTC/USDT case study

![CI](https://img.shields.io/badge/CI-passing-brightgreen)
![tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![lint](https://img.shields.io/badge/lint-ruff-purple)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

🌐 **English** · [Português](README.pt-br.md)

> **TL;DR** — Predicting the **direction** of Bitcoin is statistically
> indistinguishable from a coin flip after costs. **Volatility** looks highly
> predictable (AUC ≈ 0.79)… until a *purged cross-validation* shows a one-line
> persistence baseline **beats the XGBoost**. This repo is a disciplined,
> reproducible investigation into **what is actually predictable in markets —
> and why predictability still doesn't pay.**

This is not a trading bot. It is a **research framework** that takes a popular
hypothesis ("ML can predict BTC direction"), tries hard to *falsify* it, follows
the evidence to where signal actually lives, and then stress-tests that finding
until it breaks. The headline result is a *negative* one — delivered with the
rigor that negative results deserve.

---

## ⏱ Results at a glance (30-second version)

| Claim under test | Verdict | Evidence |
|---|---|---|
| BTC **direction** is predictable | ❌ **Rejected** | AUC ≈ 0.50; at 250k candles a *statistically* real 51.2% edge (p=0.002) loses **−150%** after costs |
| Changing the **time horizon** fixes it | ❌ **Rejected** | 5 min → 4 h all fail: short horizons have signal but move < cost; long horizons have move > cost but no signal |
| **Volatility** is predictable | ✅ **Confirmed**… | AUC ≈ 0.79, stable across 8k→250k candles, Lift@10% ≈ 3× |
| …and that's a **model** achievement | ❌ **Rejected** | Under **purged CV**, a trivial *persistence* baseline (AUC 0.81) **beats** the XGBoost (0.74). It was volatility clustering all along. |
| Any simple strategy is **profitable** | ❌ **Rejected** | **Deflated Sharpe Ratio = 0.00** across 12 configurations |

**Three numbers that summarize the project:** `Direction AUC ≈ 0.50` · `Round-trip
cost 0.12% > median 5-min move 0.05%` · `Deflated Sharpe ≈ 0.00`.

![Predictability by target](reports/target_predictability.png)

---

## 1. Motivation

Financial ML is drowning in positive results that don't survive contact with
out-of-sample data, transaction costs, or multiple-testing corrections. The
overwhelmingly common portfolio project — *"my model predicts BTC with 90%
accuracy"* — is almost always leakage.

This project inverts the incentive. The goal was never to *confirm* that BTC is
predictable; it was to find out **whether it is at all, and at what cost** — and
to be honest when the answer is no.

## 2. Research questions

1. Is BTC **price direction** predictable out-of-sample, after costs?
2. Does the **prediction horizon** (5 min → 4 h) change the answer?
3. If not direction, **what** *is* predictable?
4. Does statistical predictability translate into **economic value**?
5. Do any findings **survive** purged cross-validation and selection-bias correction?

## 3. Methodology

| Component | Choice |
|---|---|
| **Data** | Binance BTC/USDT OHLCV, 1-minute, up to **250k candles (~174 days)**, stored in SQLite (idempotent ingestion). See [`docs/DATA_CARD.md`](docs/DATA_CARD.md). |
| **Features** | RSI, MACD, EMA9, EMA21 (+ 1-bar return, volume change) — deliberately fixed across all experiments as a *control*. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md). |
| **Model** | A single **XGBoost** classifier, reused for every target (direction, magnitude, volatility, regime). No model is changed between studies — the *target* is the independent variable. |
| **Validation** | Temporal split (no shuffle), walk-forward, **purged & embargoed K-fold** (López de Prado), **non-overlapping sampling** for significance, Wilson confidence intervals, z-tests, **Deflated Sharpe Ratio** (Bailey & López de Prado). |
| **Economics** | Explicit cost model (taker fee + slippage ≈ 0.12% round trip), time-based and SL/TP exits, profit factor, expectancy, drawdown. |

## 4. Key findings

**Direction is noise.** Across horizons and sample sizes (8k / 100k / 250k), the
directional AUC sits at ≈ 0.50. The only statistically significant edge (51.2% at
5 min, p=0.002, visible *only* with 250k candles) is economically worthless: it
loses ~150% to costs because the median 5-minute move (~0.05%) is **less than half
the round-trip cost (~0.12%)**.

**Volatility is predictable — trivially.** Volatility is the most predictable
target by a wide margin (AUC ≈ 0.79 vs. 0.50 for direction). But this is the
textbook *volatility-clustering* effect: under purged CV a one-line persistence
baseline (`vol_{t+1} ≈ vol_t`, AUC 0.81) **outperforms** the XGBoost (0.74). The
ML model adds **negative** value over a trivial rule.

**Predictability ≠ profitability.** Even using the volatility signal as a *filter*
to trade only high-movement regimes improves profit factor (it removes
cost-dominated trades) but never produces a profitable strategy. The **Deflated
Sharpe Ratio across 12 strategy configurations is 0.00** — the best result is
exactly what you'd expect from testing many strategies with no real edge.

## 5. The methodological catches (the interesting part)

Real research is mostly about *not fooling yourself*. Three moments where the
naive answer was wrong:

- **Overlapping windows inflate significance.** For a 4-hour horizon on 1-minute
  bars, consecutive labels share 239/240 of their future → massive
  autocorrelation. Treating predictions as independent gave a **spurious
  p = 0.000**; correcting to non-overlapping samples (≈ N/H independent
  observations) gave **p = 0.53**. The "edge" was an artifact of the sample count.
- **A trivial baseline beat the ML model.** The celebrated AUC 0.79 only looked
  impressive until compared, under purged CV, against persistence. Always race
  your model against the dumbest possible predictor.
- **A unit test caught a label-leakage bug.** `test_no_leakage.py` flagged that
  the last *H* rows (no observable future) were silently labeled `0` instead of
  dropped — because `NaN > x` is `False` in pandas. Fixed, with a regression test.

## 6. Limitations & threats to validity

Stated plainly, because pretending they don't exist is the amateur move:

- **Single asset, single venue, single period.** Findings are demonstrated on
  BTC/USDT spot; generalization to other assets/regimes is future work.
- **Feature set is direction-oriented.** No explicit volatility feature (ATR,
  realized vol) — which is *why* the trivial vol baseline wins. A vol-native
  feature set was deliberately out of scope (controlled comparison).
- **Costs are modeled, not measured.** Real fills, maker rebates, and queue
  position would change the economics (likely favorably at lower frequency).
- **No order-flow / L2 data.** The one information source with documented
  short-horizon directional value is absent by design.

## 7. Reproducibility

```bash
pip install -e ".[dev]"        # install package + dev tools
make test                      # 25 tests, core numerical logic
make data                      # fetch 250k candles from Binance -> SQLite
make experiments               # direction / horizon / target studies
make validate                  # purged CV + Deflated Sharpe
```

Everything is config-driven and seeded; reports and figures regenerate into
[`reports/`](reports/).

## 8. Project structure

```
src/
├── data/          # Binance client + SQLite store (idempotent ingestion)
├── features/      # RSI, MACD, EMA — causal, unit-tested
├── model/         # XGBoost training / dataset construction (no leakage)
├── backtest/      # cost-aware engine, metrics, baseline strategies
├── experiments/   # horizon study, target study, statistical validity
├── economics/     # economic value of the volatility signal
└── validation/    # purged CV, Deflated Sharpe, persistence baseline
tests/             # indicators, no-leakage, engine, metrics, purged CV
reports/           # auto-generated markdown + figures
```

## 9. Roadmap

- **v0.1 — MVP:** pipeline, XGBoost, backtest, direction/horizon/target studies.
- **v0.2 — Portfolio-ready (this release):** tests + CI, purged CV, Deflated
  Sharpe, persistence baseline, hero README, model/data cards.
- **v0.3 — Research depth:** hypothesis-agnostic & multi-asset framework;
  funding/basis-carry study; experiment tracking; written report.
- **v0.4 — Platform:** Docker, docs site, DVC data pipeline.

## 10. What this project demonstrates

Hypothesis design · out-of-sample discipline · transaction-cost realism ·
autocorrelation-aware significance testing · purged cross-validation ·
selection-bias correction (Deflated Sharpe) · honest reporting of negative
results. The skill on display is **knowing how to validate a claim**, not
producing a number.

## References

- M. López de Prado, *Advances in Financial Machine Learning* (2018) — purged CV, embargo.
- D. Bailey & M. López de Prado, *The Deflated Sharpe Ratio* (2014).
- C. Harvey & Y. Liu, *Backtesting* / multiple-testing in finance (2015).

## License

MIT — research/educational use. **Not financial advice.** See [LICENSE](LICENSE)
and [CITATION.cff](CITATION.cff).
