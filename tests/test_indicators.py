"""Testes de corretude dos indicadores técnicos."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.indicators import FEATURE_COLUMNS, add_indicators, ema, macd, rsi


def test_ema_matches_pandas_ewm():
    s = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected = s.ewm(span=3, adjust=False).mean()
    pd.testing.assert_series_equal(ema(s, 3), expected)


def test_rsi_bounds_and_monotonic_series():
    # Série estritamente crescente -> RSI satura perto de 100.
    up = pd.Series(np.arange(1, 200, dtype=float))
    r = rsi(up, period=14).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    assert r.iloc[-1] > 99.0

    # Série estritamente decrescente -> RSI perto de 0.
    down = pd.Series(np.arange(200, 1, -1, dtype=float))
    rd = rsi(down, period=14).dropna()
    assert rd.iloc[-1] < 1.0


def test_macd_histogram_identity():
    s = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 1, 300)) + 100)
    macd_line, signal_line, hist = macd(s)
    # Histograma = linha MACD - linha de sinal (por definição).
    pd.testing.assert_series_equal(hist, macd_line - signal_line,
                                   check_names=False)


def test_add_indicators_creates_all_feature_columns(synthetic_candles):
    out = add_indicators(synthetic_candles)
    for col in FEATURE_COLUMNS:
        assert col in out.columns
    # ema9 reage mais rápido que ema21: após warm-up não são idênticas.
    tail = out.dropna(subset=["ema9", "ema21"]).iloc[-1]
    assert tail["ema9"] != tail["ema21"]
