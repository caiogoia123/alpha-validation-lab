"""Testes de ausência de vazamento temporal (look-ahead).

Dois invariantes críticos:
  1. CAUSALIDADE DAS FEATURES: o valor de um indicador em t não pode depender de
     dados de t+1 em diante. Verificamos que calcular sobre a série truncada até
     t dá o mesmo valor que calcular sobre a série inteira.
  2. ALINHAMENTO DO RÓTULO: em build_supervised, y[t] deve refletir o futuro
     (close[t+H] > close[t]) e as últimas H linhas (sem futuro) devem sair.
"""
from __future__ import annotations

import numpy as np

from src.features.indicators import FEATURE_COLUMNS, add_indicators
from src.model.dataset import build_supervised


def test_features_are_causal(synthetic_candles):
    """Indicador em t calculado com dados até t == calculado com a série toda."""
    full = add_indicators(synthetic_candles)
    for t in (300, 700, 1_200):
        truncated = add_indicators(synthetic_candles.iloc[: t + 1])
        a = full.loc[t, FEATURE_COLUMNS].to_numpy(dtype=float)
        b = truncated.iloc[-1][FEATURE_COLUMNS].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9), f"vazamento em t={t}"


def test_label_reflects_future_direction(synthetic_candles):
    horizon = 5
    X, y = build_supervised(synthetic_candles, horizon=horizon)
    close = synthetic_candles["close"].to_numpy()
    # Reconstrói o rótulo esperado para alguns índices presentes em X.
    for pos in X.index[:50]:
        expected = 1 if close[pos + horizon] > close[pos] else 0
        assert int(y.loc[pos]) == expected


def test_last_horizon_rows_are_dropped(synthetic_candles):
    """As últimas H linhas não têm futuro observável e não podem estar em X."""
    horizon = 5
    X, _ = build_supervised(synthetic_candles, horizon=horizon)
    n = len(synthetic_candles)
    assert X.index.max() < n - horizon


def test_no_nan_in_training_matrix(synthetic_candles):
    X, y = build_supervised(synthetic_candles, horizon=5)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert len(X) == len(y)
