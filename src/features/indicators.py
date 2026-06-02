"""Cálculo de indicadores técnicos com pandas (sem dependência de TA-Lib).

Funções puras: recebem um DataFrame de candles (colunas open/high/low/close/
volume) e retornam Series alinhadas pelo índice. A função `add_indicators`
agrega tudo e devolve o DataFrame enriquecido.
"""
from __future__ import annotations

import pandas as pd

import config


def ema(series: pd.Series, period: int) -> pd.Series:
    """Média móvel exponencial."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Índice de Força Relativa (RSI) clássico de Wilder.

    Usa média exponencial de ganhos/perdas (alpha = 1/period), que é a
    formulação padrão de Wilder. Retorna valores entre 0 e 100.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100.0 - (100.0 / (1.0 + rs))
    # Quando avg_loss == 0 o RS é infinito -> RSI satura em 100.
    rsi_values = rsi_values.where(avg_loss != 0, 100.0)
    return rsi_values


def macd(series: pd.Series,
         fast: int = config.MACD_FAST,
         slow: int = config.MACD_SLOW,
         signal: int = config.MACD_SIGNAL):
    """MACD = EMA(fast) - EMA(slow), com linha de sinal e histograma.

    Retorna (macd_line, signal_line, histogram).
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona todas as colunas de indicadores ao DataFrame de candles.

    Colunas geradas:
      ema9, ema21, ema_diff (ema9-ema21 normalizada),
      rsi,
      macd, macd_signal, macd_hist,
      ret_1 (retorno do último candle), vol_change (variação de volume).
    """
    out = df.copy()
    close = out["close"]

    out["ema9"] = ema(close, config.EMA_SHORT)
    out["ema21"] = ema(close, config.EMA_LONG)
    # Diferença relativa entre as EMAs: captura cruzamentos independente do preço.
    out["ema_diff"] = (out["ema9"] - out["ema21"]) / close

    out["rsi"] = rsi(close)

    macd_line, signal_line, hist = macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    # Atributos auxiliares de momentum/volume — baratos e úteis para o modelo.
    out["ret_1"] = close.pct_change()
    out["vol_change"] = out["volume"].pct_change().replace([float("inf")], 0.0)

    return out


# Lista canônica de features consumidas pelo modelo. Mantida aqui para que
# treino e inferência usem exatamente as mesmas colunas, na mesma ordem.
FEATURE_COLUMNS = [
    "ema9",
    "ema21",
    "ema_diff",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "ret_1",
    "vol_change",
]
