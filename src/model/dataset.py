"""Construção do dataset supervisionado.

A partir dos candles + indicadores, gera a matriz de features X e o rótulo y.

Rótulo (classificação binária da direção):
    y[t] = 1  se  close[t + HORIZON] > close[t] * (1 + MIN_MOVE_PCT/100)
    y[t] = 0  caso contrário

Ou seja, "o preço daqui a HORIZON_MINUTES estará acima do atual?".
"""
from __future__ import annotations

import pandas as pd

import config
from src.features.indicators import FEATURE_COLUMNS, add_indicators


def build_supervised(df: pd.DataFrame,
                     horizon: int = None) -> tuple[pd.DataFrame, pd.Series]:
    """Monta (X, y) para treino.

    Recebe candles crus (ordenados por tempo crescente) e devolve features e
    rótulo já alinhados, sem linhas com NaN (warm-up dos indicadores) e sem as
    últimas HORIZON linhas (que não têm futuro observado).

    `horizon` (em candles/minutos) permite variar o horizonte de previsão sem
    alterar o pipeline. Quando None, usa config.HORIZON_MINUTES.
    """
    enriched = add_indicators(df)

    horizon = horizon or config.HORIZON_MINUTES
    threshold = 1.0 + config.MIN_MOVE_PCT / 100.0

    future_close = enriched["close"].shift(-horizon)
    label = (future_close > enriched["close"] * threshold)
    # IMPORTANTE: onde não há futuro observável (últimas `horizon` linhas),
    # future_close é NaN e a comparação retornaria False — rótulo espúrio. Força
    # NaN nessas linhas para que sejam descartadas (evita vazamento/viés).
    label = label.where(future_close.notna()).astype("Int64")

    enriched = enriched.assign(label=label)

    # Remove warm-up dos indicadores e a cauda sem futuro.
    enriched = enriched.dropna(subset=FEATURE_COLUMNS + ["label"])

    X = enriched[FEATURE_COLUMNS].astype(float)
    y = enriched["label"].astype(int)
    return X, y


def build_inference_row(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Monta a única linha de features do candle mais recente já fechado.

    Retorna (X_row, info) onde info traz candle_time e price (close) usados
    para registrar a previsão. Levanta ValueError se não há dados suficientes.
    """
    enriched = add_indicators(df)
    enriched = enriched.dropna(subset=FEATURE_COLUMNS)
    if enriched.empty:
        raise ValueError("Dados insuficientes para calcular indicadores.")

    last = enriched.iloc[-1]
    X_row = last[FEATURE_COLUMNS].astype(float).to_frame().T
    info = {
        "candle_time": int(last["open_time"]),
        "close_time": int(last["close_time"]),
        "price": float(last["close"]),
    }
    return X_row, info
