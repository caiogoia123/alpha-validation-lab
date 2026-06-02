"""Geradores de sinal para o backtest.

Cada estratégia recebe o DataFrame de candles do período de teste (já com
indicadores) e devolve duas Series alinhadas por posição:
  * signal     -> +1 (long), -1 (short) ou 0 (fora)
  * confidence -> convicção em [0,1] (0.5 quando não aplicável)

O modelo é o ÚNICO componente de IA. Os demais são baselines de controle para
medir se a IA agrega valor sobre regras triviais.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.features.indicators import FEATURE_COLUMNS
from src.model.dataset import build_supervised


def train_model_in_sample(train_df: pd.DataFrame):
    """Treina o MESMO XGBoost (mesmos XGB_PARAMS) apenas no in-sample.

    Reutiliza `build_supervised` e os hiperparâmetros de produção — nenhum
    modelo novo é introduzido. Retorna o classificador ajustado.
    """
    from xgboost import XGBClassifier  # import local: dependência pesada

    X, y = build_supervised(train_df)
    if len(X) < 100 or y.nunique() < 2:
        raise RuntimeError("In-sample insuficiente para treinar o modelo.")
    model = XGBClassifier(**config.XGB_PARAMS)
    model.fit(X, y)
    return model


def model_signals(model, test_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Sinais do modelo: long se prob_up>=0.5, short caso contrário.

    A convicção é max(prob_up, prob_down); o filtro de confiança mínima é
    aplicado depois, no motor. Linhas sem features válidas ficam com sinal 0.
    """
    feats = test_df[FEATURE_COLUMNS]
    valid = feats.notna().all(axis=1)

    signal = pd.Series(0, index=test_df.index, dtype=int)
    confidence = pd.Series(0.5, index=test_df.index, dtype=float)

    if valid.any():
        proba_up = model.predict_proba(feats[valid])[:, 1]
        idx = test_df.index[valid]
        sig = np.where(proba_up >= 0.5, 1, -1)
        conf = np.maximum(proba_up, 1.0 - proba_up)
        signal.loc[idx] = sig
        confidence.loc[idx] = conf

    return signal, confidence


def random_signals(test_df: pd.DataFrame, seed: int) -> tuple[pd.Series, pd.Series]:
    """Cara ou coroa: direção aleatória, convicção neutra (0.5)."""
    rng = np.random.default_rng(seed)
    sig = rng.choice([-1, 1], size=len(test_df))
    signal = pd.Series(sig, index=test_df.index, dtype=int)
    confidence = pd.Series(0.5, index=test_df.index, dtype=float)
    return signal, confidence


def always_long_signals(test_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Sempre comprado."""
    signal = pd.Series(1, index=test_df.index, dtype=int)
    confidence = pd.Series(0.5, index=test_df.index, dtype=float)
    return signal, confidence


def always_short_signals(test_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Sempre vendido."""
    signal = pd.Series(-1, index=test_df.index, dtype=int)
    confidence = pd.Series(0.5, index=test_df.index, dtype=float)
    return signal, confidence


def ema_cross_signals(test_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Cruzamento EMA9/EMA21 sem IA: long se EMA9>EMA21, senão short.

    Usa as colunas de EMA já calculadas pelos indicadores existentes.
    """
    sig = np.where(test_df["ema9"] > test_df["ema21"], 1, -1)
    signal = pd.Series(sig, index=test_df.index, dtype=int)
    confidence = pd.Series(0.5, index=test_df.index, dtype=float)
    # Onde as EMAs não existem (warm-up), fica fora.
    invalid = test_df["ema9"].isna() | test_df["ema21"].isna()
    signal[invalid] = 0
    return signal, confidence
