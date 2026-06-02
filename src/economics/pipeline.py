"""Pipeline central da validação econômica.

Treina o MESMO classificador de volatilidade (alta/baixa) usado no estudo de
alvos, gera um SCORE de volatilidade por candle no out-of-sample (probabilidade
da classe 'alta volatilidade') e calcula as quantidades realizadas futuras
necessárias às etapas seguintes (retorno, |movimento|, vol realizada, MAE/MFE).

Nada de novo é introduzido: features = indicadores atuais; modelo = XGBoost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.experiments.target_study import _make_xgb
from src.experiments.targets import build_target, future_quantities
from src.features.indicators import FEATURE_COLUMNS, add_indicators


@dataclass
class VolContext:
    """Artefato consumido por todas as etapas econômicas."""
    test_df: pd.DataFrame        # candles do out-of-sample (índice 0..n-1)
    vol_score: np.ndarray        # prob. de alta volatilidade por candle de teste
    realized: pd.DataFrame       # ret, absmove, rv, er, mae_long, mfe_long, y_high, extreme
    horizon: int                 # H candles (= holding das estratégias)
    base_params: dict            # parâmetros de custo/backtest
    rv_median: float             # limiar treino de 'alta vol' (mediana)
    rv_extreme: float            # limiar treino de evento extremo (p90)
    auc: float                   # AUC do modelo de vol no teste (referência)


def _future_excursions(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                       horizon: int):
    """MAE/MFE de uma posição comprada mantida por `horizon` candles.

    mae_long[t] = (min low em t+1..t+H) / close[t] - 1   (<= 0)
    mfe_long[t] = (max high em t+1..t+H) / close[t] - 1   (>= 0)
    """
    n = len(close)
    acc_min = np.full(n, np.inf)
    acc_max = np.full(n, -np.inf)
    for k in range(1, horizon + 1):
        shifted_low = np.full(n, np.nan)
        shifted_high = np.full(n, np.nan)
        shifted_low[:n - k] = low[k:]
        shifted_high[:n - k] = high[k:]
        acc_min = np.fmin(acc_min, shifted_low)
        acc_max = np.fmax(acc_max, shifted_high)
    valid = np.arange(n) < (n - horizon)
    acc_min = np.where(valid, acc_min, np.nan)
    acc_max = np.where(valid, acc_max, np.nan)
    mae = acc_min / close - 1.0
    mfe = acc_max / close - 1.0
    return mae, mfe


def build_context(candles: pd.DataFrame, horizon: int,
                  base_params: dict) -> VolContext:
    """Treina o modelo de volatilidade e monta o contexto out-of-sample."""
    from sklearn.metrics import roc_auc_score

    enriched = add_indicators(candles).reset_index(drop=True)
    close = enriched["close"]
    fq = future_quantities(close, horizon)

    feat_ok = enriched[FEATURE_COLUMNS].notna().all(axis=1)
    test_fraction = base_params["test_fraction"]
    split = int(len(enriched) * (1.0 - test_fraction))
    train_region = feat_ok & (enriched.index < split)

    # Rótulo de alta/baixa volatilidade (mediana de treino) — mesmo do alvo.
    spec = build_target("volatilidade", fq, train_region)
    y = spec.y
    rv_median = float(fq["rv"][train_region & fq["rv"].notna()].median())
    rv_extreme = float(fq["rv"][train_region & fq["rv"].notna()].quantile(0.90))

    train_mask = feat_ok & y.notna() & (enriched.index < split)
    X_train = enriched.loc[train_mask, FEATURE_COLUMNS].astype(float)
    y_train = y[train_mask].astype(int)

    model = _make_xgb(2)
    model.fit(X_train, y_train)

    # Score de volatilidade para TODO o out-of-sample.
    test_df = enriched.iloc[split:].reset_index(drop=True)
    X_test = test_df[FEATURE_COLUMNS].astype(float)
    vol_score = model.predict_proba(X_test)[:, 1]

    # Quantidades realizadas alinhadas ao test_df.
    mae, mfe = _future_excursions(close.to_numpy(), enriched["high"].to_numpy(),
                                  enriched["low"].to_numpy(), horizon)
    realized = pd.DataFrame({
        "ret": fq["ret"].to_numpy()[split:],
        "absmove": fq["absmove"].to_numpy()[split:],
        "rv": fq["rv"].to_numpy()[split:],
        "er": fq["er"].to_numpy()[split:],
        "mae_long": mae[split:],
        "mfe_long": mfe[split:],
    }).reset_index(drop=True)
    realized["y_high"] = (realized["rv"] > rv_median).astype(float)
    realized["extreme"] = (realized["rv"] > rv_extreme).astype(float)
    # Onde a vol realizada é desconhecida (cauda), invalida os rótulos.
    realized.loc[realized["rv"].isna(), ["y_high", "extreme"]] = np.nan

    # AUC de referência (em amostra cheia de teste; descritivo).
    valid = realized["y_high"].notna()
    try:
        auc = float(roc_auc_score(realized["y_high"][valid], vol_score[valid.values]))
    except ValueError:
        auc = float("nan")

    return VolContext(test_df=test_df, vol_score=vol_score, realized=realized,
                      horizon=horizon, base_params=base_params,
                      rv_median=rv_median, rv_extreme=rv_extreme, auc=auc)
