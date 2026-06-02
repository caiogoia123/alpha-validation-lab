"""Estratégias simples (regras de preço, NÃO features do modelo) e o filtro de
volatilidade aplicado sobre elas.

As regras usam apenas preço/indicadores já existentes (EMA9/EMA21, máximas,
retorno) e servem de cobaia para testar se o SCORE de volatilidade do modelo
elimina operações ruins. O XGBoost não é re-treinado aqui — só o score já
produzido pelo pipeline é usado como filtro.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, simulate_sequential
from src.backtest.metrics import compute_metrics

# Lookbacks das estratégias (parâmetros de regra, não de modelo).
BREAKOUT_LOOKBACK = 20
MOMENTUM_LOOKBACK = 10


def strat_ema(test_df: pd.DataFrame) -> pd.Series:
    """Estratégia A — EMA9 > EMA21 → comprado (senão fora)."""
    sig = np.where(test_df["ema9"] > test_df["ema21"], 1, 0)
    s = pd.Series(sig, index=test_df.index, dtype=int)
    s[test_df["ema9"].isna() | test_df["ema21"].isna()] = 0
    return s


def strat_breakout(test_df: pd.DataFrame, lookback: int = BREAKOUT_LOOKBACK) -> pd.Series:
    """Estratégia B — rompimento da máxima das últimas `lookback` barras."""
    prior_high = test_df["high"].shift(1).rolling(lookback).max()
    sig = np.where(test_df["close"] > prior_high, 1, 0)
    s = pd.Series(sig, index=test_df.index, dtype=int)
    s[prior_high.isna()] = 0
    return s


def strat_momentum(test_df: pd.DataFrame, lookback: int = MOMENTUM_LOOKBACK) -> pd.Series:
    """Estratégia C — momentum simples: preço acima do de `lookback` barras atrás."""
    past = test_df["close"].shift(lookback)
    sig = np.where(test_df["close"] > past, 1, 0)
    s = pd.Series(sig, index=test_df.index, dtype=int)
    s[past.isna()] = 0
    return s


STRATEGIES = {
    "A_ema": ("EMA9 > EMA21", strat_ema),
    "B_breakout": (f"Rompimento da máxima ({BREAKOUT_LOOKBACK})", strat_breakout),
    "C_momentum": (f"Momentum simples ({MOMENTUM_LOOKBACK})", strat_momentum),
}


def vol_threshold(vol_score: np.ndarray, top_pct: float) -> float:
    """Limiar do score para manter apenas o top `top_pct` (0-1) mais volátil."""
    return float(np.quantile(vol_score, 1.0 - top_pct))


def apply_filter(signal: pd.Series, vol_score: np.ndarray,
                 top_pct: float) -> pd.Series:
    """Zera os sinais cujo score de volatilidade está abaixo do limiar top_pct."""
    thr = vol_threshold(vol_score, top_pct)
    keep = vol_score >= thr
    return signal.where(pd.Series(keep, index=signal.index), 0)


def _cfg(base_params: dict, horizon: int) -> BacktestConfig:
    """Config de backtest: saída por tempo em H candles, custos do config."""
    params = dict(base_params)
    params["stop_loss_pct"] = 0.0
    params["take_profit_pct"] = 0.0
    params["max_holding"] = horizon
    params["min_confidence"] = 0.0
    return BacktestConfig.from_dict(params)


def run_strategy(test_df: pd.DataFrame, signal: pd.Series, horizon: int,
                 base_params: dict) -> dict:
    """Backtesta uma estratégia (sinal long-only) e devolve métricas + trades."""
    cfg = _cfg(base_params, horizon)
    conf = pd.Series(1.0, index=test_df.index)  # sem filtro de confiança do motor
    trades = simulate_sequential(test_df, signal, conf, cfg)
    metrics = compute_metrics(trades, cfg)
    return {"metrics": metrics, "trades": trades, "n_signals": int((signal != 0).sum())}
