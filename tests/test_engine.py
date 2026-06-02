"""Testes do motor de backtest: SL/TP/tempo, sinais de custo e slippage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, simulate_single_trade

_BASE = {
    "initial_capital": 10_000.0, "value_per_trade": 1_000.0,
    "stop_loss_pct": 1.0, "take_profit_pct": 1.0, "fee_pct": 0.0,
    "slippage_pct": 0.0, "min_confidence": 0.0, "test_fraction": 0.3,
    "max_holding": 5, "random_seed": 0,
}


def _candles(prices):
    """Constrói candles com high/low = close (sem ruído intrabar) para controle."""
    n = len(prices)
    return pd.DataFrame({
        "open_time": np.arange(n) * 60_000,
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": np.ones(n), "close_time": np.arange(n) * 60_000 + 59_999,
    })


def test_take_profit_long_hit():
    cfg = BacktestConfig.from_dict(_BASE)
    # Preço sobe 1% no candle seguinte -> TP de 1% é atingido.
    candles = _candles([100.0, 101.5, 101.5, 101.5, 101.5, 101.5])
    trade = simulate_single_trade(candles, 0, direction=1, cfg=cfg)
    assert trade.exit_reason == "TP"
    assert trade.net_pnl > 0
    assert trade.direction_correct == 1


def test_stop_loss_long_hit():
    cfg = BacktestConfig.from_dict(_BASE)
    candles = _candles([100.0, 98.0, 98.0, 98.0, 98.0, 98.0])
    trade = simulate_single_trade(candles, 0, direction=1, cfg=cfg)
    assert trade.exit_reason == "SL"
    assert trade.net_pnl < 0
    assert trade.direction_correct == 0


def test_time_exit_when_no_bracket():
    # SL/TP = 0 desativa o bracket -> saída por tempo no horizonte.
    params = dict(_BASE, stop_loss_pct=0.0, take_profit_pct=0.0, max_holding=3)
    cfg = BacktestConfig.from_dict(params)
    candles = _candles([100.0, 100.2, 100.4, 100.6, 100.8])
    trade = simulate_single_trade(candles, 0, direction=1, cfg=cfg)
    assert trade.exit_reason == "TIME"
    # Saída no fechamento do 3º candle à frente (índice 3).
    assert trade.exit_time == candles["open_time"].iloc[3]


def test_costs_reduce_pnl():
    # Mesmo trade, com e sem custos: a versão com custos tem PnL menor.
    candles = _candles([100.0, 101.5, 101.5, 101.5, 101.5, 101.5])
    no_cost = simulate_single_trade(candles, 0, 1, BacktestConfig.from_dict(_BASE))
    with_cost = simulate_single_trade(
        candles, 0, 1,
        BacktestConfig.from_dict(dict(_BASE, fee_pct=0.1, slippage_pct=0.05)))
    assert with_cost.net_pnl < no_cost.net_pnl
    assert with_cost.fees > 0


def test_short_direction_profits_on_drop():
    cfg = BacktestConfig.from_dict(_BASE)
    candles = _candles([100.0, 98.5, 98.5, 98.5, 98.5, 98.5])
    trade = simulate_single_trade(candles, 0, direction=-1, cfg=cfg)
    assert trade.exit_reason == "TP"   # short atinge alvo na queda
    assert trade.net_pnl > 0
