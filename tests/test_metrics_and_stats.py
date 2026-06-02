"""Testes de métricas de trading e de validade estatística."""
from __future__ import annotations

import math

from src.backtest.engine import BacktestConfig, Trade
from src.backtest.metrics import compute_metrics
from src.experiments import stats_validity as sv
from src.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_stats,
)

_CFG = BacktestConfig.from_dict({
    "initial_capital": 10_000.0, "value_per_trade": 1_000.0,
    "stop_loss_pct": 1.0, "take_profit_pct": 1.0, "fee_pct": 0.0,
    "slippage_pct": 0.0, "min_confidence": 0.0, "test_fraction": 0.3,
    "max_holding": 5, "random_seed": 0,
})


def _trade(net: float, t: int) -> Trade:
    return Trade(entry_time=t, exit_time=t + 1, direction=1, entry_price=100.0,
                 exit_price=100.0 + net, exit_reason="TIME", gross_pnl=net,
                 fees=0.0, net_pnl=net, return_pct=net / 1_000.0,
                 confidence=0.5, direction_correct=1 if net > 0 else 0)


def test_profit_factor_and_win_rate():
    trades = [_trade(10, 0), _trade(10, 1), _trade(-5, 2), _trade(-5, 3)]
    m = compute_metrics(trades, _CFG)
    assert m["win_rate"] == 0.5
    # PF = lucro bruto (20) / prejuízo bruto (10) = 2.0
    assert math.isclose(m["profit_factor"], 2.0, rel_tol=1e-9)
    assert math.isclose(m["expectancy"], (10 + 10 - 5 - 5) / 4, rel_tol=1e-9)


def test_max_drawdown_is_negative_after_loss_sequence():
    trades = [_trade(100, 0), _trade(-50, 1), _trade(-50, 2)]
    m = compute_metrics(trades, _CFG)
    assert m["max_drawdown_abs"] <= 0
    assert m["max_drawdown_pct"] <= 0


def test_wilson_ci_contains_point_estimate():
    low, high = sv.wilson_ci(60, 100)
    assert low < 0.60 < high
    assert 0.0 <= low <= high <= 1.0


def test_proportion_z_test_significant_for_strong_signal():
    _, p = sv.proportion_z_test(700, 1000, p0=0.5)  # 70% vs 50%
    assert p < 1e-6


def test_required_n_grows_for_smaller_edge():
    assert sv.required_n(0.52) > sv.required_n(0.55)


def test_margin_of_error_shrinks_with_n():
    assert sv.margin_of_error(10_000) < sv.margin_of_error(100)


def test_psr_high_for_strong_consistent_returns():
    # Retornos pequenos e consistentemente positivos -> PSR alto.
    rng = __import__("numpy").random.default_rng(0)
    rets = rng.normal(0.01, 0.01, 500)
    st = sharpe_stats(rets)
    assert probabilistic_sharpe_ratio(st, 0.0) > 0.95


def test_deflated_sharpe_penalizes_many_trials():
    rng = __import__("numpy").random.default_rng(1)
    rets = rng.normal(0.02, 0.05, 300)
    st = sharpe_stats(rets)
    dsr_few = deflated_sharpe_ratio(st, n_trials=1, sr_trials_std=0.1)
    dsr_many = deflated_sharpe_ratio(st, n_trials=200, sr_trials_std=0.1)
    # Mais tentativas -> benchmark deflacionado maior -> DSR menor.
    assert dsr_many["deflated_benchmark"] > dsr_few["deflated_benchmark"]
    assert dsr_many["deflated_sharpe_ratio"] <= dsr_few["deflated_sharpe_ratio"]
