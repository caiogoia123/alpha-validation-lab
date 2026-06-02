"""Métricas de trading a partir de uma lista de operações simuladas.

Todas as métricas são computadas sobre objetos `Trade` (ver engine.py). A
curva de capital é sequencial: saldo inicial + soma acumulada do PnL líquido.
"""
from __future__ import annotations

import math

import numpy as np

from src.backtest.engine import Trade

# ~525.600 minutos por ano: usado para anualizar o Sharpe a partir da duração
# real do período de teste (operamos em candles de 1 minuto).
_MINUTES_PER_YEAR = 365 * 24 * 60


def equity_curve(trades: list[Trade], initial_capital: float) -> np.ndarray:
    """Saldo após cada operação (ordem de fechamento)."""
    ordered = sorted(trades, key=lambda t: t.exit_time)
    nets = np.array([t.net_pnl for t in ordered], dtype=float)
    return initial_capital + np.cumsum(nets)


def max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    """Drawdown máximo absoluto e percentual a partir da curva de capital."""
    if equity.size == 0:
        return 0.0, 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd_abs = float(drawdown.min())
    # Percentual relativo ao pico vigente (evita divisão por zero).
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(running_max != 0, drawdown / running_max, 0.0)
    max_dd_pct = float(dd_pct.min())
    return max_dd_abs, max_dd_pct


def compute_metrics(trades: list[Trade], cfg) -> dict:
    """Conjunto completo de métricas de uma estratégia.

    Retorna dicionário com: n_trades, win_rate, profit_factor, sharpe (por
    operação e anualizado), max drawdown (abs/%), retorno acumulado, ganho/
    perda médios, expectância e exposição por motivo de saída.
    """
    n = len(trades)
    if n == 0:
        return _empty_metrics()

    nets = np.array([t.net_pnl for t in trades], dtype=float)
    rets = np.array([t.return_pct for t in trades], dtype=float)

    wins = nets[nets > 0]
    losses = nets[nets < 0]

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())  # positivo

    win_rate = len(wins) / n
    # Acurácia direcional: skill do sinal antes de custos (vs. ~50% do acaso).
    directional_accuracy = float(
        np.mean([t.direction_correct for t in trades]))
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0  # negativo
    expectancy = float(nets.mean())  # PnL esperado por operação (moeda)
    total_pnl = float(nets.sum())

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 \
        else (math.inf if gross_profit > 0 else 0.0)

    # Sharpe por operação a partir dos retornos por operação.
    std = float(rets.std(ddof=1)) if n > 1 else 0.0
    sharpe_trade = float(rets.mean() / std) if std > 0 else 0.0

    # Anualização pela frequência real de operações no período de teste.
    sharpe_annual = _annualize_sharpe(trades, sharpe_trade)

    equity = equity_curve(trades, cfg.initial_capital)
    max_dd_abs, max_dd_pct = max_drawdown(equity)
    cumulative_return = total_pnl / cfg.initial_capital

    reasons = {"SL": 0, "TP": 0, "TIME": 0}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    return {
        "n_trades": n,
        "win_rate": win_rate,
        "directional_accuracy": directional_accuracy,
        "profit_factor": profit_factor,
        "sharpe_trade": sharpe_trade,
        "sharpe_annual": sharpe_annual,
        "max_drawdown_abs": max_dd_abs,
        "max_drawdown_pct": max_dd_pct,
        "cumulative_return": cumulative_return,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "expectancy_pct": expectancy / cfg.value_per_trade,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "exit_reasons": reasons,
        "final_equity": float(equity[-1]) if equity.size else cfg.initial_capital,
    }


def _annualize_sharpe(trades: list[Trade], sharpe_trade: float) -> float:
    """Escala o Sharpe por operação pela quantidade de operações/ano."""
    if len(trades) < 2 or sharpe_trade == 0.0:
        return 0.0
    span_ms = trades[-1].exit_time - trades[0].entry_time
    if span_ms <= 0:
        return 0.0
    span_minutes = span_ms / 60_000.0
    trades_per_year = len(trades) * (_MINUTES_PER_YEAR / span_minutes)
    return sharpe_trade * math.sqrt(max(trades_per_year, 1.0))


def _empty_metrics() -> dict:
    return {
        "n_trades": 0, "win_rate": 0.0, "directional_accuracy": 0.0,
        "profit_factor": 0.0,
        "sharpe_trade": 0.0, "sharpe_annual": 0.0,
        "max_drawdown_abs": 0.0, "max_drawdown_pct": 0.0,
        "cumulative_return": 0.0, "total_pnl": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0,
        "expectancy_pct": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
        "exit_reasons": {"SL": 0, "TP": 0, "TIME": 0}, "final_equity": 0.0,
    }
