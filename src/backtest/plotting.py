"""Geração do gráfico da curva de capital.

Usa backend não interativo (Agg) para salvar PNG sem janela. Mostra três
camadas em um único painel comparável:
  * Evolução do saldo (curva de capital do modelo)
  * Drawdowns (área sombreada abaixo do pico vigente)
  * Lucro acumulado (eixo secundário)
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # sem display; apenas arquivo
import matplotlib.pyplot as plt
import numpy as np

import config
from src.backtest.engine import Trade
from src.backtest.metrics import equity_curve


def plot_equity_curve(trades: list[Trade], cfg, path: str = None,
                      title: str = "Curva de Capital — Modelo (out-of-sample)") -> str:
    """Gera e salva o gráfico. Retorna o caminho do arquivo."""
    path = path or os.path.join(config.REPORTS_DIR, "equity_curve.png")

    equity = equity_curve(trades, cfg.initial_capital)
    if equity.size == 0:
        # Sem operações: gera um gráfico vazio informativo.
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(title + " (sem operações)")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return path

    x = np.arange(1, equity.size + 1)
    running_max = np.maximum.accumulate(equity)
    cumulative_profit = equity - cfg.initial_capital

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Saldo e pico vigente.
    ax1.plot(x, equity, color="#2563eb", linewidth=1.6, label="Saldo")
    ax1.plot(x, running_max, color="#9ca3af", linewidth=0.9,
             linestyle="--", label="Pico vigente")
    # Drawdown como área entre pico e saldo.
    ax1.fill_between(x, equity, running_max, where=equity < running_max,
                     color="#ef4444", alpha=0.20, label="Drawdown")
    ax1.axhline(cfg.initial_capital, color="#6b7280", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel("Operação nº")
    ax1.set_ylabel("Saldo (USDT)")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)

    # Lucro acumulado no eixo secundário.
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative_profit, color="#16a34a", linewidth=1.0,
             alpha=0.7, label="Lucro acumulado")
    ax2.set_ylabel("Lucro acumulado (USDT)")

    # Legenda combinada dos dois eixos.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
