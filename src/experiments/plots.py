"""Gráficos comparativos do estudo de horizonte.

Gera, para cada tamanho de amostra, um painel com as métricas-chave por
horizonte e, ao final, uma figura comparando a acurácia direcional entre os
diferentes tamanhos de histórico.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _hz_labels(horizons: list[int]) -> list[str]:
    out = []
    for h in horizons:
        out.append(f"{h//60}h" if h >= 60 and h % 60 == 0 else f"{h}m")
    return out


def plot_size_panel(study: dict, horizons: list[int], size_label: str,
                    path: str) -> str:
    """Painel 2x3 com as métricas por horizonte para um tamanho de amostra."""
    rows = study["results"]
    labels = _hz_labels(horizons)
    x = np.arange(len(horizons))

    acc = [r["directional_accuracy"] * 100 for r in rows]
    ci_lo = [(r["directional_accuracy"] - r["acc_ci_low"]) * 100 for r in rows]
    ci_hi = [(r["acc_ci_high"] - r["directional_accuracy"]) * 100 for r in rows]
    pf = [min(r["profit_factor"], 3.0) for r in rows]  # corta ∞ p/ visual
    cum = [r["cumulative_return"] * 100 for r in rows]
    sharpe = [r["sharpe_trade"] for r in rows]
    expect = [r["expectancy"] for r in rows]
    move = [r["median_move_pct"] for r in rows]
    hurdle = [r["cost_hurdle_pct"] for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Estudo de horizonte — {size_label}", fontsize=14, weight="bold")

    # (a) Acurácia direcional com IC 95% e linha do acaso.
    ax = axes[0, 0]
    ax.errorbar(x, acc, yerr=[ci_lo, ci_hi], fmt="o-", color="#2563eb",
                capsize=4, label="Acurácia ±IC95%")
    ax.axhline(50, color="#ef4444", linestyle="--", label="Acaso (50%)")
    ax.set_title("Acurácia direcional (out-of-sample)")
    ax.set_ylabel("%")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (b) Profit factor.
    ax = axes[0, 1]
    ax.bar(x, pf, color="#0ea5e9")
    ax.axhline(1.0, color="#ef4444", linestyle="--", label="Break-even (1,0)")
    ax.set_title("Profit Factor (após custos)")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (c) Retorno acumulado.
    ax = axes[0, 2]
    colors = ["#16a34a" if v >= 0 else "#ef4444" for v in cum]
    ax.bar(x, cum, color=colors)
    ax.axhline(0, color="#6b7280", linewidth=0.8)
    ax.set_title("Retorno acumulado (após custos)")
    ax.set_ylabel("%")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(alpha=0.25)

    # (d) Sharpe por operação.
    ax = axes[1, 0]
    colors = ["#16a34a" if v >= 0 else "#ef4444" for v in sharpe]
    ax.bar(x, sharpe, color=colors)
    ax.axhline(0, color="#6b7280", linewidth=0.8)
    ax.set_title("Sharpe por operação")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(alpha=0.25)

    # (e) Expectância por operação.
    ax = axes[1, 1]
    colors = ["#16a34a" if v >= 0 else "#ef4444" for v in expect]
    ax.bar(x, expect, color=colors)
    ax.axhline(0, color="#6b7280", linewidth=0.8)
    ax.set_title("Expectância (USDT/operação)")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(alpha=0.25)

    # (f) Movimento típico vs. custo de ida-e-volta.
    ax = axes[1, 2]
    width = 0.38
    ax.bar(x - width/2, move, width, color="#8b5cf6", label="Movimento mediano |H|")
    ax.bar(x + width/2, hurdle, width, color="#f59e0b", label="Custo ida+volta")
    ax.set_title("Movimento típico vs. custo")
    ax.set_ylabel("%")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accuracy_across_sizes(studies: dict[str, dict], horizons: list[int],
                               path: str) -> str:
    """Acurácia direcional por horizonte, uma linha por tamanho de amostra."""
    labels = _hz_labels(horizons)
    x = np.arange(len(horizons))

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#2563eb", "#16a34a", "#db2777", "#f59e0b"]
    for i, (size_label, study) in enumerate(studies.items()):
        acc = [r["directional_accuracy"] * 100 for r in study["results"]]
        lo = [(r["directional_accuracy"] - r["acc_ci_low"]) * 100
              for r in study["results"]]
        hi = [(r["acc_ci_high"] - r["directional_accuracy"]) * 100
              for r in study["results"]]
        ax.errorbar(x + (i - len(studies)/2) * 0.04, acc, yerr=[lo, hi],
                    fmt="o-", capsize=3, color=palette[i % len(palette)],
                    label=size_label)
    ax.axhline(50, color="#ef4444", linestyle="--", label="Acaso (50%)")
    ax.set_title("Acurácia direcional por horizonte — consistência vs. tamanho de amostra")
    ax.set_xlabel("Horizonte"); ax.set_ylabel("Acurácia (%)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
