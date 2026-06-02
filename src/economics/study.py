"""Cálculo das Etapas 1–5 da validação econômica da volatilidade.

Cada função recebe o VolContext e devolve estruturas prontas para o relatório.
Não treina modelos nem cria features — apenas analisa o score já produzido.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.economics.pipeline import VolContext
from src.economics.strategies import STRATEGIES, apply_filter, run_strategy

# Rótulos dos 5 grupos de volatilidade prevista (quintis do score).
QUINTILE_LABELS = ["muito baixa", "baixa", "média", "alta", "muito alta"]


# ---------------------------------------------------------------------------
# ETAPA 1 — grupos de volatilidade
# ---------------------------------------------------------------------------
def etapa1_groups(ctx: VolContext) -> list[dict]:
    """Divide as previsões em 5 quintis de score e descreve o realizado."""
    df = ctx.realized.copy()
    df["score"] = ctx.vol_score
    df = df[df["rv"].notna()]                      # precisa de realizado
    # Quintis pelo score previsto.
    df["grp"] = pd.qcut(df["score"], 5, labels=False, duplicates="drop")

    cost = 2 * (ctx.base_params["fee_pct"] + ctx.base_params["slippage_pct"]) / 100.0
    rows = []
    for g in sorted(df["grp"].dropna().unique()):
        sub = df[df["grp"] == g]
        ret = sub["ret"]
        rows.append({
            "group": QUINTILE_LABELS[int(g)],
            "n": int(len(sub)),
            "mean_score": float(sub["score"].mean()),
            "mean_ret_pct": float(ret.mean() * 100),
            "mean_absmove_pct": float(sub["absmove"].mean() * 100),
            "std_ret_pct": float(ret.std() * 100),
            "p05_pct": float(ret.quantile(0.05) * 100),
            "p95_pct": float(ret.quantile(0.95) * 100),
            "mean_mae_pct": float(sub["mae_long"].mean() * 100),   # drawdown médio (long)
            "realized_vol_pct": float(sub["rv"].mean() * 100),
            "cost_pct": cost * 100,
        })
    return rows


# ---------------------------------------------------------------------------
# ETAPA 2 — filtro de oportunidades (varredura de percentis)
# ---------------------------------------------------------------------------
def etapa2_filter(ctx: VolContext, strategy_key: str = "A_ema",
                  top_pcts=(0.10, 0.20, 0.30, 0.40, 0.50)) -> dict:
    """Aplica o filtro de volatilidade a uma estratégia de referência e varre
    os percentis top X%. Inclui a linha 'sem filtro' (100%) para comparação.
    """
    _, fn = STRATEGIES[strategy_key]
    base_signal = fn(ctx.test_df)

    rows = []
    # Sem filtro (todas as oportunidades).
    rows.append(_filter_row(ctx, base_signal, 1.00, "sem filtro (100%)"))
    for p in top_pcts:
        filt = apply_filter(base_signal, ctx.vol_score, p)
        rows.append(_filter_row(ctx, filt, p, f"top {int(p*100)}%"))
    return {"strategy_key": strategy_key,
            "strategy_label": STRATEGIES[strategy_key][0], "rows": rows}


def _filter_row(ctx: VolContext, signal: pd.Series, top_pct: float,
                label: str) -> dict:
    res = run_strategy(ctx.test_df, signal, ctx.horizon, ctx.base_params)
    m = res["metrics"]
    return {
        "label": label, "top_pct": top_pct,
        "n_trades": m["n_trades"],
        "win_rate": m["win_rate"],
        "profit_factor": m["profit_factor"],
        "expectancy": m["expectancy"],
        "cumulative_return": m["cumulative_return"],
        "max_drawdown_pct": m["max_drawdown_pct"],
        "sharpe": m["sharpe_trade"],
    }


# ---------------------------------------------------------------------------
# ETAPA 3 — estratégias simples com e sem filtro
# ---------------------------------------------------------------------------
def etapa3_strategies(ctx: VolContext, top_pct: float = 0.30) -> dict:
    """Roda A/B/C sem filtro e com filtro (top X% de volatilidade)."""
    out = []
    for key, (label, fn) in STRATEGIES.items():
        base_signal = fn(ctx.test_df)
        no_filter = run_strategy(ctx.test_df, base_signal, ctx.horizon, ctx.base_params)["metrics"]
        filt_signal = apply_filter(base_signal, ctx.vol_score, top_pct)
        filt = run_strategy(ctx.test_df, filt_signal, ctx.horizon, ctx.base_params)["metrics"]
        out.append({"key": key, "label": label,
                    "no_filter": _slim(no_filter), "with_filter": _slim(filt)})
    return {"top_pct": top_pct, "strategies": out}


def _slim(m: dict) -> dict:
    return {k: m[k] for k in ("n_trades", "win_rate", "profit_factor",
                              "expectancy", "cumulative_return",
                              "max_drawdown_pct", "sharpe_trade")}


# ---------------------------------------------------------------------------
# ETAPA 4 — valor econômico da informação
# ---------------------------------------------------------------------------
def etapa4_value(ctx: VolContext, ks=(0.10, 0.20, 0.30)) -> dict:
    """Lift, Precision@K, Recall (eventos extremos), Gain e Information Ratio."""
    df = ctx.realized.copy()
    df["score"] = ctx.vol_score
    df = df[df["y_high"].notna()].reset_index(drop=True)

    y = df["y_high"].to_numpy()
    ext = df["extreme"].to_numpy()
    score = df["score"].to_numpy()
    n = len(df)
    base_rate = float(y.mean())
    ext_rate = float(ext.mean())

    order = np.argsort(-score)  # do maior score para o menor
    y_sorted = y[order]
    ext_sorted = ext[order]

    rows = []
    for k in ks:
        topn = max(1, int(n * k))
        sel_y = y_sorted[:topn]
        sel_ext = ext_sorted[:topn]
        precision = float(sel_y.mean())
        lift = precision / base_rate if base_rate > 0 else float("nan")
        recall_pos = float(sel_y.sum() / y.sum()) if y.sum() > 0 else float("nan")
        recall_ext = float(sel_ext.sum() / ext.sum()) if ext.sum() > 0 else float("nan")
        rows.append({
            "k": k, "n_top": topn, "precision": precision, "lift": lift,
            "recall_positives": recall_pos, "recall_extreme": recall_ext,
        })

    # Curva de ganho (gain): % de positivos capturados vs % da população.
    fracs = np.linspace(0.0, 1.0, 21)
    cum_pos = np.cumsum(y_sorted)
    total_pos = y_sorted.sum()
    gain = [float(cum_pos[min(int(f * n), n - 1)] / total_pos) if total_pos > 0 else 0.0
            for f in fracs]

    # Information Ratio: Sharpe (anualizado) da estratégia de referência
    # filtrada vs. não filtrada — mede se a informação melhora risco-retorno.
    from src.economics.strategies import STRATEGIES, run_strategy
    _, fn = STRATEGIES["A_ema"]
    base_signal = fn(ctx.test_df)
    m_nf = run_strategy(ctx.test_df, base_signal, ctx.horizon, ctx.base_params)["metrics"]
    filt = apply_filter(base_signal, ctx.vol_score, 0.30)
    m_f = run_strategy(ctx.test_df, filt, ctx.horizon, ctx.base_params)["metrics"]

    return {
        "base_rate": base_rate, "ext_rate": ext_rate, "n": n,
        "rows": rows, "gain_fracs": fracs.tolist(), "gain": gain,
        "ir_no_filter": m_nf["sharpe_annual"], "ir_with_filter": m_f["sharpe_annual"],
        "sharpe_no_filter": m_nf["sharpe_trade"], "sharpe_with_filter": m_f["sharpe_trade"],
    }


# ---------------------------------------------------------------------------
# ETAPA 5 — identificação de regimes
# ---------------------------------------------------------------------------
def etapa5_regimes(ctx: VolContext) -> dict:
    """Define regimes por tercis do score (baixa/média/alta vol) e mede
    duração média (persistência), frequência e potencial econômico de cada um.
    """
    df = ctx.realized.copy()
    df["score"] = ctx.vol_score
    valid = df["rv"].notna()

    # Limiares por tercil do score (sobre o teste).
    q33, q67 = np.quantile(ctx.vol_score, [1 / 3, 2 / 3])
    regime = np.where(ctx.vol_score >= q67, 2,
                      np.where(ctx.vol_score <= q33, 0, 1))  # 0 baixa,1 média,2 alta
    df["regime"] = regime
    names = {0: "baixa volatilidade (lateral)", 1: "volatilidade média",
             2: "alta volatilidade (explosivo)"}

    cost = 2 * (ctx.base_params["fee_pct"] + ctx.base_params["slippage_pct"]) / 100.0

    # Durações: run-length encoding da sequência temporal de regimes.
    durations = _run_lengths(regime)

    rows = []
    for r in (0, 1, 2):
        sub = df[(df["regime"] == r) & valid]
        runs = durations.get(r, [])
        rows.append({
            "regime": names[r],
            "frequency": float((regime == r).mean()),
            "n": int(len(sub)),
            "avg_duration": float(np.mean(runs)) if runs else 0.0,
            "max_duration": int(np.max(runs)) if runs else 0,
            "mean_absmove_pct": float(sub["absmove"].mean() * 100),
            "mean_realized_vol_pct": float(sub["rv"].mean() * 100),
            "absmove_minus_cost_pct": float(sub["absmove"].mean() * 100 - cost * 100),
            "exploitable": bool(sub["absmove"].mean() > cost),
        })
    return {"rows": rows, "cost_pct": cost * 100,
            "n_regime_changes": int(np.sum(np.diff(regime) != 0))}


def _run_lengths(labels: np.ndarray) -> dict[int, list[int]]:
    """Comprimentos das sequências consecutivas por rótulo (persistência)."""
    out: dict[int, list[int]] = {}
    if len(labels) == 0:
        return out
    cur = labels[0]
    length = 1
    for v in labels[1:]:
        if v == cur:
            length += 1
        else:
            out.setdefault(int(cur), []).append(length)
            cur, length = v, 1
    out.setdefault(int(cur), []).append(length)
    return out
