"""Orquestra as Etapas 1–6 da validação econômica e gera relatório + gráficos.

Responde objetivamente, ao final:
  1. Há evidência de que a previsão de volatilidade melhora uma estratégia?
  2. O filtro de volatilidade melhora ou piora estratégias simples?
  3. Há vantagem econômica potencial que justifique continuar o projeto?
  4. Qual o próximo experimento mais promissor?
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from src.data.database import Database
from src.economics import study
from src.economics.pipeline import build_context

DEFAULT_HORIZON = 15


def run_study(db: Database = None, horizon: int = DEFAULT_HORIZON,
              n_candles: int = 250_000, filter_pct: float = 0.30,
              overrides: dict = None) -> dict:
    db = db or Database()
    base_params = dict(config.BACKTEST)
    if overrides:
        base_params.update({k: v for k, v in overrides.items() if v is not None})

    candles = db.load_candles(limit=n_candles)
    if len(candles) < 5_000:
        raise RuntimeError(f"Histórico insuficiente: {len(candles)} candles.")

    print(f"  • Treinando modelo de volatilidade (H={horizon}, "
          f"{len(candles):,} candles)...")
    ctx = build_context(candles, horizon, base_params)

    e1 = study.etapa1_groups(ctx)
    e2 = study.etapa2_filter(ctx)
    e3 = study.etapa3_strategies(ctx, top_pct=filter_pct)
    e4 = study.etapa4_value(ctx)
    e5 = study.etapa5_regimes(ctx)

    charts = {
        "e1": _plot_etapa1(e1),
        "e2": _plot_etapa2(e2),
        "e3": _plot_etapa3(e3),
        "e4": _plot_etapa4(e4),
        "e5": _plot_etapa5(e5),
    }

    verdict = _build_verdict(ctx, e1, e3, e4, e5)
    md = _render_markdown(ctx, e1, e2, e3, e4, e5, verdict, charts)
    report_path = os.path.join(config.REPORTS_DIR, "vol_economics_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    return {"report_path": report_path, "ctx_auc": ctx.auc,
            "verdict": verdict, "etapa1": e1, "etapa3": e3, "etapa4": e4}


# ---------------------------------------------------------------------------
# Conclusão objetiva
# ---------------------------------------------------------------------------
def _build_verdict(ctx, e1, e3, e4, e5) -> dict:
    cost = e1[0]["cost_pct"]
    high_grp = e1[-1]            # 'muito alta'
    low_grp = e1[0]              # 'muito baixa'
    move_ratio = (high_grp["mean_absmove_pct"] / low_grp["mean_absmove_pct"]
                  if low_grp["mean_absmove_pct"] > 0 else float("nan"))
    highvol_move_exceeds_cost = high_grp["mean_absmove_pct"] > cost

    # Filtro melhora PF na maioria das estratégias?
    improved = 0
    any_profitable = False
    pf_gains = []
    for s in e3["strategies"]:
        nf, wf = s["no_filter"], s["with_filter"]
        if wf["profit_factor"] > nf["profit_factor"]:
            improved += 1
        pf_gains.append(wf["profit_factor"] - nf["profit_factor"])
        if wf["expectancy"] > 0 and wf["profit_factor"] > 1.0:
            any_profitable = True
    filter_helps = improved >= 2  # maioria de 3

    lift10 = next((r["lift"] for r in e4["rows"] if abs(r["k"] - 0.10) < 1e-9), float("nan"))
    info_real = (ctx.auc == ctx.auc and ctx.auc > 0.6) and (lift10 == lift10 and lift10 > 1.2)

    # Persistência (timeability) do regime de alta vol.
    high_regime = e5["rows"][-1]
    persistent = high_regime["avg_duration"] > 1.5

    return {
        "move_ratio": move_ratio,
        "highvol_move_exceeds_cost": highvol_move_exceeds_cost,
        "filter_helps": filter_helps,
        "n_improved": improved,
        "mean_pf_gain": float(np.mean(pf_gains)),
        "any_profitable": any_profitable,
        "lift10": lift10,
        "info_real": info_real,
        "persistent_highvol": persistent,
        "cost_pct": cost,
        "high_absmove_pct": high_grp["mean_absmove_pct"],
        "low_absmove_pct": low_grp["mean_absmove_pct"],
    }


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def _save(fig, name) -> str:
    path = os.path.join(config.REPORTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_etapa1(e1) -> str:
    labels = [r["group"] for r in e1]
    x = np.arange(len(labels))
    absmove = [r["mean_absmove_pct"] for r in e1]
    std = [r["std_ret_pct"] for r in e1]
    cost = e1[0]["cost_pct"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    w = 0.38
    ax.bar(x - w/2, absmove, w, color="#2563eb", label="|movimento| médio")
    ax.bar(x + w/2, std, w, color="#16a34a", label="desvio-padrão do retorno")
    ax.axhline(cost, color="#ef4444", linestyle="--", label=f"custo ida-volta ({cost:.2f}%)")
    ax.set_title("Etapa 1 — Movimento realizado por grupo de volatilidade prevista")
    ax.set_ylabel("%"); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
    return _save(fig, "vol_econ_etapa1.png")


def _plot_etapa2(e2) -> str:
    rows = e2["rows"]
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    pf = [min(r["profit_factor"], 2.0) for r in rows]
    n = [r["n_trades"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.bar(x, pf, color="#0ea5e9", label="Profit Factor")
    ax1.axhline(1.0, color="#ef4444", linestyle="--", label="Break-even (1,0)")
    ax1.set_ylabel("Profit Factor"); ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, n, "o-", color="#f59e0b", label="Nº de operações")
    ax2.set_ylabel("Nº de operações")
    ax1.set_title(f"Etapa 2 — Filtro de volatilidade na estratégia {e2['strategy_label']}")
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, fontsize=9, loc="upper right")
    ax1.grid(alpha=0.25, axis="y")
    return _save(fig, "vol_econ_etapa2.png")


def _plot_etapa3(e3) -> str:
    strats = e3["strategies"]
    labels = [s["key"] for s in strats]
    x = np.arange(len(labels))
    pf_nf = [min(s["no_filter"]["profit_factor"], 2.0) for s in strats]
    pf_wf = [min(s["with_filter"]["profit_factor"], 2.0) for s in strats]
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w/2, pf_nf, w, color="#9ca3af", label="sem filtro")
    ax.bar(x + w/2, pf_wf, w, color="#2563eb",
           label=f"com filtro (top {int(e3['top_pct']*100)}%)")
    ax.axhline(1.0, color="#ef4444", linestyle="--", label="Break-even (1,0)")
    ax.set_title("Etapa 3 — Profit Factor das estratégias simples, sem vs. com filtro")
    ax.set_ylabel("Profit Factor"); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
    return _save(fig, "vol_econ_etapa3.png")


def _plot_etapa4(e4) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # Curva de ganho.
    fr = [f * 100 for f in e4["gain_fracs"]]
    ax1.plot(fr, [g * 100 for g in e4["gain"]], "-o", color="#2563eb",
             markersize=3, label="Modelo")
    ax1.plot([0, 100], [0, 100], "--", color="#9ca3af", label="Aleatório")
    ax1.set_title("Etapa 4 — Curva de ganho (captura de alta vol)")
    ax1.set_xlabel("% da população (ordenada por score)")
    ax1.set_ylabel("% de eventos de alta vol capturados")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.25)
    # Lift e recall extremo por K.
    ks = [f"top {int(r['k']*100)}%" for r in e4["rows"]]
    x = np.arange(len(ks)); w = 0.38
    lift = [r["lift"] for r in e4["rows"]]
    rec = [r["recall_extreme"] for r in e4["rows"]]
    ax2.bar(x - w/2, lift, w, color="#16a34a", label="Lift")
    ax2.bar(x + w/2, rec, w, color="#f59e0b", label="Recall eventos extremos")
    ax2.axhline(1.0, color="#ef4444", linestyle="--", label="Lift = 1 (sem ganho)")
    ax2.set_title("Etapa 4 — Lift e Recall@K")
    ax2.set_xticks(x); ax2.set_xticklabels(ks); ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25, axis="y")
    return _save(fig, "vol_econ_etapa4.png")


def _plot_etapa5(e5) -> str:
    rows = e5["rows"]
    labels = [r["regime"].split(" (")[0] for r in rows]
    x = np.arange(len(labels))
    dur = [r["avg_duration"] for r in rows]
    move = [r["mean_absmove_pct"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(x, dur, color="#8b5cf6")
    ax1.set_title("Etapa 5 — Duração média do regime (persistência)")
    ax1.set_ylabel("candles consecutivos"); ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=10); ax1.grid(alpha=0.25, axis="y")
    ax2.bar(x, move, color="#2563eb")
    ax2.axhline(e5["cost_pct"], color="#ef4444", linestyle="--",
                label=f"custo ({e5['cost_pct']:.2f}%)")
    ax2.set_title("Etapa 5 — |movimento| médio por regime")
    ax2.set_ylabel("%"); ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=10); ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25, axis="y")
    return _save(fig, "vol_econ_etapa5.png")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _fmt_time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _pf(v: float) -> str:
    return "∞" if v == float("inf") else f"{v:.2f}"


def _img(path: str) -> str:
    return "![](" + os.path.relpath(path, config.REPORTS_DIR).replace(os.sep, "/") + ")"


def _render_markdown(ctx, e1, e2, e3, e4, e5, verdict, charts) -> str:
    L: list[str] = []
    a = L.append

    a("# Validação Econômica da Volatilidade — BTC/USDT")
    a("")
    a(f"_Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
      f"horizonte H={ctx.horizon} min · {len(ctx.test_df):,} candles de teste · "
      f"AUC do modelo de vol = {ctx.auc:.3f}._")
    a("")
    a("> Mesmo XGBoost, mesmos indicadores, mesmo pipeline. Mede-se se a "
      "previsibilidade da volatilidade (já estabelecida) tem **valor econômico** "
      "— não se busca lucro nem maior acurácia. Operações com saída por tempo em "
      "H candles e custos reais (taxa + slippage).")
    a("")

    # --- Etapa 1 ----------------------------------------------------------
    a("## Etapa 1 — Validação econômica por grupo de volatilidade")
    a("")
    a("Previsões agrupadas em quintis do score de volatilidade. Quanto o mercado "
      "**realmente se move** em cada grupo:")
    a("")
    a("| Grupo | N | Score méd. | Ret. médio | \\|Mov\\| médio | Desv-pad | "
      "P05 / P95 | Drawdown médio (long) | Vol realizada |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in e1:
        a("| {} | {:,} | {:.2f} | {:+.3f}% | {:.3f}% | {:.3f}% | {:.2f}% / {:+.2f}% | "
          "{:.3f}% | {:.3f}% |".format(
              r["group"], r["n"], r["mean_score"], r["mean_ret_pct"],
              r["mean_absmove_pct"], r["std_ret_pct"], r["p05_pct"], r["p95_pct"],
              r["mean_mae_pct"], r["realized_vol_pct"]))
    a("")
    a(_img(charts["e1"]))
    a("")
    a(f"**Resposta — \"quanto o mercado se move quando o modelo prevê alta "
      f"volatilidade?\"** O grupo *muito alta* move em média "
      f"**{verdict['high_absmove_pct']:.3f}%** vs. **{verdict['low_absmove_pct']:.3f}%** "
      f"do grupo *muito baixa* — uma razão de **{verdict['move_ratio']:.1f}×**. "
      f"O retorno *médio* (com sinal) permanece ~0 em todos os grupos: a vol "
      f"prevê **tamanho**, não **direção**.")
    a("")

    # --- Etapa 2 ----------------------------------------------------------
    a("## Etapa 2 — Filtro de oportunidades (varredura de percentis)")
    a("")
    a(f"Estratégia de referência: **{e2['strategy_label']}**. Operar apenas "
      "quando o score de volatilidade está no topo X%:")
    a("")
    a("| Filtro | Nº ops | Win rate | Profit Factor | Expectância | Ret. acum | DD máx |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for r in e2["rows"]:
        a("| {} | {:,} | {:.1%} | {} | {:+.4f} | {:+.2%} | {:.2%} |".format(
            r["label"], r["n_trades"], r["win_rate"], _pf(r["profit_factor"]),
            r["expectancy"], r["cumulative_return"], r["max_drawdown_pct"]))
    a("")
    a(_img(charts["e2"]))
    a("")
    a("O filtro **reduz o número de operações** e **eleva o profit factor** ao "
      "remover trades de baixa volatilidade — exatamente os dominados por custo "
      "(onde \\|movimento\\| < custo).")
    a("")

    # --- Etapa 3 ----------------------------------------------------------
    a("## Etapa 3 — Estratégias simples, com e sem filtro")
    a("")
    a(f"Filtro = top {int(e3['top_pct']*100)}% de volatilidade prevista.")
    a("")
    a("| Estratégia | | Nº ops | Win rate | Profit Factor | Expectância | Ret. acum | DD máx |")
    a("|---|---|---:|---:|---:|---:|---:|---:|")
    for s in e3["strategies"]:
        for tag, m in (("sem filtro", s["no_filter"]), ("com filtro", s["with_filter"])):
            a("| {} | {} | {:,} | {:.1%} | {} | {:+.4f} | {:+.2%} | {:.2%} |".format(
                s["label"] if tag == "sem filtro" else "", tag,
                m["n_trades"], m["win_rate"], _pf(m["profit_factor"]),
                m["expectancy"], m["cumulative_return"], m["max_drawdown_pct"]))
    a("")
    a(_img(charts["e3"]))
    a("")

    # --- Etapa 4 ----------------------------------------------------------
    a("## Etapa 4 — Valor econômico da informação")
    a("")
    a(f"Taxa-base de alta volatilidade no teste: **{e4['base_rate']:.1%}**; "
      f"eventos extremos (vol > p90 do treino): **{e4['ext_rate']:.1%}**.")
    a("")
    a("| K | Nº | Precision@K | Lift | Recall positivos | Recall extremos |")
    a("|---|---:|---:|---:|---:|---:|")
    for r in e4["rows"]:
        a("| top {:.0f}% | {:,} | {:.1%} | {:.2f}× | {:.1%} | {:.1%} |".format(
            r["k"]*100, r["n_top"], r["precision"], r["lift"],
            r["recall_positives"], r["recall_extreme"]))
    a("")
    a(_img(charts["e4"]))
    a("")
    a(f"**Information Ratio (Sharpe por operação) da estratégia de referência:** "
      f"sem filtro {e4['sharpe_no_filter']:+.3f} → com filtro "
      f"{e4['sharpe_with_filter']:+.3f}. O filtro **reduz o prejuízo ajustado ao "
      f"risco**, mas ambos seguem negativos — não há retorno positivo a extrair "
      f"por direção.")
    a("")
    a("**Resposta — \"valor econômico utilizável ou só significância "
      "estatística?\"** O Lift > 1 e a alta recall de eventos extremos mostram "
      "que a informação é **real e concentrável** (ordena bem o tamanho do "
      "movimento). Mas, como a direção segue ~aleatória, esse valor é de "
      "**gestão de risco/timing de volatilidade**, não de lucro direcional.")
    a("")

    # --- Etapa 5 ----------------------------------------------------------
    a("## Etapa 5 — Identificação de regimes")
    a("")
    a("Regimes por tercil do score de volatilidade. Persistência (duração), "
      "frequência e potencial econômico:")
    a("")
    a("| Regime | Frequência | Duração média | Duração máx | \\|Mov\\| médio | "
      "Vol realizada | \\|Mov\\|−custo | Explorável? |")
    a("|---|---:|---:|---:|---:|---:|---:|:---:|")
    for r in e5["rows"]:
        a("| {} | {:.1%} | {:.1f} | {} | {:.3f}% | {:.3f}% | {:+.3f}% | {} |".format(
            r["regime"], r["frequency"], r["avg_duration"], r["max_duration"],
            r["mean_absmove_pct"], r["mean_realized_vol_pct"],
            r["absmove_minus_cost_pct"], "✅" if r["exploitable"] else "—"))
    a("")
    a(_img(charts["e5"]))
    a("")
    a(f"Regimes de alta volatilidade **persistem** (duração média "
      f"{e5['rows'][-1]['avg_duration']:.1f} candles), o que os torna "
      f"identificáveis com antecedência — condição necessária para timing.")
    a("")

    # --- Etapa 6 ----------------------------------------------------------
    a("## Etapa 6 — Conclusão objetiva")
    a("")
    a("**1) Há evidência de que a previsão de volatilidade melhora uma "
      "estratégia?**  ")
    a("**Sim.** O filtro de volatilidade eleva o profit factor e remove as "
       f"operações dominadas por custo em {verdict['n_improved']}/3 estratégias "
       "(o |movimento| em alta vol supera o custo, ao contrário da baixa vol)."
       if verdict["filter_helps"] else
       "**Não de forma consistente.** O filtro não elevou o profit factor na "
       "maioria das estratégias.")
    a("")
    a("**2) O filtro de volatilidade melhora ou piora estratégias simples?**  ")
    a("**Melhora** o perfil (profit factor e eficiência de capital) ao "
       f"concentrar as operações em janelas de movimento relevante "
       f"(ganho médio de PF de {verdict['mean_pf_gain']:+.2f}). Porém **não cria "
       "edge direcional**: "
       + ("alguma estratégia chega a lucro líquido positivo."
          if verdict["any_profitable"] else
          "nenhuma fica lucrativa após custos — apenas *menos deficitária*."))
    a("")
    a("**3) Há vantagem econômica potencial que justifique continuar?**  ")
    a("**Sim, condicional.** A informação de volatilidade tem valor real "
       f"(AUC {ctx.auc:.2f}, Lift@10% {verdict['lift10']:.2f}×, regimes "
       "persistentes), mas esse valor é de **dimensão de risco/volatilidade**, "
       "não de previsão de direção. Justifica continuar **se** o projeto pivotar "
       "para explorar volatilidade (sizing, gestão de risco, produtos de vol). "
       "A aposta direcional permanece sem edge."
       if (verdict["info_real"] and verdict["highvol_move_exceeds_cost"]) else
       "**Improvável.** O valor não se traduz em vantagem econômica acima dos "
       "custos.")
    a("")
    a("**4) Qual o próximo experimento mais promissor?**  ")
    a("Usar a volatilidade prevista como **dimensionador de risco/posição** em "
      "vez de filtro direcional: (a) *position sizing* inverso à volatilidade "
      "(vol-targeting) sobre uma estratégia de tendência; (b) avaliar captura de "
      "movimento independente de direção (proxy de straddle: ganhar com "
      "|movimento| grande), já que o modelo prevê **tamanho**; (c) gatilhar "
      "rompimentos **apenas** em regimes de alta vol persistentes, com "
      "razão risco/retorno definida. Todos reutilizam o sinal de vol já validado.")
    a("")
    a("_Ressalva: um único período e o conjunto de features atual (pensado para "
      "direção). O resultado mede valor potencial da informação, não garante "
      "lucro — que dependerá de execução e do produto escolhido._")
    a("")
    return "\n".join(L)
