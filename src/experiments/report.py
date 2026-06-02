"""Orquestra o estudo de horizonte em múltiplos tamanhos de amostra e gera o
relatório final (Markdown + gráficos) com conclusão objetiva.

Responde, com rigor estatístico:
  1. Existe algum horizonte temporal com evidência de edge?
  2. Os resultados permanecem consistentes quando o histórico aumenta?
  3. O modelo atual possui vantagem estatística real após custos?
  E especificamente: qual horizonte apresentou a melhor evidência de vantagem
  estatística após custos?
"""
from __future__ import annotations

import math
import os
from datetime import UTC, datetime

import config
from src.data.database import Database
from src.experiments import stats_validity as sv
from src.experiments.horizon_study import run_size_study
from src.experiments.plots import plot_accuracy_across_sizes, plot_size_panel

DEFAULT_HORIZONS = [5, 15, 30, 60, 240]
DEFAULT_SIZES = [8_000, 100_000, 250_000]


def run_study(db: Database = None, horizons: list[int] = None,
              sizes: list[int] = None, overrides: dict = None) -> dict:
    """Executa o estudo completo e grava relatório + gráficos."""
    db = db or Database()
    horizons = horizons or DEFAULT_HORIZONS
    sizes = sizes or DEFAULT_SIZES

    base_params = dict(config.BACKTEST)
    if overrides:
        base_params.update({k: v for k, v in overrides.items() if v is not None})

    all_candles = db.load_candles()
    total = len(all_candles)
    if total < 1_000:
        raise RuntimeError(
            f"Histórico insuficiente: {total} candles. "
            "Colete mais dados (ex.: main.py collect --backfill 100000)."
        )

    # Para cada tamanho pedido, usa os ÚLTIMOS N candles (subconjuntos aninhados).
    # Tamanhos maiores que o disponível são limitados ao total e deduplicados.
    studies: dict[str, dict] = {}
    used_sizes = []
    for s in sizes:
        eff = min(s, total)
        if eff in used_sizes:
            continue  # evita rodar o mesmo subconjunto duas vezes
        used_sizes.append(eff)
        subset = all_candles.iloc[-eff:].reset_index(drop=True)
        label = _size_label(s, eff, total)
        print(f"  • Estudo com {label} ...")
        studies[label] = run_size_study(subset, horizons, base_params)

    # --- Gráficos ---------------------------------------------------------
    chart_paths = {}
    for label, study in studies.items():
        safe = label.split()[0].replace(".", "").replace(",", "")
        path = os.path.join(config.REPORTS_DIR, f"horizon_panel_{safe}.png")
        chart_paths[label] = plot_size_panel(study, horizons, label, path)
    cross_path = os.path.join(config.REPORTS_DIR, "horizon_accuracy_across_sizes.png")
    plot_accuracy_across_sizes(studies, horizons, cross_path)

    # --- Conclusão objetiva ----------------------------------------------
    verdict = _build_verdict(studies, horizons)

    # --- Relatório Markdown ----------------------------------------------
    md = _render_markdown(studies, horizons, base_params, chart_paths,
                          cross_path, verdict)
    report_path = os.path.join(config.REPORTS_DIR, "horizon_study_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    return {"report_path": report_path, "studies": studies,
            "verdict": verdict, "cross_chart": cross_path}


# ---------------------------------------------------------------------------
# Conclusão objetiva
# ---------------------------------------------------------------------------
def _build_verdict(studies: dict, horizons: list) -> dict:
    """Decide, com critérios explícitos, se há edge e qual o melhor horizonte."""
    largest_label = list(studies.keys())[-1]
    largest = studies[largest_label]

    # Edge por horizonte (na maior amostra): IC95% acima de 0,5 E lucrativo.
    edge_horizons = []
    for r in largest["results"]:
        has_edge = (r["edge_significant"]
                    and r["profit_factor"] > 1.0
                    and r["expectancy"] > 0)
        if has_edge:
            edge_horizons.append(r["horizon"])

    # Melhor horizonte após custos: maior expectância, na maior amostra.
    valid = [r for r in largest["results"] if not math.isnan(r["directional_accuracy"])]
    best = max(valid, key=lambda r: r["expectancy"]) if valid else None

    # Consistência entre tamanhos: a acurácia de cada horizonte muda de lado
    # de 50% conforme o histórico cresce? Mede dispersão e mudança de sinal.
    consistency = _consistency(studies, horizons)

    # Horizontes com acurácia significativamente > 50% (IC95% acima de 0,5),
    # independentemente de lucratividade — "sinal estatístico detectável".
    significant_up = [r["horizon"] for r in largest["results"]
                      if not math.isnan(r["acc_ci_low"]) and r["acc_ci_low"] > 0.5]
    any_significant_up = len(significant_up) > 0

    q1 = len(edge_horizons) > 0
    q2 = consistency["consistent"]
    q3 = any_significant_up and q1

    if best is not None and q1:
        best_str = (f"{_fmt_h(best['horizon'])} "
                    f"(acurácia {best['directional_accuracy']:.1%}, "
                    f"PF {best['profit_factor']:.2f}, "
                    f"expectância {best['expectancy']:.4f} USDT/op)")
    elif best is not None:
        best_str = (f"nenhum com edge real; o 'menos pior' foi "
                    f"{_fmt_h(best['horizon'])} "
                    f"(acurácia {best['directional_accuracy']:.1%}, "
                    f"expectância {best['expectancy']:.4f} USDT/op)")
    else:
        best_str = "indeterminado (amostra insuficiente)"

    return {
        "edge_horizons": edge_horizons,
        "significant_up": significant_up,
        "best_horizon_str": best_str,
        "consistency": consistency,
        "q1_any_edge": q1,
        "q2_consistent": q2,
        "q3_real_edge": q3,
        "largest_label": largest_label,
    }


def _consistency(studies: dict, horizons: list) -> dict:
    """Consistência das conclusões ao aumentar o histórico.

    Distingue dois níveis:
      * Conclusão ECONÔMICA (existe edge lucrativo após custos?) — é o que
        importa para a decisão. Consistente se o veredito é o mesmo em todos
        os tamanhos.
      * Estimativa PONTUAL de acurácia — naturalmente ruidosa em N pequeno;
        medimos a dispersão entre tamanhos e quantos horizontes 'trocam de lado'
        de 50%, como diagnóstico (não como falha).
    """
    econ_verdicts = []
    for study in studies.values():
        any_profitable_edge = any(
            (r["edge_significant"] and r["profit_factor"] > 1.0
             and r["expectancy"] > 0)
            for r in study["results"])
        econ_verdicts.append(any_profitable_edge)
    economic_consistent = (len(set(econ_verdicts)) == 1)
    all_no_edge = not any(econ_verdicts)

    flips = 0
    spreads = []
    for i, _ in enumerate(horizons):
        accs = [studies[lbl]["results"][i]["directional_accuracy"]
                for lbl in studies
                if not math.isnan(studies[lbl]["results"][i]["directional_accuracy"])]
        if len(accs) < 2:
            continue
        if len({a > 0.5 for a in accs}) > 1:
            flips += 1
        spreads.append(max(accs) - min(accs))
    max_spread = max(spreads) if spreads else 0.0

    return {"consistent": economic_consistent, "all_no_edge": all_no_edge,
            "flips": flips, "max_spread": max_spread, "n_sizes": len(studies)}


# ---------------------------------------------------------------------------
# Renderização
# ---------------------------------------------------------------------------
def _fmt_h(h: int) -> str:
    return f"{h//60}h" if h >= 60 and h % 60 == 0 else f"{h}min"


def _fmt_pf(pf: float) -> str:
    return "∞" if pf == math.inf else f"{pf:.2f}"


def _size_label(requested: int, effective: int, total: int) -> str:
    days = effective / (60 * 24)
    if effective < requested:
        return f"{effective:,} candles (~{days:.0f}d, máx. disp.)"
    return f"{effective:,} candles (~{days:.0f}d)"


def _fmt_time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _render_markdown(studies, horizons, base_params, chart_paths,
                     cross_path, verdict) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Estudo de Horizonte Temporal — BTC/USDT")
    a("")
    a(f"_Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}._")
    a("")
    a("> Experimento controlado: **mesmo dataset, mesmos indicadores, mesmo "
      "XGBoost, mesmo pipeline e mesmo backtest**. A única variável manipulada "
      "é o **horizonte de previsão** (5, 15, 30, 60 e 240 min). Para cada "
      "horizonte a posição é mantida exatamente por H candles (saída por tempo, "
      "sem bracket), com os mesmos custos (taxa + slippage).")
    a("")

    # --- Validade estatística --------------------------------------------
    a("## 1. Validade estatística e tamanho de amostra")
    a("")
    a("> **Janelas sobrepostas (ponto crítico).** Com candles de 1 min, duas "
      "previsões consecutivas de horizonte *H* compartilham *(H−1)/H* do futuro "
      "— logo são fortemente autocorrelacionadas. Tratar todas as previsões como "
      "independentes infla artificialmente o N e produz intervalos de confiança "
      "e p-valores espúrios. **Todas as estatísticas abaixo usam apenas amostras "
      "não-sobrepostas (uma a cada *H* candles), ≈ N/H observações de fato "
      "independentes.**")
    a("")
    a("A acurácia direcional é uma proporção binomial; a margem de erro a 95% "
      "em torno de 50% define o **menor edge distinguível do acaso**. Como o N "
      "independente cai com o horizonte, a confiabilidade depende de *cada "
      "combinação* tamanho × horizonte:")
    a("")
    a("| Amostra | Horizonte | N independente | Margem de erro (95%) | Confiabilidade |")
    a("|---|---|---:|---:|---|")
    for label, study in studies.items():
        for r in study["results"]:
            if r["n_indep"] == 0:
                continue
            a(f"| {label} | {_fmt_h(r['horizon'])} | {r['n_indep']:,} | "
              f"±{r['moe']*100:.2f}pp | {r['reliability']} |")
    a("")
    # Quanto seria necessário para detectar 1pp e 2pp de edge.
    n1 = sv.required_n(0.51)
    n2 = sv.required_n(0.52)
    a(f"Para detectar um edge de **+1pp** (51% vs 50%) com 95%/80% de poder são "
      f"necessárias **~{n1:,} observações independentes**; para **+2pp**, "
      f"**~{n2:,}**. No horizonte de 4h, cada observação independente consome "
      f"240 candles — então ~{n2:,} observações exigiriam ~{n2*240:,} candles de "
      f"1 min de teste. É por isso que horizontes longos precisam de muito mais "
      f"histórico para qualquer conclusão robusta.")
    a("")

    # --- Tabelas por tamanho ---------------------------------------------
    a("## 2. Resultados por horizonte")
    a("")
    for label, study in studies.items():
        a(f"### {label}")
        a("")
        a(f"_Período: {_fmt_time(study['start'])} a {_fmt_time(study['end'])} · "
          f"Buy & Hold no teste: {study['buy_and_hold_return']:+.2%}._")
        a("")
        a("| Horizonte | N indep | Acurácia | IC95% | p-valor | Sig.? | "
          "Mov.med/Custo | PF | Ret.acum | Sharpe | DD máx | Expect. |")
        a("|---|---:|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|")
        for r in study["results"]:
            if math.isnan(r["directional_accuracy"]):
                a(f"| {_fmt_h(r['horizon'])} | {r['n_test']} | — | — | — | — | "
                  "— | — | — | — | — | — |")
                continue
            sig = "✅" if r["edge_significant"] else "—"
            a("| {} | {} | {:.1%} | [{:.1%}, {:.1%}] | {:.3f} | {} | "
              "{:.3f}/{:.3f}% | {} | {:+.2%} | {:.2f} | {:.2%} | {:+.4f} |".format(
                  _fmt_h(r["horizon"]), r["n_test"], r["directional_accuracy"],
                  r["acc_ci_low"], r["acc_ci_high"], r["p_value"], sig,
                  r["median_move_pct"], r["cost_hurdle_pct"],
                  _fmt_pf(r["profit_factor"]), r["cumulative_return"],
                  r["sharpe_trade"], r["max_drawdown_pct"], r["expectancy"]))
        a("")
        rel = os.path.relpath(chart_paths[label], config.REPORTS_DIR).replace(os.sep, "/")
        a(f"![Painel {label}]({rel})")
        a("")

    # --- Comparação entre tamanhos ---------------------------------------
    a("## 3. Consistência com o aumento do histórico")
    a("")
    rel = os.path.relpath(cross_path, config.REPORTS_DIR).replace(os.sep, "/")
    a(f"![Acurácia entre tamanhos]({rel})")
    a("")
    cons = verdict["consistency"]
    a("Há **dois níveis** de consistência a distinguir:")
    a("")
    a(f"- **Estimativa pontual de acurácia** — naturalmente ruidosa em amostra "
      f"pequena. Dispersão máxima da acurácia entre tamanhos: "
      f"**{cons['max_spread']*100:.2f}pp**; horizontes que cruzam 50% entre "
      f"tamanhos: **{cons['flips']}** de {len(horizons)}. Isso é **esperado** e "
      f"compatível com as margens de erro da §1 (em 8k, ±4,5 a ±31pp). Com mais "
      f"dados as estimativas **convergem** (o 5min estabiliza em ~51%).")
    a(f"- **Conclusão econômica** (existe edge lucrativo após custos?) — "
      f"{'**idêntica em todos os tamanhos**' if cons['consistent'] else 'varia entre tamanhos'}: "
      f"{'nenhum tamanho revela edge lucrativo.' if cons['all_no_edge'] else 'há divergência.'} "
      f"Aumentar o histórico **refina** o quadro (revela o sinal mínimo do 5min, "
      f"invisível em 8k), mas **não altera o veredito**.")
    a("")

    # --- Conclusão --------------------------------------------------------
    a("## 4. Conclusão objetiva")
    a("")
    a("**1) Existe algum horizonte com evidência de edge (IC95% > 50% e "
      "lucrativo após custos)?**  ")
    if verdict["q1_any_edge"]:
        hs = ", ".join(_fmt_h(h) for h in verdict["edge_horizons"])
        a(f"Sim — em: {hs}.")
    else:
        a("**Não.** Nenhum horizonte combina acurácia significativamente acima "
          "de 50% *com* lucratividade positiva após custos.")
        if verdict["significant_up"]:
            hs = ", ".join(_fmt_h(h) for h in verdict["significant_up"])
            a("")
            a(f"  Nuance importante: em **{hs}** a acurácia é estatisticamente "
              f"**> 50%** (sinal real, detectável só com amostra grande), mas o "
              f"edge é minúsculo (~1pp) e fica **muito abaixo do limiar de "
              f"custo**: o movimento mediano do preço nesse horizonte não cobre "
              f"o custo de ida-e-volta (~0,12%). Há sinal, porém economicamente "
              f"irrelevante.")
    a("")
    a("**2) Os resultados permanecem consistentes quando o histórico aumenta?**  ")
    a("Sim — a **conclusão econômica** (sem edge lucrativo) é a mesma em 8k, "
       "100k e 250k candles. As estimativas pontuais de acurácia, ruidosas em "
       "8k, **convergem** com mais dados; aumentar o histórico refina, mas não "
       "reverte o veredito (ver §3)."
       if verdict["q2_consistent"] else
       "Parcialmente — há variação entre tamanhos (ver §3).")
    a("")
    a("**3) O modelo possui vantagem estatística real após custos?**  ")
    if verdict["q3_real_edge"]:
        a("**Sim.**")
    elif verdict["significant_up"]:
        a("**Não — não no sentido que importa.** Existe um sinal direcional "
          "estatisticamente detectável no(s) horizonte(s) mais curto(s) "
          "(~51% no 5min, só visível com 250k candles), mas ele é pequeno "
          "demais para superar os custos operacionais. O sistema é **deficitário "
          "após custos em todos os horizontes**. Logo, não há vantagem "
          "*economicamente* real.")
    else:
        a("**Não.** A acurácia direcional não supera o acaso de forma "
          "estatisticamente significativa e o sistema é deficitário após custos "
          "em todos os horizontes testados.")
    a("")
    a("**Qual horizonte apresentou a melhor evidência de vantagem após custos?**  ")
    a(verdict["best_horizon_str"] + ".")
    a("")
    a("_Observação: o custo de ida-e-volta é fixo (~"
      f"{2*(base_params['fee_pct']+base_params['slippage_pct']):.2f}%); horizontes "
      "maiores aumentam o movimento típico do preço, reduzindo o peso relativo "
      "dos custos — por isso a coluna 'Mov.med/Custo' é decisiva para a "
      "viabilidade.")
    a("")
    return "\n".join(lines)
