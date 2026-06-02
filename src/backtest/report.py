"""Orquestração do backtest e geração do relatório final.

Fluxo:
  1. Carrega candles do SQLite e calcula indicadores.
  2. Split temporal in-sample / out-of-sample (sem shuffle).
  3. Treina o MESMO XGBoost no in-sample e gera sinais no out-of-sample.
  4. Simula o modelo e cada baseline pelo mesmo motor (modo sequencial).
  5. Análise por faixa de confiança (modo independente, amostra cheia).
  6. Curva de capital (PNG) + relatório Markdown + conclusão objetiva.

A conclusão responde, de forma programática e objetiva:
  "O modelo atual apresenta vantagem estatística suficiente para justificar
   mais desenvolvimento?"
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

import config
from src.backtest import strategies
from src.backtest.confidence import analyze_confidence, monotonicity_score
from src.backtest.engine import BacktestConfig, simulate_independent, simulate_sequential
from src.backtest.metrics import compute_metrics
from src.backtest.plotting import plot_equity_curve
from src.data.database import Database
from src.features.indicators import add_indicators


def run_backtest(db: Database = None, overrides: dict = None) -> dict:
    """Executa toda a validação e grava os artefatos. Retorna um resumo."""
    db = db or Database()
    params = dict(config.BACKTEST)
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    cfg = BacktestConfig.from_dict(params)

    candles = db.load_candles()
    if len(candles) < 500:
        raise RuntimeError(
            f"Histórico insuficiente para backtest: {len(candles)} candles. "
            "Colete mais dados (ex.: main.py collect --backfill 5000)."
        )

    enriched = add_indicators(candles).reset_index(drop=True)

    # --- Split temporal in-sample / out-of-sample -------------------------
    split = int(len(enriched) * (1.0 - cfg.test_fraction))
    train_df = enriched.iloc[:split].reset_index(drop=True)
    test_df = enriched.iloc[split:].reset_index(drop=True)

    # --- Modelo: treina no in-sample, opera no out-of-sample --------------
    model = strategies.train_model_in_sample(train_df)
    model_sig, model_conf = strategies.model_signals(model, test_df)

    # --- Sinais dos baselines (mesmo período de teste) --------------------
    rnd_sig, rnd_conf = strategies.random_signals(test_df, cfg.random_seed)
    long_sig, long_conf = strategies.always_long_signals(test_df)
    short_sig, short_conf = strategies.always_short_signals(test_df)
    ema_sig, ema_conf = strategies.ema_cross_signals(test_df)

    # --- Simulação sequencial (conta realista) p/ todas estratégias -------
    strategies_signals = {
        "Modelo (IA)": (model_sig, model_conf),
        "Aleatório (cara/coroa)": (rnd_sig, rnd_conf),
        "Sempre comprado": (long_sig, long_conf),
        "Sempre vendido": (short_sig, short_conf),
        "Cruzamento EMA9/21": (ema_sig, ema_conf),
    }

    results: dict[str, dict] = {}
    model_trades = None
    for name, (sig, conf) in strategies_signals.items():
        # Baselines não devem ser filtrados por confiança (sempre 0.5);
        # o filtro min_confidence só faz sentido para o modelo.
        eff_cfg = cfg
        if name != "Modelo (IA)":
            eff_cfg = _clone_cfg(cfg, min_confidence=0.0)
        trades = simulate_sequential(test_df, sig, conf, eff_cfg)
        results[name] = compute_metrics(trades, cfg)
        if name == "Modelo (IA)":
            model_trades = trades

    # --- Análise por faixa de confiança (modo independente) ---------------
    indep_trades = simulate_independent(test_df, model_sig, model_conf, cfg)
    confidence_rows = analyze_confidence(indep_trades)
    mono = monotonicity_score(confidence_rows)

    # --- Curva de capital do modelo ---------------------------------------
    chart_path = plot_equity_curve(model_trades or [], cfg)

    # --- Conclusão objetiva ------------------------------------------------
    verdict = _build_verdict(results, confidence_rows, mono)

    # --- Relatório Markdown ------------------------------------------------
    period = {
        "n_candles": len(enriched),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "start": _fmt_time(int(test_df["open_time"].iloc[0])),
        "end": _fmt_time(int(test_df["open_time"].iloc[-1])),
    }
    report_md = _render_markdown(cfg, params, period, results,
                                 confidence_rows, mono, verdict, chart_path)
    report_path = os.path.join(config.REPORTS_DIR, "backtest_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return {
        "report_path": report_path,
        "chart_path": chart_path,
        "results": results,
        "confidence_rows": confidence_rows,
        "monotonicity": mono,
        "verdict": verdict,
        "period": period,
    }


def _clone_cfg(cfg: BacktestConfig, **changes) -> BacktestConfig:
    data = cfg.__dict__.copy()
    data.update(changes)
    return BacktestConfig(**data)


# ---------------------------------------------------------------------------
# Conclusão objetiva (decisão baseada em critérios explícitos)
# ---------------------------------------------------------------------------
def _build_verdict(results: dict, confidence_rows: list, mono: float) -> dict:
    """Aplica critérios objetivos e devolve verdict + justificativas."""
    model = results["Modelo (IA)"]
    baselines = {k: v for k, v in results.items() if k != "Modelo (IA)"}

    checks = []

    # Critério 1: expectância positiva por operação (após custos).
    c1 = model["expectancy"] > 0
    checks.append(("Expectância por operação positiva (após custos)", c1,
                   f"{model['expectancy']:.4f} USDT/op"))

    # Critério 2: profit factor com margem mínima.
    c2 = model["profit_factor"] > 1.1
    checks.append(("Profit Factor > 1,10", c2,
                   f"{model['profit_factor']:.3f}"))

    # Critério 3: acurácia direcional acima do acaso com margem (> 52%).
    c3 = model["directional_accuracy"] > 0.52
    checks.append(("Acurácia direcional > 52% (skill acima do acaso)", c3,
                   f"{model['directional_accuracy']:.1%}"))

    # Critério 4: bate o melhor baseline em expectância.
    best_base_name = max(baselines, key=lambda k: baselines[k]["expectancy"])
    best_base_exp = baselines[best_base_name]["expectancy"]
    c4 = model["expectancy"] > best_base_exp
    checks.append((f"Supera o melhor baseline em expectância ({best_base_name})",
                   c4, f"modelo {model['expectancy']:.4f} vs {best_base_exp:.4f}"))

    # Critério 5: confiança maior -> maior acerto direcional (monotonicidade).
    c5 = mono > 0.3
    checks.append(("Confiança maior associada a maior acerto direcional "
                   "(Spearman > 0,3)", c5, f"rho = {mono:.2f}"))

    passed = sum(1 for _, ok, _ in checks if ok)

    if passed >= 4:
        decision = "SIM"
        summary = ("O sistema apresenta evidências consistentes de vantagem "
                   "estatística fora da amostra e justifica mais desenvolvimento.")
    elif passed >= 2:
        decision = "INCONCLUSIVO"
        summary = ("Há sinais fracos/mistos de vantagem. Não há base sólida "
                   "para justificar mais desenvolvimento sem antes endereçar "
                   "os critérios não atendidos (ex.: custos, calibração).")
    else:
        decision = "NÃO"
        summary = ("O sistema não demonstra vantagem estatística suficiente "
                   "sobre baselines triviais após custos. Mais desenvolvimento "
                   "no formato atual não se justifica.")

    return {
        "decision": decision,
        "passed": passed,
        "total": len(checks),
        "checks": checks,
        "summary": summary,
        "best_baseline": best_base_name,
    }


# ---------------------------------------------------------------------------
# Renderização Markdown
# ---------------------------------------------------------------------------
def _fmt_time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_pf(pf: float) -> str:
    return "∞" if pf == float("inf") else f"{pf:.3f}"


def _render_markdown(cfg, params, period, results, confidence_rows, mono,
                     verdict, chart_path) -> str:
    lines: list[str] = []
    a = lines.append

    a(f"# Relatório de Validação — BTC/USDT (+{config.HORIZON_MINUTES} min)")
    a("")
    a(f"_Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
      f"out-of-sample de {period['start']} a {period['end']}._")
    a("")
    a("> Validação científica do sistema **atual** (sem novos indicadores, "
      "modelos ou fontes de dados). O modelo é treinado **apenas no in-sample** "
      "e avaliado **out-of-sample**; todos os baselines passam pelo mesmo motor "
      "de simulação (mesmos SL/TP, taxa e slippage).")
    a("")

    # --- Configuração -----------------------------------------------------
    a("## 1. Configuração do backtest")
    a("")
    a("| Parâmetro | Valor |")
    a("|---|---|")
    a(f"| Capital inicial | {cfg.initial_capital:,.2f} USDT |")
    a(f"| Valor por operação | {cfg.value_per_trade:,.2f} USDT |")
    a(f"| Stop Loss | {params['stop_loss_pct']:.3f}% |")
    a(f"| Take Profit | {params['take_profit_pct']:.3f}% |")
    a(f"| Taxa da corretora (por lado) | {params['fee_pct']:.3f}% |")
    a(f"| Slippage (por execução) | {params['slippage_pct']:.3f}% |")
    a(f"| Confiança mínima (modelo) | {cfg.min_confidence:.0%} |")
    a(f"| Horizonte / holding máx. | {cfg.max_holding} candles |")
    a(f"| Candles totais | {period['n_candles']} "
      f"(treino {period['n_train']} / teste {period['n_test']}) |")
    a("")

    # --- Métricas comparativas -------------------------------------------
    a("## 2. Métricas de trading — Modelo vs Baselines")
    a("")
    a("| Estratégia | Ops | Acerto dir. | Win rate líq. | Profit Factor | "
      "Expectância (USDT/op) | Sharpe/op | Retorno acum. | Drawdown máx. |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    # Modelo primeiro, baselines depois.
    order = ["Modelo (IA)", "Aleatório (cara/coroa)", "Sempre comprado",
             "Sempre vendido", "Cruzamento EMA9/21"]
    for name in order:
        m = results[name]
        a("| {} | {} | {:.1%} | {:.1%} | {} | {:.4f} | {:.3f} | {:.2%} | {:.2%} |".format(
            name, m["n_trades"], m["directional_accuracy"], m["win_rate"],
            _fmt_pf(m["profit_factor"]), m["expectancy"], m["sharpe_trade"],
            m["cumulative_return"], m["max_drawdown_pct"]))
    a("")
    a("> **Acerto dir.** = preço foi na direção prevista (skill do sinal, sem "
      "custos). **Win rate líq.** = operações lucrativas após taxa e slippage. "
      "A diferença entre as duas colunas é o quanto os custos corroem o sinal.")
    a("")
    a("Detalhe de ganhos/perdas do modelo:")
    m = results["Modelo (IA)"]
    a("")
    a("| Métrica | Valor |")
    a("|---|---|")
    a(f"| Ganho médio (operações vencedoras) | {m['avg_win']:.4f} USDT |")
    a(f"| Perda média (operações perdedoras) | {m['avg_loss']:.4f} USDT |")
    a(f"| Lucro bruto / Prejuízo bruto | {m['gross_profit']:.2f} / {m['gross_loss']:.2f} USDT |")
    a(f"| PnL líquido total | {m['total_pnl']:.2f} USDT |")
    a(f"| Saldo final | {m['final_equity']:,.2f} USDT |")
    a(f"| Saídas (SL / TP / Tempo) | {m['exit_reasons']['SL']} / "
      f"{m['exit_reasons']['TP']} / {m['exit_reasons']['TIME']} |")
    a("")

    # --- Faixas de confiança ---------------------------------------------
    a("## 3. Análise por faixa de confiança (modelo)")
    a("")
    a("Operações independentes por sinal (amostra cheia), para avaliar se "
      "convicção maior produz resultado melhor.")
    a("")
    a("| Faixa | Operações | Acerto direcional | Win rate líq. | Lucro líquido (USDT) |")
    a("|---|---:|---:|---:|---:|")
    for r in confidence_rows:
        da = "—" if r["directional_accuracy"] is None else f"{r['directional_accuracy']:.1%}"
        wr = "—" if r["win_rate"] is None else f"{r['win_rate']:.1%}"
        a(f"| {r['bucket']} | {r['n_trades']} | {da} | {wr} | {r['net_profit']:.2f} |")
    a("")
    a(f"**Monotonicidade (Spearman faixa × acerto direcional):** rho = {mono:.2f} "
      f"— {'tendência positiva' if mono > 0.3 else 'sem tendência clara' if mono > -0.3 else 'tendência negativa'}.")
    a("")

    # --- Curva de capital -------------------------------------------------
    a("## 4. Curva de capital")
    a("")
    # Caminho relativo à pasta do próprio relatório (que vive em REPORTS_DIR).
    rel = os.path.relpath(chart_path, config.REPORTS_DIR).replace(os.sep, "/")
    a(f"![Curva de capital]({rel})")
    a("")
    a("Saldo (azul), pico vigente (tracejado), drawdowns (vermelho) e lucro "
      "acumulado (verde, eixo à direita).")
    a("")

    # --- Conclusão --------------------------------------------------------
    a("## 5. Conclusão objetiva")
    a("")
    a("**Pergunta:** o modelo atual apresenta vantagem estatística suficiente "
      "para justificar mais desenvolvimento?")
    a("")
    a(f"**Resposta: {verdict['decision']}** "
      f"({verdict['passed']}/{verdict['total']} critérios atendidos).")
    a("")
    a("| Critério | Resultado | Atende? |")
    a("|---|---|:---:|")
    for desc, ok, detail in verdict["checks"]:
        a(f"| {desc} | {detail} | {'✅' if ok else '❌'} |")
    a("")
    a(verdict["summary"])
    a("")
    a("_Ressalva metodológica: resultado de um único split temporal e de um "
      "período de mercado específico. Custos (taxa+slippage) penalizam fortemente "
      "estratégias de alta frequência — exatamente o efeito que este backtest "
      "busca expor._")
    a("")
    return "\n".join(lines)
