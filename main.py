"""CLI orquestradora da aplicação BTC/USDT.

Comandos:
  collect   Coleta candles da Binance e grava no SQLite.
            --backfill N  busca N candles históricos de uma vez (paginado).
  train     Treina o XGBoost com o histórico armazenado.
  predict   Gera e registra uma previsão para os próximos minutos.
  evaluate  Verifica previsões pendentes e imprime a taxa de acerto.
  serve     Sobe a interface web Flask.
  loop      Executa coleta + previsão + avaliação em ciclo contínuo.
  backtest  Valida o sistema out-of-sample e gera relatório + gráficos.

Exemplos:
  python main.py collect --backfill 5000
  python main.py train
  python main.py serve
  python main.py loop --interval 60
  python main.py backtest --sl 0.3 --tp 0.3 --min-confidence 0.6
"""
from __future__ import annotations

import argparse
import time

import config
from src.data.binance_client import BinanceClient
from src.data.database import Database
from src.model.predictor import Predictor
from src.model.trainer import train as train_model
from src.tracking.evaluator import Evaluator


def cmd_collect(args) -> None:
    db = Database()
    client = BinanceClient()
    if args.backfill:
        print(f"Buscando {args.backfill} candles históricos (paginado)...")
        candles = client.backfill(args.backfill)
    else:
        candles = client.fetch_recent(limit=config.FETCH_LIMIT)
    new = db.upsert_candles(candles)
    print(f"Coletados {len(candles)} candles · {new} novos · "
          f"total no banco: {db.candle_count()}")


def cmd_train(args) -> None:
    print("Treinando XGBoost...")
    metrics = train_model(valid_fraction=args.valid_fraction)
    print("Treino concluído:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Modelo salvo em: {config.MODEL_PATH}")


def cmd_predict(args) -> None:
    result = Predictor().predict_and_log(refresh=not args.no_refresh)
    print(f"Previsão #{result['id']}: {result['direction']} "
          f"(alta={result['prob_up']:.1%}, baixa={result['prob_down']:.1%}) "
          f"· preço base ${result['price_at_prediction']:,.2f} "
          f"· horizonte {result['horizon_minutes']} min")


def cmd_evaluate(args) -> None:
    stats = Evaluator().evaluate_pending(refresh=not args.no_refresh)
    print(f"Avaliadas agora: {stats['newly_evaluated']} · "
          f"total avaliado: {stats['evaluated']} · "
          f"acertos: {stats['hits']} · "
          f"taxa de acerto: {stats['accuracy']:.1%}")


def cmd_serve(args) -> None:
    from src.web.app import create_app
    print(f"Servindo em http://{args.host}:{args.port}")
    create_app().run(host=args.host, port=args.port, debug=args.debug)


def cmd_loop(args) -> None:
    """Ciclo contínuo: coleta -> avalia pendentes -> nova previsão."""
    db = Database()
    client = BinanceClient()
    predictor = Predictor(db=db)
    evaluator = Evaluator(db=db)
    print(f"Loop iniciado (intervalo {args.interval}s). Ctrl+C para parar.")
    try:
        while True:
            db.upsert_candles(client.fetch_recent(limit=config.FETCH_LIMIT))
            stats = evaluator.evaluate_pending(refresh=False)
            pred = predictor.predict_and_log(refresh=False)
            print(f"[{time.strftime('%H:%M:%S')}] {pred['direction']} "
                  f"alta={pred['prob_up']:.1%} · taxa de acerto "
                  f"{stats['accuracy']:.1%} ({stats['evaluated']} avaliadas)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nLoop encerrado.")


def cmd_backtest(args) -> None:
    from src.backtest.report import run_backtest
    overrides = {
        "stop_loss_pct": args.sl,
        "take_profit_pct": args.tp,
        "value_per_trade": args.value,
        "fee_pct": args.fee,
        "slippage_pct": args.slippage,
        "min_confidence": args.min_confidence,
        "test_fraction": args.test_fraction,
    }
    print("Rodando backtest out-of-sample (treina no in-sample, opera no teste)...")
    summary = run_backtest(overrides=overrides)

    v = summary["verdict"]
    p = summary["period"]
    print(f"\nPeríodo de teste: {p['start']} a {p['end']} "
          f"({p['n_test']} candles)")
    print("\n--- Métricas por estratégia ---")
    print(f"{'Estratégia':<26}{'Ops':>6}{'AcDir':>8}{'WinLiq':>8}{'PF':>8}"
          f"{'Expect.':>11}{'Ret.acum':>11}")
    for name, m in summary["results"].items():
        pf = "inf" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"
        print(f"{name:<26}{m['n_trades']:>6}{m['directional_accuracy']:>7.1%}"
              f"{m['win_rate']:>8.1%}{pf:>8}"
              f"{m['expectancy']:>11.4f}{m['cumulative_return']:>10.2%}")

    print("\n--- Faixas de confiança (modelo) ---")
    for r in summary["confidence_rows"]:
        da = "—" if r["directional_accuracy"] is None else f"{r['directional_accuracy']:.1%}"
        print(f"  {r['bucket']:<9} ops={r['n_trades']:<5} acerto_dir={da:<7} "
              f"lucro={r['net_profit']:.2f}")

    print(f"\nMonotonicidade confiança×acerto (Spearman): {summary['monotonicity']:.2f}")
    print(f"\n=== CONCLUSÃO: {v['decision']} "
          f"({v['passed']}/{v['total']} critérios) ===")
    print(v["summary"])
    print(f"\nRelatório: {summary['report_path']}")
    print(f"Gráfico:   {summary['chart_path']}")


def cmd_experiment(args) -> None:
    from src.experiments.report import (DEFAULT_HORIZONS, DEFAULT_SIZES,
                                        run_study)
    horizons = ([int(x) for x in args.horizons.split(",")]
                if args.horizons else DEFAULT_HORIZONS)
    sizes = ([int(x) for x in args.sizes.split(",")]
             if args.sizes else DEFAULT_SIZES)
    print("Estudo de horizonte (mesmo dataset/indicadores/modelo/backtest)...")
    summary = run_study(horizons=horizons, sizes=sizes)

    v = summary["verdict"]
    print("\n--- Resumo por horizonte (maior amostra) ---")
    largest = summary["studies"][v["largest_label"]]
    print(f"{'Horiz.':>7}{'NTest':>8}{'Acur.':>8}{'IC-low':>8}{'p':>8}"
          f"{'PF':>7}{'Ret':>9}{'Expect':>10}")
    for r in largest["results"]:
        if r["directional_accuracy"] != r["directional_accuracy"]:  # NaN
            continue
        pf = "inf" if r["profit_factor"] == float("inf") else f"{r['profit_factor']:.2f}"
        print(f"{r['horizon']:>6}m{r['n_test']:>8}{r['directional_accuracy']:>7.1%}"
              f"{r['acc_ci_low']:>8.1%}{r['p_value']:>8.3f}{pf:>7}"
              f"{r['cumulative_return']:>8.1%}{r['expectancy']:>10.4f}")

    print(f"\n1) Algum horizonte com edge? {'SIM' if v['q1_any_edge'] else 'NÃO'}")
    print(f"2) Consistente entre tamanhos? {'SIM' if v['q2_consistent'] else 'PARCIAL/NÃO'}")
    print(f"3) Vantagem real após custos?  {'SIM' if v['q3_real_edge'] else 'NÃO'}")
    print(f"Melhor horizonte: {v['best_horizon_str']}")
    print(f"\nRelatório: {summary['report_path']}")


def cmd_targets(args) -> None:
    from src.experiments.target_report import DEFAULT_HORIZONS, run_study
    horizons = ([int(x) for x in args.horizons.split(",")]
                if args.horizons else DEFAULT_HORIZONS)
    print("Estudo comparativo de variáveis-alvo (mesmo modelo/indicadores)...")
    summary = run_study(horizons=horizons, n_candles=args.candles)

    v = summary["verdict"]
    print("\n--- Ranking de previsibilidade (AUC médio entre horizontes) ---")
    print(f"{'#':>2}  {'Alvo':<14}{'AUC':>7}{'SkillAcc':>10}{'SepEcon':>9}"
          f"{'Estab':>8}{'Signif':>8}")
    for i, (t, m) in enumerate(v["ranking"], start=1):
        print(f"{i:>2}  {t:<14}{m['mean_auc']:>7.3f}{m['mean_acc_skill']:>9.1%}"
              f"{m['mean_econ_sep']:>8.3f}%{m['mean_auc_std']:>8.3f}"
              f"{m['n_significant']:>5}/{m['n_horizons']}")
    print(f"\nMais previsível que direção: {'SIM' if v['has_better_target'] else 'NÃO'}")
    print(f"Melhor alvo: {v['best_target']} (AUC {v['best_metrics']['mean_auc']:.3f} "
          f"vs direção {v['direction_auc']:.3f})")
    print(f"\nRelatório: {summary['report_path']}")


def cmd_vol_economics(args) -> None:
    from src.economics.report import run_study
    print("Validação econômica da volatilidade (mesmo modelo/indicadores)...")
    summary = run_study(horizon=args.horizon, n_candles=args.candles,
                        filter_pct=args.filter_pct)
    v = summary["verdict"]
    print(f"\nAUC modelo de vol: {summary['ctx_auc']:.3f}")
    print(f"Razão |mov| alta/baixa vol: {v['move_ratio']:.1f}x "
          f"(alta {v['high_absmove_pct']:.3f}% vs custo {v['cost_pct']:.2f}%)")
    print(f"Lift@10%: {v['lift10']:.2f}x · regime alta vol persiste: "
          f"{'sim' if v['persistent_highvol'] else 'não'}")
    print("\n--- Etapa 3: estratégias (PF sem -> com filtro) ---")
    for s in summary["etapa3"]["strategies"]:
        nf, wf = s["no_filter"], s["with_filter"]
        pfn = "inf" if nf["profit_factor"] == float("inf") else f"{nf['profit_factor']:.2f}"
        pfw = "inf" if wf["profit_factor"] == float("inf") else f"{wf['profit_factor']:.2f}"
        print(f"  {s['key']:<12} PF {pfn:>5} -> {pfw:>5} | "
              f"exp {nf['expectancy']:+.3f} -> {wf['expectancy']:+.3f}")
    print(f"\n1) Vol melhora estratégia? {'SIM' if v['filter_helps'] else 'NÃO'}")
    print(f"2) Filtro melhora/piora?   {'MELHORA' if v['filter_helps'] else 'NÃO MELHORA'}")
    print(f"3) Vantagem econômica?     "
          f"{'POTENCIAL (dimensão de risco/vol)' if v['info_real'] and v['highvol_move_exceeds_cost'] else 'IMPROVÁVEL'}")
    print(f"\nRelatório: {summary['report_path']}")


def cmd_validate(args) -> None:
    from src.validation.report import run_validation
    print("Validação anti-overfitting (CV purgada + Deflated Sharpe)...")
    s = run_validation(horizon=args.horizon, n_candles=args.candles,
                       n_splits=args.splits)
    cv = s["cv"]
    print(f"\nCV purgada da volatilidade ({cv['n_splits']} folds):")
    print(f"  modelo  AUC = {cv['model_auc_mean']:.3f} ± {cv['model_auc_std']:.3f}")
    print(f"  baseline AUC = {cv['baseline_auc_mean']:.3f} ± {cv['baseline_auc_std']:.3f}")
    print(f"  ganho do modelo = {cv['model_beats_baseline']:+.3f}")
    print(f"\nDeflated Sharpe ({s['n_trials']} trials, melhor = {s['best_trial']}):")
    print(f"  Sharpe observado    = {s['dsr']['observed_sharpe']:+.4f}")
    print(f"  benchmark deflac.   = {s['dsr']['deflated_benchmark']:+.4f}")
    print(f"  Deflated Sharpe Ratio = {s['dsr']['deflated_sharpe_ratio']:.3f}")
    print(f"\nRelatório: {s['report_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Previsor de direção BTC/USDT.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="Coleta candles da Binance.")
    p_collect.add_argument("--backfill", type=int, default=0,
                           help="Quantidade de candles históricos a buscar.")
    p_collect.set_defaults(func=cmd_collect)

    p_train = sub.add_parser("train", help="Treina o modelo XGBoost.")
    p_train.add_argument("--valid-fraction", type=float, default=0.2,
                         dest="valid_fraction",
                         help="Fração final usada para validação temporal.")
    p_train.set_defaults(func=cmd_train)

    p_predict = sub.add_parser("predict", help="Gera e registra uma previsão.")
    p_predict.add_argument("--no-refresh", action="store_true",
                           help="Não buscar candles novos antes de prever.")
    p_predict.set_defaults(func=cmd_predict)

    p_eval = sub.add_parser("evaluate", help="Verifica previsões pendentes.")
    p_eval.add_argument("--no-refresh", action="store_true",
                        help="Não buscar candles novos antes de avaliar.")
    p_eval.set_defaults(func=cmd_evaluate)

    p_serve = sub.add_parser("serve", help="Sobe a interface web.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--debug", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_loop = sub.add_parser("loop", help="Ciclo contínuo de coleta/previsão.")
    p_loop.add_argument("--interval", type=int, default=60,
                        help="Segundos entre iterações.")
    p_loop.set_defaults(func=cmd_loop)

    p_bt = sub.add_parser("backtest",
                          help="Valida o sistema out-of-sample e gera relatório.")
    p_bt.add_argument("--sl", type=float, default=None,
                      help="Stop loss em %% (default: config).")
    p_bt.add_argument("--tp", type=float, default=None,
                      help="Take profit em %% (default: config).")
    p_bt.add_argument("--value", type=float, default=None,
                      help="Valor por operação em USDT.")
    p_bt.add_argument("--fee", type=float, default=None,
                      help="Taxa da corretora por lado em %%.")
    p_bt.add_argument("--slippage", type=float, default=None,
                      help="Slippage por execução em %%.")
    p_bt.add_argument("--min-confidence", type=float, default=None,
                      dest="min_confidence",
                      help="Confiança mínima do modelo para operar (0-1).")
    p_bt.add_argument("--test-fraction", type=float, default=None,
                      dest="test_fraction",
                      help="Fração final usada como out-of-sample.")
    p_bt.set_defaults(func=cmd_backtest)

    p_exp = sub.add_parser("experiment",
                           help="Estudo comparativo de horizontes de previsão.")
    p_exp.add_argument("--horizons", default=None,
                       help="Horizontes em min, separados por vírgula "
                            "(default: 5,15,30,60,240).")
    p_exp.add_argument("--sizes", default=None,
                       help="Tamanhos de amostra (candles), separados por "
                            "vírgula (default: 8000,100000,250000).")
    p_exp.set_defaults(func=cmd_experiment)

    p_tgt = sub.add_parser("targets",
                           help="Compara previsibilidade de variáveis-alvo.")
    p_tgt.add_argument("--horizons", default=None,
                       help="Horizontes em min (default: 5,15,60).")
    p_tgt.add_argument("--candles", type=int, default=250_000,
                       help="Quantos candles (mais recentes) usar.")
    p_tgt.set_defaults(func=cmd_targets)

    p_ve = sub.add_parser("vol-economics",
                          help="Valida o valor econômico da previsão de volatilidade.")
    p_ve.add_argument("--horizon", type=int, default=15,
                      help="Horizonte/holding em candles (default: 15).")
    p_ve.add_argument("--candles", type=int, default=250_000,
                      help="Quantos candles (mais recentes) usar.")
    p_ve.add_argument("--filter-pct", type=float, default=0.30, dest="filter_pct",
                      help="Top fração de vol p/ o filtro na Etapa 3 (default 0.30).")
    p_ve.set_defaults(func=cmd_vol_economics)

    p_val = sub.add_parser("validate",
                           help="CV purgada + Deflated Sharpe (anti-overfitting).")
    p_val.add_argument("--horizon", type=int, default=15)
    p_val.add_argument("--candles", type=int, default=120_000)
    p_val.add_argument("--splits", type=int, default=5)
    p_val.set_defaults(func=cmd_validate)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
