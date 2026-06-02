"""Estudo experimental: varia apenas o horizonte de previsão.

Para cada horizonte H (em minutos = candles de 1m):
  1. Constrói o rótulo "preço sobe em H candles?" (mesmo pipeline de features).
  2. Split temporal in-sample / out-of-sample (sem shuffle).
  3. Treina o MESMO XGBoost (mesmos XGB_PARAMS) no in-sample.
  4. Mede a acurácia direcional out-of-sample direto das previsões (N cheio),
     com IC de Wilson e p-valor contra o acaso (50%).
  5. Simula as operações no mesmo motor de backtest, com saída por TEMPO em H
     (sem bracket SL/TP, para que a posição reflita exatamente a previsão de H)
     e os mesmos custos (taxa + slippage). Calcula as métricas de trading.

Tudo é controlado: dataset, indicadores, modelo, pipeline e backtest são os
mesmos; só o horizonte muda.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import config
from src.backtest.engine import BacktestConfig, simulate_sequential
from src.backtest.metrics import compute_metrics
from src.backtest.strategies import model_signals
from src.experiments import stats_validity as sv
from src.features.indicators import FEATURE_COLUMNS, add_indicators


def _horizon_cfg(base_params: dict, horizon: int) -> BacktestConfig:
    """BacktestConfig para o horizonte: saída por tempo (SL/TP=0), holding=H."""
    params = dict(base_params)
    params["stop_loss_pct"] = 0.0   # desativa bracket -> saída por tempo
    params["take_profit_pct"] = 0.0
    params["max_holding"] = horizon
    return BacktestConfig.from_dict(params)


def run_single_horizon(enriched: pd.DataFrame, horizon: int,
                       base_params: dict) -> dict:
    """Roda treino + backtest para um horizonte e devolve as métricas."""
    cfg = _horizon_cfg(base_params, horizon)

    close = enriched["close"]
    future = close.shift(-horizon)
    label = (future > close).astype("float")  # 1 sobe, 0 desce/igual

    feat_ok = enriched[FEATURE_COLUMNS].notna().all(axis=1)
    trainable = feat_ok & label.notna()

    split_idx = int(len(enriched) * (1.0 - cfg.test_fraction))

    train_mask = trainable & (enriched.index < split_idx)
    test_mask = trainable & (enriched.index >= split_idx)

    X_train = enriched.loc[train_mask, FEATURE_COLUMNS].astype(float)
    y_train = label[train_mask].astype(int)
    test_positions = np.where(test_mask.values)[0]

    if len(X_train) < 200 or y_train.nunique() < 2 or len(test_positions) < horizon:
        return _empty_horizon(horizon, len(test_positions))

    # --- Treino (mesmos hiperparâmetros de produção) ----------------------
    model = XGBClassifier(**config.XGB_PARAMS)
    model.fit(X_train, y_train)

    # --- Acurácia direcional out-of-sample em amostras INDEPENDENTES -------
    # CRÍTICO: com candles de 1m, janelas de horizonte H consecutivas se
    # sobrepõem em (H-1)/H. Usar todas as previsões infla o N e produz IC/p
    # espúrios. Amostramos a cada H candles para obter janelas não-sobrepostas
    # (≈ N/H observações de fato independentes), base honesta para os testes.
    indep_positions = test_positions[::horizon]
    X_indep = enriched.loc[indep_positions, FEATURE_COLUMNS].astype(float)
    y_indep = label.loc[indep_positions].astype(int)

    preds = (model.predict_proba(X_indep)[:, 1] >= 0.5).astype(int)
    hits = int((preds == y_indep.values).sum())
    n_indep = int(len(y_indep))
    n_overlap = int(len(test_positions))
    accuracy = hits / n_indep
    ci_low, ci_high = sv.wilson_ci(hits, n_indep)
    z, p_value = sv.proportion_z_test(hits, n_indep)
    reliability = sv.assess_reliability(n_indep)

    # --- Movimento típico vs. custo (contexto do horizonte) ---------------
    fut_ret = (future[test_mask] / close[test_mask] - 1.0).abs()
    median_move_pct = float(fut_ret.median() * 100.0)
    cost_hurdle_pct = float(2 * (cfg.fee + cfg.slippage) * 100.0)  # ida+volta

    # --- Backtest no out-of-sample (mesmo motor, saída por tempo) ---------
    test_df = enriched.iloc[split_idx:].reset_index(drop=True)
    sig, conf = model_signals(model, test_df)
    trades = simulate_sequential(test_df, sig, conf, cfg)
    m = compute_metrics(trades, cfg)

    return {
        "horizon": horizon,
        "n_test": n_indep,        # amostras INDEPENDENTES (base dos testes)
        "n_indep": n_indep,
        "n_overlap": n_overlap,   # previsões sobrepostas (apenas referência)
        "moe": reliability.moe,
        "reliability": reliability.label,
        "directional_accuracy": accuracy,
        "acc_ci_low": ci_low,
        "acc_ci_high": ci_high,
        "p_value": p_value,
        "edge_significant": (ci_low > 0.5),  # IC 95% inteiramente acima de 0,5
        "median_move_pct": median_move_pct,
        "cost_hurdle_pct": cost_hurdle_pct,
        "n_trades": m["n_trades"],
        "profit_factor": m["profit_factor"],
        "cumulative_return": m["cumulative_return"],
        "sharpe_trade": m["sharpe_trade"],
        "max_drawdown_pct": m["max_drawdown_pct"],
        "expectancy": m["expectancy"],
        "total_pnl": m["total_pnl"],
    }


def run_size_study(candles: pd.DataFrame, horizons: list[int],
                   base_params: dict) -> dict:
    """Roda o estudo de horizonte para um tamanho de amostra (subset de candles).

    Retorna dict com: n_candles, período, buy_and_hold do teste e a lista de
    resultados por horizonte.
    """
    enriched = add_indicators(candles).reset_index(drop=True)
    rows = [run_single_horizon(enriched, h, base_params) for h in horizons]

    # Contexto de mercado: retorno buy & hold no trecho de teste.
    split_idx = int(len(enriched) * (1.0 - base_params["test_fraction"]))
    test_close = enriched["close"].iloc[split_idx:]
    bh_return = float(test_close.iloc[-1] / test_close.iloc[0] - 1.0)

    return {
        "n_candles": len(enriched),
        "n_test_candles": len(test_close),
        "start": int(enriched["open_time"].iloc[0]),
        "end": int(enriched["open_time"].iloc[-1]),
        "buy_and_hold_return": bh_return,
        "results": rows,
    }


def _empty_horizon(horizon: int, n_overlap: int) -> dict:
    return {
        "horizon": horizon, "n_test": 0, "n_indep": 0, "n_overlap": n_overlap,
        "moe": 1.0, "reliability": "insuficiente",
        "directional_accuracy": float("nan"),
        "acc_ci_low": float("nan"), "acc_ci_high": float("nan"),
        "p_value": float("nan"), "edge_significant": False,
        "median_move_pct": float("nan"), "cost_hurdle_pct": float("nan"),
        "n_trades": 0, "profit_factor": 0.0, "cumulative_return": 0.0,
        "sharpe_trade": 0.0, "max_drawdown_pct": 0.0, "expectancy": 0.0,
        "total_pnl": 0.0,
    }
