"""Motor de simulação de operações (backtesting realista).

Modela uma operação por sinal, com:
  * Stop Loss / Take Profit percentuais
  * Notional fixo por operação (valor por operação)
  * Taxa da corretora por lado
  * Slippage por execução (entrada e saída)

Premissas explícitas (documentadas para honestidade do backtest):
  1. Entrada no fechamento do candle de sinal, com slippage aplicada ao fill.
  2. Saída por SL/TP verificada candle a candle pelos high/low subsequentes.
  3. Se, no mesmo candle, tanto SL quanto TP forem atingíveis, assume-se que o
     SL ocorreu primeiro (cenário conservador — não há tick data para desempate).
  4. Se nem SL nem TP ocorrerem dentro de `max_holding` candles, sai-se por
     tempo no fechamento do último candle do horizonte.

Dois modos de simulação:
  * `simulate_sequential` — uma posição por vez (conta realista, p/ curva de
    capital, drawdown e Sharpe). Sinais durante uma posição aberta são ignorados.
  * `simulate_independent` — cada sinal vira uma operação independente (para a
    análise por faixa de confiança, que precisa de amostra cheia).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestConfig:
    """Parâmetros do backtest já normalizados para FRAÇÃO (não percentual)."""

    initial_capital: float
    value_per_trade: float
    stop_loss: float       # fração (0.003 = 0,3%)
    take_profit: float     # fração
    fee: float             # fração por lado
    slippage: float        # fração por execução
    min_confidence: float
    test_fraction: float
    max_holding: int
    random_seed: int

    @classmethod
    def from_dict(cls, d: dict) -> BacktestConfig:
        return cls(
            initial_capital=float(d["initial_capital"]),
            value_per_trade=float(d["value_per_trade"]),
            stop_loss=float(d["stop_loss_pct"]) / 100.0,
            take_profit=float(d["take_profit_pct"]) / 100.0,
            fee=float(d["fee_pct"]) / 100.0,
            slippage=float(d["slippage_pct"]) / 100.0,
            min_confidence=float(d["min_confidence"]),
            test_fraction=float(d["test_fraction"]),
            max_holding=int(d["max_holding"]),
            random_seed=int(d["random_seed"]),
        )


@dataclass
class Trade:
    """Resultado de uma operação simulada."""

    entry_time: int
    exit_time: int
    direction: int          # +1 long, -1 short
    entry_price: float
    exit_price: float
    exit_reason: str        # 'SL', 'TP' ou 'TIME'
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float       # net_pnl / value_per_trade
    confidence: float
    direction_correct: int  # 1 se o preço foi na direção prevista (sem custos)


def simulate_single_trade(candles: pd.DataFrame, entry_idx: int, direction: int,
                          cfg: BacktestConfig,
                          confidence: float = 0.5) -> Trade | None:
    """Simula uma operação aberta no candle `entry_idx` na direção dada.

    Retorna None se não há candles suficientes à frente para operar.
    `candles` deve ter colunas: open_time, high, low, close.
    """
    n = len(candles)
    if entry_idx + 1 >= n:
        return None  # sem candle futuro para acompanhar a posição

    raw_entry = float(candles["close"].iloc[entry_idx])
    # Slippage na entrada: long compra mais caro; short vende mais barato.
    entry = raw_entry * (1 + cfg.slippage) if direction == 1 \
        else raw_entry * (1 - cfg.slippage)

    if direction == 1:
        sl_price = entry * (1 - cfg.stop_loss)
        tp_price = entry * (1 + cfg.take_profit)
    else:
        sl_price = entry * (1 + cfg.stop_loss)
        tp_price = entry * (1 - cfg.take_profit)

    qty = cfg.value_per_trade / entry

    last_idx = min(entry_idx + cfg.max_holding, n - 1)
    exit_price = None
    exit_reason = "TIME"
    exit_idx = last_idx

    # SL/TP <= 0 desativa o bracket -> saída puramente por tempo (no horizonte).
    # Usado no estudo de horizonte, onde a posição deve refletir exatamente a
    # previsão de H candles, sem um bracket curto curto-circuitando a aposta.
    check_sl = cfg.stop_loss > 0
    check_tp = cfg.take_profit > 0

    for j in range(entry_idx + 1, last_idx + 1):
        if not (check_sl or check_tp):
            break  # sem bracket: vai direto para a saída por tempo

        high = float(candles["high"].iloc[j])
        low = float(candles["low"].iloc[j])

        if direction == 1:
            hit_sl = check_sl and low <= sl_price
            hit_tp = check_tp and high >= tp_price
        else:
            hit_sl = check_sl and high >= sl_price
            hit_tp = check_tp and low <= tp_price

        if hit_sl and hit_tp:
            # Conservador: assume o pior (stop primeiro).
            exit_price, exit_reason, exit_idx = sl_price, "SL", j
            break
        if hit_sl:
            exit_price, exit_reason, exit_idx = sl_price, "SL", j
            break
        if hit_tp:
            exit_price, exit_reason, exit_idx = tp_price, "TP", j
            break

    if exit_price is None:
        # Saída por tempo no fechamento do último candle do horizonte.
        exit_price = float(candles["close"].iloc[last_idx])
        exit_reason = "TIME"
        exit_idx = last_idx

    # Slippage na saída: long vende mais barato; short compra mais caro.
    exit_fill = exit_price * (1 - cfg.slippage) if direction == 1 \
        else exit_price * (1 + cfg.slippage)

    if direction == 1:
        gross = qty * (exit_fill - entry)
    else:
        gross = qty * (entry - exit_fill)

    # Taxa cobrada sobre o notional em cada lado (entrada e saída).
    fees = cfg.fee * (cfg.value_per_trade + qty * exit_fill)
    net = gross - fees

    # Acurácia direcional: compara preços CRUS (sem slippage/taxa). Mede o
    # acerto de direção do sinal, separado da viabilidade após custos.
    if direction == 1:
        direction_correct = 1 if exit_price > raw_entry else 0
    else:
        direction_correct = 1 if exit_price < raw_entry else 0

    return Trade(
        entry_time=int(candles["open_time"].iloc[entry_idx]),
        exit_time=int(candles["open_time"].iloc[exit_idx]),
        direction=direction,
        entry_price=entry,
        exit_price=exit_fill,
        exit_reason=exit_reason,
        gross_pnl=gross,
        fees=fees,
        net_pnl=net,
        return_pct=net / cfg.value_per_trade,
        confidence=confidence,
        direction_correct=direction_correct,
    )


def simulate_sequential(candles: pd.DataFrame, signals: pd.Series,
                        confidences: pd.Series, cfg: BacktestConfig) -> list[Trade]:
    """Conta realista: uma posição por vez.

    `signals` e `confidences` são alinhados por posição com `candles` (mesmo
    índice 0..n-1). Sinais que chegam com posição aberta são ignorados.
    """
    trades: list[Trade] = []
    n = len(candles)
    # Mapa tempo->posição pré-computado (O(1) por lookup em vez de varredura).
    otime = candles["open_time"].to_numpy()
    time_to_pos = {int(t): p for p, t in enumerate(otime)}
    sig = signals.to_numpy()
    conf_arr = confidences.to_numpy()

    i = 0
    while i < n:
        d = sig[i]
        direction = 0 if d != d else int(d)            # trata NaN
        c = conf_arr[i]
        conf = 0.5 if c != c else float(c)             # trata NaN
        if direction == 0 or conf < cfg.min_confidence:
            i += 1
            continue
        trade = simulate_single_trade(candles, i, direction, cfg, conf)
        if trade is None:
            break
        trades.append(trade)
        # Avança até o candle de saída (não sobrepõe posições).
        exit_pos = time_to_pos.get(trade.exit_time, i + cfg.max_holding)
        i = max(exit_pos + 1, i + 1)
    return trades


def simulate_independent(candles: pd.DataFrame, signals: pd.Series,
                         confidences: pd.Series, cfg: BacktestConfig) -> list[Trade]:
    """Cada sinal vira uma operação independente (podem se sobrepor no tempo).

    Usado pela análise por faixa de confiança, que avalia a qualidade do sinal
    em si — não um saldo sequencial — e por isso precisa de amostra cheia.
    """
    trades: list[Trade] = []
    n = len(candles)
    for i in range(n):
        direction = int(signals.iloc[i]) if not pd.isna(signals.iloc[i]) else 0
        conf = float(confidences.iloc[i]) if not pd.isna(confidences.iloc[i]) else 0.5
        if direction == 0 or conf < cfg.min_confidence:
            continue
        trade = simulate_single_trade(candles, i, direction, cfg, conf)
        if trade is not None:
            trades.append(trade)
    return trades
