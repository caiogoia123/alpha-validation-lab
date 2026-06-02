"""Configuração central da aplicação BTC/USDT.

Todos os parâmetros que controlam coleta, modelo e horizonte de previsão
ficam concentrados aqui para facilitar ajuste sem mexer no código dos módulos.
"""
from __future__ import annotations

import os

# --- Diretórios -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "btc.db")
MODEL_PATH = os.path.join(DATA_DIR, "xgb_model.json")
FEATURES_PATH = os.path.join(DATA_DIR, "feature_list.json")

# --- Mercado --------------------------------------------------------------
SYMBOL = "BTCUSDT"        # par único suportado pelo MVP
INTERVAL = "1m"           # candles de 1 minuto
BINANCE_BASE_URL = "https://api.binance.com"

# --- Horizonte de previsão ------------------------------------------------
# Prevemos a direção do preço daqui a HORIZON_MINUTES minutos.
# Com candles de 1m, isso equivale a "5 candles à frente".
HORIZON_MINUTES = 5

# --- Indicadores ----------------------------------------------------------
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 9
EMA_LONG = 21

# Limite mínimo de movimento (em %) para considerar o candle como "subiu".
# Movimentos menores que isso são tratados como queda/lateralização (classe 0),
# evitando que ruído ínfimo vire sinal de compra.
MIN_MOVE_PCT = 0.0

# --- Modelo XGBoost -------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
}

# Quantos candles históricos buscar de uma vez na Binance (máx. 1000 por request).
FETCH_LIMIT = 1000

# --- Backtesting / validação ----------------------------------------------
# Valores em PERCENTUAL (ex.: 0.3 = 0,3%). O motor converte para fração.
# Taxa e slippage refletem condições realistas de execução na Binance spot.
BACKTEST = {
    "initial_capital": 10_000.0,   # saldo inicial da conta simulada
    "value_per_trade": 1_000.0,    # notional (stake) por operação
    "stop_loss_pct": 0.30,         # stop loss em %
    "take_profit_pct": 0.30,       # take profit em %
    "fee_pct": 0.04,               # taxa da corretora por lado (taker ~0,04%)
    "slippage_pct": 0.02,          # slippage por execução (entrada e saída)
    "min_confidence": 0.50,        # modelo só opera com convicção >= isto
    "test_fraction": 0.30,         # fração final usada como out-of-sample
    "max_holding": HORIZON_MINUTES,  # tempo máx. em posição (candles)
    "random_seed": 42,             # semente do baseline aleatório
}

# Diretório onde o relatório e os gráficos são gravados.
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
