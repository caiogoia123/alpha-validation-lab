"""Cliente REST para a API pública da Binance.

Usa apenas o endpoint público de klines (não requer chave de API). Retorna os
candles já normalizados para o formato que o restante da aplicação consome.
"""
from __future__ import annotations

import time

import requests

import config

# Índices das colunas no array de kline retornado pela Binance.
# Doc: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
_OPEN_TIME = 0
_OPEN = 1
_HIGH = 2
_LOW = 3
_CLOSE = 4
_VOLUME = 5
_CLOSE_TIME = 6


class BinanceClient:
    """Wrapper fino sobre o endpoint /api/v3/klines da Binance."""

    def __init__(self, base_url: str = config.BINANCE_BASE_URL,
                 symbol: str = config.SYMBOL, interval: str = config.INTERVAL):
        self.base_url = base_url.rstrip("/")
        self.symbol = symbol
        self.interval = interval
        self._session = requests.Session()

    def fetch_klines(self, limit: int = config.FETCH_LIMIT,
                     start_time: int | None = None,
                     end_time: int | None = None) -> list[dict]:
        """Busca candles e devolve uma lista de dicts normalizados.

        Cada dict contém: open_time, open, high, low, close, volume, close_time.
        Os tempos são em milissegundos (epoch), como a Binance entrega.
        """
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": min(limit, 1000),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        url = f"{self.base_url}/api/v3/klines"
        resp = self._session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        candles = []
        for k in raw:
            candles.append({
                "open_time": int(k[_OPEN_TIME]),
                "open": float(k[_OPEN]),
                "high": float(k[_HIGH]),
                "low": float(k[_LOW]),
                "close": float(k[_CLOSE]),
                "volume": float(k[_VOLUME]),
                "close_time": int(k[_CLOSE_TIME]),
            })
        return candles

    def fetch_recent(self, limit: int = config.FETCH_LIMIT) -> list[dict]:
        """Atalho para os `limit` candles mais recentes."""
        return self.fetch_klines(limit=limit)

    def backfill(self, total: int) -> list[dict]:
        """Busca `total` candles paginando para trás a partir de agora.

        A Binance limita 1000 candles por request; aqui paginamos usando
        end_time para montar um histórico maior de uma vez.
        """
        collected: list[dict] = []
        end_time: int | None = None
        remaining = total

        while remaining > 0:
            batch = self.fetch_klines(limit=min(remaining, 1000), end_time=end_time)
            if not batch:
                break
            collected = batch + collected
            remaining -= len(batch)
            # Próxima página termina 1ms antes do candle mais antigo deste lote.
            end_time = batch[0]["open_time"] - 1
            if len(batch) < 1000:
                break  # Não há mais histórico disponível.
            time.sleep(0.2)  # cortesia com o rate limit público

        return collected
