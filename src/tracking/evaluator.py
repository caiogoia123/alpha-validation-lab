"""Avaliação a posteriori das previsões.

Para cada previsão cujo `target_time` já passou e que ainda não foi avaliada,
buscamos o preço real do candle alvo e comparamos com o preço no momento da
previsão para decidir se a direção prevista estava correta.
"""
from __future__ import annotations

import time

import config
from src.data.binance_client import BinanceClient
from src.data.database import Database


class Evaluator:
    """Confere previsões pendentes e atualiza a taxa de acerto."""

    def __init__(self, db: Database | None = None):
        self.db = db or Database()
        self.client = BinanceClient()

    def evaluate_pending(self, refresh: bool = True) -> dict:
        """Avalia todas as previsões maduras e retorna um resumo.

        Uma previsão é considerada correta quando o sinal da direção prevista
        bate com o sinal do retorno real entre o preço base e o preço no alvo.
        """
        if refresh:
            # Garante que o candle alvo já está no banco.
            self.db.upsert_candles(self.client.fetch_recent(limit=config.FETCH_LIMIT))

        now_ms = int(time.time() * 1000)
        pending = self.db.pending_predictions(now_ms)

        evaluated = 0
        for row in pending:
            target_close = self.db.get_close_at(row["target_time"])
            if target_close is None:
                # Candle alvo ainda não disponível localmente; tenta de novo depois.
                continue

            actual_up = target_close > row["price_at_prediction"]
            predicted_up = row["predicted_direction"] == 1
            correct = 1 if (actual_up == predicted_up) else 0

            self.db.mark_evaluated(row["id"], target_close, correct)
            evaluated += 1

        stats = self.db.accuracy_stats()
        stats["newly_evaluated"] = evaluated
        return stats
