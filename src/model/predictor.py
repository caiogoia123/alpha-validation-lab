"""Inferência: carrega o modelo treinado e emite previsões.

Cada chamada a `predict_and_log` calcula a probabilidade de alta para o
candle mais recente e registra a previsão no banco para verificação futura.
"""
from __future__ import annotations

import json
import os
import time

from xgboost import XGBClassifier

import config
from src.data.binance_client import BinanceClient
from src.data.database import Database
from src.model.dataset import build_inference_row

# Cada candle de 1m tem 60_000 ms; o alvo fica HORIZON candles à frente.
_MS_PER_MINUTE = 60_000


class Predictor:
    """Encapsula o modelo carregado e a lógica de previsão + registro."""

    def __init__(self, db: Database | None = None):
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                "Modelo não encontrado. Treine primeiro com: main.py train"
            )
        self.db = db or Database()
        self.model = XGBClassifier()
        self.model.load_model(config.MODEL_PATH)
        with open(config.FEATURES_PATH, encoding="utf-8") as f:
            self.feature_columns = json.load(f)

    def predict_and_log(self, refresh: bool = True) -> dict:
        """Gera uma previsão para o próximo horizonte e a persiste.

        Se `refresh`, busca candles recentes da Binance antes de prever, para
        garantir que estamos olhando o estado de mercado mais atual.
        Retorna um dict com os detalhes da previsão.
        """
        if refresh:
            client = BinanceClient()
            self.db.upsert_candles(client.fetch_recent(limit=config.FETCH_LIMIT))

        candles = self.db.load_candles(limit=config.FETCH_LIMIT)
        X_row, info = build_inference_row(candles)
        X_row = X_row[self.feature_columns]  # garante ordem correta

        prob_up = float(self.model.predict_proba(X_row)[:, 1][0])
        prob_down = 1.0 - prob_up
        direction = 1 if prob_up >= 0.5 else 0

        candle_time = info["candle_time"]
        target_time = candle_time + config.HORIZON_MINUTES * _MS_PER_MINUTE
        now_ms = int(time.time() * 1000)

        pred_id = self.db.insert_prediction(
            created_at=now_ms,
            candle_time=candle_time,
            target_time=target_time,
            predicted_direction=direction,
            prob_up=prob_up,
            prob_down=prob_down,
            price_at_prediction=info["price"],
        )

        return {
            "id": pred_id,
            "candle_time": candle_time,
            "target_time": target_time,
            "direction": "ALTA" if direction == 1 else "BAIXA",
            "prob_up": prob_up,
            "prob_down": prob_down,
            "price_at_prediction": info["price"],
            "horizon_minutes": config.HORIZON_MINUTES,
        }
