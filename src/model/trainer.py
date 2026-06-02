"""Treino do classificador XGBoost.

O split é temporal (sem embaralhar): treinamos no passado e validamos no
trecho mais recente, refletindo o uso real do modelo.
"""
from __future__ import annotations

import json

from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

import config
from src.data.database import Database
from src.features.indicators import FEATURE_COLUMNS
from src.model.dataset import build_supervised


def train(db: Database | None = None, valid_fraction: float = 0.2) -> dict:
    """Treina e salva o modelo. Retorna métricas de validação.

    Parâmetros
    ----------
    db : Database opcional (usa o padrão se None).
    valid_fraction : fração final da série usada como validação temporal.
    """
    db = db or Database()
    candles = db.load_candles()
    if len(candles) < 200:
        raise RuntimeError(
            f"Histórico insuficiente para treinar: {len(candles)} candles. "
            "Rode a coleta primeiro (ex.: main.py collect --backfill 5000)."
        )

    X, y = build_supervised(candles)
    if len(X) < 100:
        raise RuntimeError("Amostras válidas insuficientes após gerar features.")

    # Split temporal: nada de shuffle, para não vazar futuro no treino.
    split = int(len(X) * (1.0 - valid_fraction))
    X_train, X_valid = X.iloc[:split], X.iloc[split:]
    y_train, y_valid = y.iloc[:split], y.iloc[split:]

    model = XGBClassifier(**config.XGB_PARAMS)
    model.fit(X_train, y_train)

    # Métricas na validação.
    proba = model.predict_proba(X_valid)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "valid_accuracy": float(accuracy_score(y_valid, preds)),
        "class_balance_train": float(y_train.mean()),
    }
    try:
        metrics["valid_auc"] = float(roc_auc_score(y_valid, proba))
    except ValueError:
        metrics["valid_auc"] = None  # uma única classe na validação

    # Persistência: modelo + lista de features (contrato treino/inferência).
    model.save_model(config.MODEL_PATH)
    with open(config.FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f)

    return metrics
