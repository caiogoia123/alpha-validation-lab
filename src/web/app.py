"""Aplicação Flask: dashboard simples de previsões BTC/USDT.

Endpoints:
  GET /              -> página HTML com probabilidades e histórico.
  GET /api/predict   -> gera nova previsão e a registra (JSON).
  GET /api/evaluate  -> avalia previsões pendentes (JSON).
  GET /api/stats     -> taxa de acerto + últimas previsões (JSON).
"""
from __future__ import annotations

from flask import Flask, jsonify, render_template

import config
from src.data.database import Database
from src.model.predictor import Predictor
from src.tracking.evaluator import Evaluator


def create_app() -> Flask:
    app = Flask(__name__)
    db = Database()

    def _serialize_predictions(df) -> list:
        records = []
        for _, r in df.iterrows():
            records.append({
                "id": int(r["id"]),
                "created_at": int(r["created_at"]),
                "direction": "ALTA" if int(r["predicted_direction"]) == 1 else "BAIXA",
                "prob_up": round(float(r["prob_up"]), 4),
                "price_at_prediction": float(r["price_at_prediction"]),
                "price_at_target": (None if r["price_at_target"] is None
                                    or (isinstance(r["price_at_target"], float)
                                        and r["price_at_target"] != r["price_at_target"])
                                    else float(r["price_at_target"])),
                "evaluated": int(r["evaluated"]),
                "correct": (None if r["correct"] is None
                            or (isinstance(r["correct"], float)
                                and r["correct"] != r["correct"])
                            else int(r["correct"])),
            })
        return records

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            symbol=config.SYMBOL,
            horizon=config.HORIZON_MINUTES,
        )

    @app.route("/api/predict")
    def api_predict():
        try:
            predictor = Predictor(db=db)
            result = predictor.predict_and_log(refresh=True)
            return jsonify({"ok": True, "prediction": result})
        except Exception as exc:  # noqa: BLE001 - superfície de erro p/ UI
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/evaluate")
    def api_evaluate():
        stats = Evaluator(db=db).evaluate_pending(refresh=True)
        return jsonify({"ok": True, "stats": stats})

    @app.route("/api/stats")
    def api_stats():
        stats = db.accuracy_stats()
        recent = _serialize_predictions(db.recent_predictions(limit=25))
        return jsonify({"ok": True, "stats": stats, "recent": recent})

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
