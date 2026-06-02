"""Camada de persistência em SQLite.

Duas tabelas:
  * candles      -> histórico de preço (OHLCV) por minuto.
  * predictions  -> cada previsão emitida pelo modelo e seu resultado.

A escrita de candles é idempotente (INSERT OR IGNORE por open_time), então
coletar repetidamente nunca duplica dados.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager

import pandas as pd

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    open_time   INTEGER PRIMARY KEY,   -- epoch ms do início do candle
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL NOT NULL,
    close_time  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          INTEGER NOT NULL,  -- quando a previsão foi feita (epoch ms)
    candle_time         INTEGER NOT NULL,  -- open_time do candle base da previsão
    target_time         INTEGER NOT NULL,  -- open_time do candle alvo (+horizonte)
    predicted_direction INTEGER NOT NULL,  -- 1 = alta, 0 = baixa
    prob_up             REAL NOT NULL,
    prob_down           REAL NOT NULL,
    price_at_prediction REAL NOT NULL,
    price_at_target     REAL,              -- preenchido na verificação
    correct             INTEGER,           -- 1/0, NULL enquanto não avaliado
    evaluated           INTEGER NOT NULL DEFAULT 0
);
"""


class Database:
    """Acesso ao banco SQLite local."""

    def __init__(self, path: str = config.DB_PATH):
        self.path = path
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # --- Candles ----------------------------------------------------------
    def upsert_candles(self, candles: Iterable[dict]) -> int:
        """Insere candles ignorando duplicatas. Retorna quantos eram novos."""
        rows = [
            (c["open_time"], c["open"], c["high"], c["low"],
             c["close"], c["volume"], c["close_time"])
            for c in candles
        ]
        if not rows:
            return 0
        with self._connect() as conn:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO candles "
                "(open_time, open, high, low, close, volume, close_time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            return conn.total_changes - before

    def load_candles(self, limit: int | None = None) -> pd.DataFrame:
        """Carrega candles ordenados por tempo crescente em um DataFrame."""
        query = "SELECT * FROM candles ORDER BY open_time ASC"
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn)
        if limit is not None and len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        return df

    def candle_count(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) AS n FROM candles")
            return cur.fetchone()["n"]

    def latest_open_time(self) -> int | None:
        with self._connect() as conn:
            cur = conn.execute("SELECT MAX(open_time) AS t FROM candles")
            row = cur.fetchone()
            return row["t"] if row and row["t"] is not None else None

    def get_close_at(self, open_time: int) -> float | None:
        """Preço de fechamento do candle cujo open_time == open_time."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT close FROM candles WHERE open_time = ?", (open_time,)
            )
            row = cur.fetchone()
            return row["close"] if row else None

    # --- Predictions ------------------------------------------------------
    def insert_prediction(self, *, created_at: int, candle_time: int,
                          target_time: int, predicted_direction: int,
                          prob_up: float, prob_down: float,
                          price_at_prediction: float) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO predictions "
                "(created_at, candle_time, target_time, predicted_direction, "
                " prob_up, prob_down, price_at_prediction) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (created_at, candle_time, target_time, predicted_direction,
                 prob_up, prob_down, price_at_prediction),
            )
            return cur.lastrowid

    def pending_predictions(self, now_ms: int) -> list[sqlite3.Row]:
        """Previsões ainda não avaliadas cujo alvo já passou."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM predictions "
                "WHERE evaluated = 0 AND target_time <= ? "
                "ORDER BY target_time ASC",
                (now_ms,),
            )
            return cur.fetchall()

    def mark_evaluated(self, pred_id: int, price_at_target: float,
                       correct: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE predictions SET price_at_target = ?, correct = ?, "
                "evaluated = 1 WHERE id = ?",
                (price_at_target, correct, pred_id),
            )

    def recent_predictions(self, limit: int = 50) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
                conn, params=(limit,),
            )
        return df

    def latest_prediction(self) -> sqlite3.Row | None:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 1"
            )
            return cur.fetchone()

    def accuracy_stats(self) -> dict:
        """Taxa de acerto global sobre previsões já avaliadas."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS hits "
                "FROM predictions WHERE evaluated = 1"
            )
            row = cur.fetchone()
        total = row["total"] or 0
        hits = row["hits"] or 0
        accuracy = (hits / total) if total else 0.0
        return {"evaluated": total, "hits": hits, "accuracy": accuracy}
