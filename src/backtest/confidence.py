"""Análise por faixa de confiança.

Agrupa as operações do MODELO pelas faixas de convicção e mede, em cada faixa,
quantidade de operações, taxa de acerto e lucro líquido. Objetivo: verificar se
convicções maiores realmente produzem resultados melhores (calibração útil).
"""
from __future__ import annotations

from src.backtest.engine import Trade

# Faixas pedidas no enunciado (limite inferior incluso, superior excluso,
# exceto a última que vai até 1.0 inclusive).
BUCKETS = [
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.01, "90-100%"),
]


def analyze_confidence(trades: list[Trade]) -> list[dict]:
    """Uma linha por faixa: contagem, acurácia direcional, win rate e lucro.

    Reporta DUAS taxas de acerto:
      * directional_accuracy -> preço foi na direção prevista (skill do sinal).
      * win_rate             -> operações com lucro líquido após custos.
    A análise foca em directional_accuracy, pois é o que revela se a convicção
    do modelo é informativa (independente da viabilidade após custos).
    """
    rows = []
    for low, high, label in BUCKETS:
        bucket = [t for t in trades if low <= t.confidence < high]
        n = len(bucket)
        if n == 0:
            rows.append({
                "bucket": label, "n_trades": 0, "directional_accuracy": None,
                "win_rate": None, "net_profit": 0.0, "avg_confidence": None,
            })
            continue
        dir_hits = sum(t.direction_correct for t in bucket)
        wins = sum(1 for t in bucket if t.net_pnl > 0)
        net = sum(t.net_pnl for t in bucket)
        avg_conf = sum(t.confidence for t in bucket) / n
        rows.append({
            "bucket": label,
            "n_trades": n,
            "directional_accuracy": dir_hits / n,
            "win_rate": wins / n,
            "net_profit": net,
            "avg_confidence": avg_conf,
        })
    return rows


def monotonicity_score(rows: list[dict]) -> float:
    """Correlação de postos (Spearman) entre a faixa e a acurácia direcional.

    Considera apenas faixas com operações. Retorna valor em [-1, 1]; positivo
    indica que confiança maior tende a acertar mais a DIREÇÃO. Retorna 0 se há
    menos de duas faixas povoadas.
    """
    populated = [(i, r["directional_accuracy"]) for i, r in enumerate(rows)
                 if r["n_trades"] > 0 and r["directional_accuracy"] is not None]
    if len(populated) < 2:
        return 0.0

    ranks_x = list(range(len(populated)))            # ordem das faixas
    y_values = [w for _, w in populated]
    ranks_y = _rank(y_values)

    n = len(populated)
    d2 = sum((rx - ry) ** 2 for rx, ry in zip(ranks_x, ranks_y, strict=True))
    return 1 - (6 * d2) / (n * (n ** 2 - 1))


def _rank(values: list[float]) -> list[float]:
    """Postos médios (lida com empates) para a correlação de Spearman."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks
