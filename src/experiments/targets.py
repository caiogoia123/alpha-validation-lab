"""Definição das variáveis-alvo do estudo comparativo de previsibilidade.

Todos os alvos são derivados **apenas do preço** (OHLC já no banco) e são
RÓTULOS — não entram como features do modelo (que permanece com os mesmos
indicadores). Cada alvo é enquadrado como classificação para que a
previsibilidade seja medida na mesma régua (AUC / acurácia vs. baseline) entre
alvos de natureza diferente.

Alvos:
  * direcao       — sobe vs. desce em H candles (baseline atual).            [2]
  * magnitude     — retorno em terços: forte queda / lateral / forte alta.  [3]
  * volatilidade  — volatilidade realizada dos próximos H acima da mediana. [2]
  * mov_absoluto  — |retorno| em H acima da mediana (move muito vs. pouco).  [2]
  * regime        — tendência forte/fraca, lateralização, alta/baixa vol.   [5]

Limiares (mediana, terços) são calculados **somente no treino** (sem vazamento).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Ordem canônica dos alvos no estudo.
TARGET_NAMES = ["direcao", "magnitude", "volatilidade", "mov_absoluto", "regime"]

TARGET_LABELS = {
    "direcao": "Direção (alta/baixa)",
    "magnitude": "Magnitude do retorno (terços)",
    "volatilidade": "Volatilidade futura (alta/baixa)",
    "mov_absoluto": "Movimento absoluto |ret| (grande/pequeno)",
    "regime": "Regime de mercado (5 classes)",
}


def future_quantities(close: pd.Series, horizon: int) -> pd.DataFrame:
    """Quantidades futuras (olham H candles à frente) para definir os alvos.

    Retorna DataFrame com:
      ret      — retorno simples close[t+H]/close[t]-1 (com sinal).
      absmove  — |ret|.
      rv       — volatilidade realizada dos próximos H min (sqrt da soma dos
                 retornos de 1 min ao quadrado).
      er       — efficiency ratio de Kaufman em H: |mov. líquido| / |caminho|,
                 em [0,1]; alto = tendência limpa, baixo = lateralização.
    """
    r = close.pct_change()
    r2 = (r * r).fillna(0.0)
    csum = r2.cumsum()
    fut_sumsq = csum.shift(-horizon) - csum          # soma de r2 em t+1..t+H
    rv = np.sqrt(fut_sumsq.clip(lower=0.0))

    ad = close.diff().abs().fillna(0.0)
    adc = ad.cumsum()
    path = adc.shift(-horizon) - adc                 # |caminho| percorrido
    net = (close.shift(-horizon) - close).abs()      # |movimento líquido|
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = net.values / path.values
    er = pd.Series(np.where(path.values > 0, ratio, np.nan), index=close.index)

    ret = close.shift(-horizon) / close - 1.0
    return pd.DataFrame({"ret": ret, "absmove": ret.abs(), "rv": rv, "er": er})


@dataclass
class TargetSpec:
    name: str
    y: pd.Series           # rótulo (float com NaN onde indefinido)
    classes: list[str]     # nomes das classes (ordem = código 0..K-1)
    n_classes: int


def build_target(name: str, fq: pd.DataFrame, train_mask: pd.Series) -> TargetSpec:
    """Constrói o rótulo de um alvo, com limiares calculados só no treino."""
    ret, absmove, rv, er = fq["ret"], fq["absmove"], fq["rv"], fq["er"]
    tr = train_mask  # atalho

    if name == "direcao":
        y = (ret > 0).astype(float)
        y[ret.isna()] = np.nan
        return TargetSpec(name, y, ["baixa", "alta"], 2)

    if name == "mov_absoluto":
        thr = absmove[tr & absmove.notna()].median()
        y = (absmove > thr).astype(float)
        y[absmove.isna()] = np.nan
        return TargetSpec(name, y, ["pequeno", "grande"], 2)

    if name == "volatilidade":
        thr = rv[tr & rv.notna()].median()
        y = (rv > thr).astype(float)
        y[rv.isna()] = np.nan
        return TargetSpec(name, y, ["baixa_vol", "alta_vol"], 2)

    if name == "magnitude":
        q1 = ret[tr & ret.notna()].quantile(1 / 3)
        q2 = ret[tr & ret.notna()].quantile(2 / 3)
        y = pd.Series(np.where(ret <= q1, 0.0,
                               np.where(ret >= q2, 2.0, 1.0)), index=ret.index)
        y[ret.isna()] = np.nan
        return TargetSpec(name, y, ["forte_queda", "lateral", "forte_alta"], 3)

    if name == "regime":
        valid_tr = tr & rv.notna() & er.notna()
        rv_lo, rv_hi = rv[valid_tr].quantile([1 / 3, 2 / 3])
        er_lo, er_hi = er[valid_tr].quantile([1 / 3, 2 / 3])
        y = pd.Series(np.nan, index=ret.index)
        midvol = (rv > rv_lo) & (rv < rv_hi)
        y[rv >= rv_hi] = 4.0                                   # alta_vol
        y[rv <= rv_lo] = 3.0                                   # baixa_vol
        y[midvol & (er >= er_hi)] = 0.0                        # tend_forte
        y[midvol & (er <= er_lo)] = 2.0                        # lateralizacao
        y[midvol & (er > er_lo) & (er < er_hi)] = 1.0          # tend_fraca
        y[ret.isna() | rv.isna() | er.isna()] = np.nan
        classes = ["tend_forte", "tend_fraca", "lateralizacao",
                   "baixa_vol", "alta_vol"]
        return TargetSpec(name, y, classes, 5)

    raise ValueError(f"Alvo desconhecido: {name}")
