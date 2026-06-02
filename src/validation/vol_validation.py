"""Validação anti-trivialidade do achado de volatilidade.

Duas perguntas que um quant fará imediatamente sobre o AUC ~0,79 da vol:
  1. Ele sobrevive a uma validação cruzada PURGADA (sem vazamento por janelas
     sobrepostas)?
  2. Ele bate o baseline TRIVIAL "vol futura ≈ vol recente" (clustering puro)?

Este módulo responde às duas: roda PurgedKFold treinando o mesmo XGBoost e, em
cada fold, compara o AUC do modelo ao AUC do baseline de persistência (vol
realizada das últimas H barras — causal, sem treino).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.experiments.target_study import _make_xgb
from src.experiments.targets import build_target, future_quantities
from src.features.indicators import FEATURE_COLUMNS, add_indicators
from src.validation.purged_cv import PurgedKFold


def _trailing_rv(close: pd.Series, horizon: int) -> pd.Series:
    """Volatilidade realizada das ÚLTIMAS `horizon` barras (causal).

    Baseline de persistência: usa só o passado, sem modelo nem treino.
    """
    r = close.pct_change()
    r2 = (r * r)
    return np.sqrt(r2.rolling(horizon).sum())


def run_purged_vol_cv(candles: pd.DataFrame, horizon: int = 15,
                      n_splits: int = 5, embargo: float = 0.01) -> dict:
    """CV purgada do modelo de vol vs. baseline de persistência.

    Retorna AUCs por fold do modelo e do baseline, com média e desvio.
    """
    enriched = add_indicators(candles).reset_index(drop=True)
    close = enriched["close"]
    fq = future_quantities(close, horizon)

    feat_ok = enriched[FEATURE_COLUMNS].notna().all(axis=1)
    # Rótulo de alta/baixa vol com limiar global (mediana) — só para CV interna.
    spec = build_target("volatilidade", fq, feat_ok)
    y = spec.y

    trailing = _trailing_rv(close, horizon)

    valid = feat_ok & y.notna() & trailing.notna()
    idx = np.where(valid.values)[0]
    X_all = enriched.loc[valid, FEATURE_COLUMNS].astype(float).to_numpy()
    y_all = y[valid].astype(int).to_numpy()
    base_all = trailing[valid].to_numpy()

    cv = PurgedKFold(n_splits=n_splits, horizon=horizon, embargo=embargo)
    model_aucs, base_aucs = [], []
    for tr, te in cv.split(len(idx)):
        if len(np.unique(y_all[tr])) < 2 or len(np.unique(y_all[te])) < 2:
            continue
        model = _make_xgb(2)
        model.fit(X_all[tr], y_all[tr])
        proba = model.predict_proba(X_all[te])[:, 1]
        model_aucs.append(float(roc_auc_score(y_all[te], proba)))
        # Baseline: vol recente como score (sem treino).
        base_aucs.append(float(roc_auc_score(y_all[te], base_all[te])))

    return {
        "horizon": horizon,
        "n_splits": len(model_aucs),
        "embargo": embargo,
        "model_aucs": model_aucs,
        "baseline_aucs": base_aucs,
        "model_auc_mean": float(np.mean(model_aucs)) if model_aucs else float("nan"),
        "model_auc_std": float(np.std(model_aucs)) if model_aucs else float("nan"),
        "baseline_auc_mean": float(np.mean(base_aucs)) if base_aucs else float("nan"),
        "baseline_auc_std": float(np.std(base_aucs)) if base_aucs else float("nan"),
        "model_beats_baseline": (float(np.mean(model_aucs)) - float(np.mean(base_aucs)))
        if model_aucs else float("nan"),
    }
