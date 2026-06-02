"""Estudo comparativo de previsibilidade entre variáveis-alvo.

Para cada (alvo, horizonte) usa o MESMO XGBoost e os MESMOS indicadores e mede:
  * Previsibilidade   — AUC (0,5 = acaso) e acurácia vs. baseline (classe
    majoritária), com significância em amostras INDEPENDENTES (não-sobrepostas).
  * Estabilidade temporal — AUC por fold sequencial do período de teste
    (média e desvio): um alvo cujo skill se mantém ao longo do tempo é confiável.
  * Valor econômico potencial — separação do |movimento| realizado entre as
    classes previstas (em %), comparada ao custo de ida-e-volta. Mede se a
    previsão ajuda a antecipar o TAMANHO do movimento (o que é monetizável via
    sizing/opções), sem construir nenhuma estratégia.

Nota de escopo: a previsibilidade é medida com o **conjunto de features atual**
(indicadores pensados para direção). Um alvo que já se mostre mais previsível
mesmo assim é forte indício para priorizá-lo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import config
from src.experiments import stats_validity as sv
from src.experiments.targets import TARGET_NAMES, build_target, future_quantities
from src.features.indicators import FEATURE_COLUMNS, add_indicators

_N_FOLDS = 5  # folds sequenciais para estabilidade temporal


def _make_xgb(n_classes: int) -> XGBClassifier:
    """XGBoost com os MESMOS hiperparâmetros; só o objetivo acompanha o nº de
    classes (binário vs. multiclasse). Não é um modelo novo — é o mesmo XGBoost.
    """
    p = dict(config.XGB_PARAMS)
    p.pop("objective", None)
    p.pop("eval_metric", None)
    if n_classes == 2:
        p["objective"] = "binary:logistic"
        p["eval_metric"] = "logloss"
    else:
        p["objective"] = "multi:softprob"
        p["eval_metric"] = "mlogloss"
    return XGBClassifier(**p)


def _auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    """AUC binária ou macro one-vs-rest (multiclasse). NaN se indefinida."""
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr",
                                   average="macro", labels=list(range(n_classes))))
    except ValueError:
        return float("nan")


def run_target(enriched: pd.DataFrame, target: str, horizon: int,
               base_params: dict) -> dict:
    """Treina e avalia um alvo num horizonte. Retorna o dicionário de métricas."""
    test_fraction = base_params["test_fraction"]
    cost_hurdle = 2 * (base_params["fee_pct"] + base_params["slippage_pct"]) / 100.0

    close = enriched["close"]
    fq = future_quantities(close, horizon)

    feat_ok = enriched[FEATURE_COLUMNS].notna().all(axis=1)
    split_idx = int(len(enriched) * (1.0 - test_fraction))
    train_region = feat_ok & (enriched.index < split_idx)

    spec = build_target(target, fq, train_region)
    y = spec.y
    valid = feat_ok & y.notna()

    train_mask = valid & (enriched.index < split_idx)
    test_positions = np.where((valid & (enriched.index >= split_idx)).values)[0]
    if train_mask.sum() < 200 or len(test_positions) < horizon * 5:
        return _empty_target(target, horizon, spec)

    # Amostras de teste INDEPENDENTES (não-sobrepostas: uma a cada H candles).
    indep = test_positions[::horizon]

    X_train = enriched.loc[train_mask, FEATURE_COLUMNS].astype(float)
    y_train = y[train_mask].astype(int)
    X_test = enriched.iloc[indep][FEATURE_COLUMNS].astype(float)
    y_test = y.iloc[indep].astype(int).to_numpy()

    if y_train.nunique() < spec.n_classes:
        return _empty_target(target, horizon, spec)

    model = _make_xgb(spec.n_classes)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    preds = proba.argmax(axis=1)

    n = len(y_test)
    accuracy = float((preds == y_test).mean())
    baseline = float(y_train.value_counts(normalize=True).max())  # classe majoritária
    auc = _auc(y_test, proba, spec.n_classes)

    # Significância da acurácia contra o baseline (teste z de proporção, N indep).
    hits = int((preds == y_test).sum())
    _, p_value = sv.proportion_z_test(hits, n, p0=baseline)
    ci_low, ci_high = sv.wilson_ci(hits, n)

    # Estabilidade temporal: AUC por fold sequencial do teste.
    fold_aucs = _fold_aucs(y_test, proba, spec.n_classes, _N_FOLDS)
    auc_mean = float(np.nanmean(fold_aucs)) if fold_aucs else float("nan")
    auc_std = float(np.nanstd(fold_aucs)) if fold_aucs else float("nan")

    # Valor econômico: separação do |movimento| realizado entre classes previstas.
    absmove_test = fq["absmove"].iloc[indep].to_numpy()
    sep_df = pd.DataFrame({"pred": preds, "absmove": absmove_test})
    class_means = sep_df.groupby("pred")["absmove"].mean()
    econ_sep = float(class_means.max() - class_means.min()) if len(class_means) > 1 else 0.0
    econ_ratio = float(class_means.max() / class_means.min()) if (len(class_means) > 1 and class_means.min() > 0) else float("nan")

    return {
        "target": target,
        "label": spec.classes,
        "n_classes": spec.n_classes,
        "horizon": horizon,
        "n_indep": n,
        "accuracy": accuracy,
        "baseline": baseline,
        "acc_skill": accuracy - baseline,          # ganho sobre o acaso
        "auc": auc,
        "auc_skill": (auc - 0.5) if auc == auc else float("nan"),
        "p_value": p_value,
        "acc_ci_low": ci_low,
        "acc_ci_high": ci_high,
        "auc_mean_folds": auc_mean,
        "auc_std_folds": auc_std,
        "econ_separation_pct": econ_sep * 100.0,    # em pontos percentuais
        "econ_separation_ratio": econ_ratio,
        "cost_hurdle_pct": cost_hurdle * 100.0,
        "econ_beats_cost": econ_sep > cost_hurdle,
    }


def _fold_aucs(y_test: np.ndarray, proba: np.ndarray, n_classes: int,
               k: int) -> list[float]:
    """AUC em k folds sequenciais (estabilidade ao longo do tempo de teste)."""
    n = len(y_test)
    if n < k * 10:
        return []
    bounds = np.linspace(0, n, k + 1).astype(int)
    out = []
    for i in range(k):
        a, b = bounds[i], bounds[i + 1]
        out.append(_auc(y_test[a:b], proba[a:b], n_classes))
    return out


def run_size_study(candles: pd.DataFrame, horizons: list[int],
                   base_params: dict, targets: list[str] = None) -> dict:
    """Roda todos os alvos × horizontes para um conjunto de candles."""
    targets = targets or TARGET_NAMES
    enriched = add_indicators(candles).reset_index(drop=True)
    results = []
    for target in targets:
        for h in horizons:
            results.append(run_target(enriched, target, h, base_params))
    return {
        "n_candles": len(enriched),
        "start": int(enriched["open_time"].iloc[0]),
        "end": int(enriched["open_time"].iloc[-1]),
        "results": results,
    }


def _empty_target(target: str, horizon: int, spec) -> dict:
    return {
        "target": target, "label": spec.classes, "n_classes": spec.n_classes,
        "horizon": horizon, "n_indep": 0, "accuracy": float("nan"),
        "baseline": float("nan"), "acc_skill": float("nan"), "auc": float("nan"),
        "auc_skill": float("nan"), "p_value": float("nan"),
        "acc_ci_low": float("nan"), "acc_ci_high": float("nan"),
        "auc_mean_folds": float("nan"), "auc_std_folds": float("nan"),
        "econ_separation_pct": float("nan"), "econ_separation_ratio": float("nan"),
        "cost_hurdle_pct": float("nan"), "econ_beats_cost": False,
    }
