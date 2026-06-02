"""Probabilistic e Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

Quando se testam N configurações de estratégia e se reporta a melhor, o Sharpe
observado é inflado por viés de seleção. O Deflated Sharpe Ratio (DSR) corrige
isso estimando o Sharpe esperado do MELHOR entre N tentativas sob a hipótese
nula (sem skill) e medindo a probabilidade de o Sharpe observado superá-lo,
ajustando ainda por assimetria e curtose dos retornos.

Referência: Bailey, D. & López de Prado, M. (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Constante de Euler-Mascheroni (usada na estatística do máximo de N gaussianas).
_EULER = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inversa da CDF normal padrão por bisseção (uso pontual)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    lo, hi = -8.0, 8.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@dataclass
class SharpeStats:
    sharpe: float          # Sharpe por operação/observação (não anualizado)
    n: int
    skew: float
    kurtosis: float        # curtose (não-excesso; normal = 3)


def sharpe_stats(returns: np.ndarray) -> SharpeStats:
    """Estatísticas de Sharpe + momentos de ordem superior de uma série."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 2 or r.std(ddof=1) == 0:
        return SharpeStats(0.0, n, 0.0, 3.0)
    mean = r.mean()
    sd = r.std(ddof=1)
    sharpe = mean / sd
    z = (r - mean) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    return SharpeStats(float(sharpe), n, skew, kurt)


def probabilistic_sharpe_ratio(stats: SharpeStats, sr_benchmark: float = 0.0) -> float:
    """PSR: P(Sharpe verdadeiro > sr_benchmark) dado o estimado e os momentos.

    Ajusta o erro-padrão do Sharpe por assimetria e curtose (não-normalidade).
    """
    if stats.n < 2:
        return float("nan")
    sr, n, g3, g4 = stats.sharpe, stats.n, stats.skew, stats.kurtosis
    denom = math.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2))
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return _norm_cdf(z)


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Sharpe esperado do melhor entre `n_trials` sob H0 (sem skill).

    Aproximação do valor esperado do máximo de N gaussianas i.i.d. de média 0 e
    desvio `sr_std` (dispersão dos Sharpes entre as configurações testadas).
    """
    if n_trials < 2:
        return 0.0
    a = (1 - _EULER) * _norm_ppf(1 - 1.0 / n_trials)
    b = _EULER * _norm_ppf(1 - 1.0 / (n_trials * math.e))
    return sr_std * (a + b)


def deflated_sharpe_ratio(stats: SharpeStats, n_trials: int,
                          sr_trials_std: float) -> dict:
    """DSR: PSR avaliado contra o benchmark inflado pelo viés de seleção.

    Parâmetros
    ----------
    stats : estatísticas da estratégia escolhida (a "melhor").
    n_trials : quantas configurações independentes foram testadas.
    sr_trials_std : desvio-padrão dos Sharpes entre as configurações testadas.

    Retorna dict com o benchmark deflacionado e o DSR (prob. de skill real).
    """
    sr0 = expected_max_sharpe(n_trials, sr_trials_std)
    dsr = probabilistic_sharpe_ratio(stats, sr_benchmark=sr0)
    return {
        "observed_sharpe": stats.sharpe,
        "n_obs": stats.n,
        "n_trials": n_trials,
        "deflated_benchmark": sr0,
        "psr_vs_zero": probabilistic_sharpe_ratio(stats, 0.0),
        "deflated_sharpe_ratio": dsr,
        "skew": stats.skew,
        "kurtosis": stats.kurtosis,
    }
