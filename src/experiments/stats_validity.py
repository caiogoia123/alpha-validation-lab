"""Ferramentas de validade estatística para acurácia direcional.

Trata a acurácia como uma proporção binomial (acerto/erro vs. o acaso de 50%)
e fornece:
  * Intervalo de confiança de Wilson (robusto perto de 0,5 e para N moderado).
  * p-valor de um teste z bicaudal contra H0: acurácia = 0,5.
  * Tamanho de amostra necessário para detectar um dado edge com poder definido.
  * Avaliação qualitativa da confiabilidade dado o N disponível.

Implementação em matemática pura (sem scipy) para não adicionar dependências.
A CDF normal usa math.erf.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Valores críticos usuais.
_Z_95 = 1.959964     # bicaudal 95%
_Z_POWER_80 = 0.841621  # poder de 80% (z para 0,80)


def _normal_cdf(x: float) -> float:
    """CDF da normal padrão via função erro."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilson_ci(hits: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Intervalo de confiança de Wilson para uma proporção.

    Retorna (limite_inferior, limite_superior). Para n=0 retorna (0,1).
    """
    if n == 0:
        return 0.0, 1.0
    phat = hits / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


def proportion_z_test(hits: int, n: int, p0: float = 0.5) -> tuple[float, float]:
    """Teste z para proporção contra p0. Retorna (z, p_valor_bicaudal)."""
    if n == 0:
        return 0.0, 1.0
    phat = hits / n
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return 0.0, 1.0
    z = (phat - p0) / se
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return z, p_value


def required_n(true_acc: float, alpha: float = 0.05,
               power: float = 0.80) -> int:
    """Tamanho de amostra para detectar `true_acc` vs 0,5 (teste bicaudal).

    Aproximação por normal para uma proporção. Ex.: detectar 52% vs 50% com
    95%/80% exige ~N amostras de teste.
    """
    if true_acc <= 0.5:
        return math.inf  # nenhum edge a detectar
    z_alpha = _Z_95 if abs(alpha - 0.05) < 1e-9 else _z_from_alpha(alpha)
    z_beta = _Z_POWER_80 if abs(power - 0.80) < 1e-9 else _z_from_power(power)
    effect = true_acc - 0.5
    n = ((z_alpha * math.sqrt(0.25) +
          z_beta * math.sqrt(true_acc * (1 - true_acc))) / effect) ** 2
    return math.ceil(n)


def margin_of_error(n: int, z: float = _Z_95) -> float:
    """Margem de erro (±) da acurácia ~0,5 ao nível 95% para N de teste."""
    if n == 0:
        return 1.0
    return z * math.sqrt(0.25 / n)


@dataclass
class Reliability:
    n_test: int
    moe: float              # margem de erro (±) a 95% em torno de 0,5
    detectable_edge_pct: float  # menor edge detectável (= moe), em pontos %
    label: str              # 'baixa', 'moderada' ou 'alta'
    note: str


def assess_reliability(n_test: int) -> Reliability:
    """Classifica a confiabilidade do N de teste para concluir sobre edge.

    Critério baseado na margem de erro a 95% em torno de 0,5:
      moe > 2,0%  -> baixa  (não distingue edges pequenos do acaso)
      0,8%–2,0%   -> moderada
      moe < 0,8%  -> alta
    """
    moe = margin_of_error(n_test)
    moe_pct = moe * 100.0
    if moe_pct > 2.0:
        label = "baixa"
        note = ("Amostra pequena: edges realistas (~0,5–2%) ficam dentro do "
                "ruído amostral; conclusões sobre vantagem são frágeis.")
    elif moe_pct > 0.8:
        label = "moderada"
        note = ("Amostra intermediária: detecta edges grandes, mas edges "
                "pequenos permanecem incertos.")
    else:
        label = "alta"
        note = ("Amostra grande: distingue edges pequenos do acaso com "
                "razoável segurança.")
    return Reliability(n_test=n_test, moe=moe, detectable_edge_pct=moe_pct,
                       label=label, note=note)


def _z_from_alpha(alpha: float) -> float:
    """z bicaudal para alpha arbitrário (busca por bisseção na CDF)."""
    target = 1.0 - alpha / 2.0
    return _inverse_cdf(target)


def _z_from_power(power: float) -> float:
    return _inverse_cdf(power)


def _inverse_cdf(p: float) -> float:
    """Inversa da CDF normal padrão por bisseção (suficiente para uso pontual)."""
    lo, hi = -8.0, 8.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
