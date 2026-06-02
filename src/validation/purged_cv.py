"""K-fold purgado e com embargo para validação cruzada de séries financeiras.

Problema: quando o rótulo no instante t depende de informação até t+H, um split
ingênuo coloca no treino amostras cujo horizonte invade o período de teste —
vazamento. A solução de López de Prado (Advances in Financial ML, cap. 7) é:
  * PURGE: remover do treino toda amostra cujo horizonte de rótulo [t, t+H]
    sobreponha o intervalo de teste.
  * EMBARGO: remover também uma faixa logo após o teste, para neutralizar
    autocorrelação serial residual.

Aqui os rótulos têm horizonte fixo H (em candles), então o purge equivale a
remover H amostras de cada lado da fronteira treino/teste.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class PurgedKFold:
    """K-fold temporal (sem shuffle) com purge e embargo.

    Parâmetros
    ----------
    n_splits : número de folds.
    horizon  : horizonte do rótulo em amostras (candles) — define o purge.
    embargo  : fração do total removida após o teste (ex.: 0.01 = 1%).
    """

    def __init__(self, n_splits: int = 5, horizon: int = 1, embargo: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits deve ser >= 2.")
        self.n_splits = n_splits
        self.horizon = max(int(horizon), 0)
        self.embargo = max(float(embargo), 0.0)

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Gera (train_idx, test_idx) por fold, já purgados e com embargo."""
        indices = np.arange(n_samples)
        fold_bounds = np.linspace(0, n_samples, self.n_splits + 1).astype(int)
        embargo_n = int(n_samples * self.embargo)

        for i in range(self.n_splits):
            test_start, test_end = fold_bounds[i], fold_bounds[i + 1]
            test_idx = indices[test_start:test_end]

            # Purge: descarta H amostras antes e depois do teste (horizonte do
            # rótulo). Embargo: estende a remoção pós-teste.
            left = test_start - self.horizon
            right = test_end + self.horizon + embargo_n

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[max(0, left):min(n_samples, right)] = False
            train_idx = indices[train_mask]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx
