"""Testes do PurgedKFold: purge e embargo evitam vazamento na fronteira."""
from __future__ import annotations

import numpy as np

from src.validation.purged_cv import PurgedKFold


def test_train_and_test_are_disjoint():
    cv = PurgedKFold(n_splits=5, horizon=10, embargo=0.01)
    for train_idx, test_idx in cv.split(1_000):
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_purge_gap_respects_horizon():
    horizon = 20
    cv = PurgedKFold(n_splits=4, horizon=horizon, embargo=0.0)
    for train_idx, test_idx in cv.split(800):
        t_min, t_max = test_idx.min(), test_idx.max()
        # Nenhum índice de treino pode cair dentro de [t_min-H, t_max+H].
        train_in_gap = train_idx[(train_idx >= t_min - horizon) &
                                 (train_idx <= t_max + horizon)]
        assert len(train_in_gap) == 0


def test_embargo_removes_extra_samples_after_test():
    n = 1_000
    no_emb = sum(len(tr) for tr, _ in PurgedKFold(5, 5, 0.0).split(n))
    with_emb = sum(len(tr) for tr, _ in PurgedKFold(5, 5, 0.05).split(n))
    assert with_emb < no_emb


def test_all_test_indices_cover_series():
    cv = PurgedKFold(n_splits=5, horizon=1, embargo=0.0)
    covered = np.concatenate([te for _, te in cv.split(500)])
    # Cada amostra aparece em exatamente um fold de teste.
    assert len(covered) == 500
    assert len(np.unique(covered)) == 500
