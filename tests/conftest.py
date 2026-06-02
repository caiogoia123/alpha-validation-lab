"""Fixtures compartilhadas: candles sintéticos determinísticos (sem rede/DB)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_candles() -> pd.DataFrame:
    """Gera um random walk geométrico reprodutível com OHLCV coerente.

    1.500 candles de 1 min. high/low são construídos para conter open/close.
    """
    rng = np.random.default_rng(42)
    n = 1_500
    ret = rng.normal(0, 0.0008, n)
    close = 30_000 * np.exp(np.cumsum(ret))
    open_ = np.empty(n)
    open_[0] = 30_000
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0, 0.0006, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(1, 100, n)
    open_time = (np.arange(n) * 60_000 + 1_700_000_000_000).astype("int64")
    return pd.DataFrame({
        "open_time": open_time,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
        "close_time": open_time + 59_999,
    })
