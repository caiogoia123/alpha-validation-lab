# Data Card — BTC/USDT 1-minute OHLCV

## Source
- **Provider:** Binance public REST API (`/api/v3/klines`), no API key required.
- **Symbol / interval:** `BTCUSDT`, `1m`.
- **Volume:** up to **250,000 candles** (~174 days) per the studies.
- **Storage:** local SQLite (`data/btc.db`), one row per candle.

## Schema (`candles` table)
| column | type | description |
|---|---|---|
| `open_time` | INTEGER (PK) | candle start, epoch ms |
| `open/high/low/close` | REAL | OHLC price |
| `volume` | REAL | base-asset volume |
| `close_time` | INTEGER | candle end, epoch ms |

## Collection & integrity
- **Idempotent ingestion:** `INSERT OR IGNORE` keyed on `open_time` — re-fetching
  never duplicates rows.
- **Backfill:** paginated backwards via `endTime` (1000 candles/request, rate-limit
  courtesy sleep).
- **Gaps:** Binance klines are contiguous for liquid pairs; no imputation applied.

## Known limitations / biases
- **Single venue.** Binance prices ≠ consolidated tape; basis vs. other venues
  ignored.
- **Survivorship / regime:** the ~174-day window reflects one market regime; results
  are not claimed to generalize across all regimes.
- **No L2 / order-flow / trades data** — only aggregated OHLCV.
- **Timestamps in UTC (ms).** No timezone-dependent features used.

## Leakage controls
- All features are **causal** (functions of data up to and including `t`);
  verified by `tests/test_no_leakage.py`.
- Targets look **forward** by horizon `H`; the last `H` rows (no observable future)
  are dropped, not silently labeled.
