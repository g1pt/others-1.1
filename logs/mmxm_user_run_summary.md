# MMXM run summary (user-provided output)

## 1) Baseline filter: wins vs losses ([4C-A] per dataset)

- Total datasets: **18**
- Positive runs (`total_return_pct > 0`): **12**
- Negative/flat runs (`total_return_pct <= 0`): **6**
- Hit-rate positive: **66.7%**

### By market

| Market | Positive | Negative/flat | Avg return % |
|---|---:|---:|---:|
| FX_SPX500 | 3 | 3 | 0.20 |
| ICMARKETS_EURUSD | 4 | 2 | 3.39 |
| OANDA_GBPUSD | 5 | 1 | 7.65 |
| **Total** | **12** | **6** | **3.75** |

## 2) Are we positive overall?

Yes, **overall positive** on the baseline snapshot.

- Mean return over all 18 baseline runs: **+3.75%**
- If you chain all 18 baseline returns as one synthetic equity path starting at 10,000:
  - Expected end equity: **18,938.67**
  - Synthetic growth: **+89.39%**

> Note: this chained curve is a **synthetic aggregation** of separate datasets/timeframes. It is useful as a directional summary, not as a deployable single-account backtest.

## 3) RR sweep quick filter (expectancy sign)

Across the pasted RR sweeps:

- Best-looking pockets are mainly:
  - `FX_SPX500 2m` (strongly positive expectancy across RR values).
  - `FX_SPX500 3m` (positive expectancy across RR values).
  - `ICMARKETS_EURUSD 15m` (positive expectancy across RR values).
  - `GBPUSD 30m` has selective positive RR points (not all).
- Weak pockets (mostly negative expectancy):
  - `ICMARKETS_EURUSD 1m/3m/5m/30m`
  - `GBPUSD 2m/3m/5m/15m`
  - `FX_SPX500 1m/15m` mostly weak except higher RR edge on 1m.

## 4) One-line conclusion

- **Baseline (4C-A): net positive** (12 wins vs 6 losses/flat).
- **RR behavior is regime-dependent**: a few timeframe/market combinations carry most of the edge.
- For a realistic live curve, trade only filtered combinations with positive expectancy + robustness consistency.
