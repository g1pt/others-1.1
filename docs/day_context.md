# Day Context Labels (SPX500)

This module labels each trading day using intraday ICT-style context buckets. The
labels are derived from rolling ATR on the same timeframe and basic day metrics
so they can be used for risk entry diagnostics without altering any trade logic.

## Labels

- **TREND_DAY**: range expansion with minimal return to the open and a close in
  the top/bottom 30% of the daily range.
- **RANGE_DAY**: compressed range, repeated returns to the open, and a close
  near the open.
- **REVERSAL_DAY**: pre-window extremes around NYO/LC are swept post-window,
  with a net close on the opposite side of the open.
- **CHAOS_DAY**: multiple open revisits, modest range, and sweeps on both sides.
- **UNKNOWN**: does not satisfy the above conditions.

## Metrics

Each day computes:

- **range_atr**: daily range divided by ATR.
- **close_to_open_atr**: distance between close and open in ATR.
- **open_revisit_count**: closes within the open revisit tolerance after the
  first two candles.
- **sweep counts**: intraday sweep-high and sweep-low detections.

Defaults are defined in `src/config.py` and can be adjusted later without
changing labeling logic.
