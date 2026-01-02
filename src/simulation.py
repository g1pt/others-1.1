"""Trade simulation for research backtests."""
from __future__ import annotations

from src.models import Trade
from src.data import Candle
from src.entries import EntrySignal


def simulate_trades(
    candles: list[Candle],
    entries: list[EntrySignal],
    max_hold: int = 20,
) -> list[Trade]:
    """Simulate trades using R-multiple targets and stops."""
    trades: list[Trade] = []

    for entry in entries:
        entry_candle = candles[entry.index]
        stop = entry.stop_price
        target = entry.target_price
        exit_time = None
        exit_price = None
        pnl_r = None

        for idx in range(entry.index, min(entry.index + max_hold, len(candles))):
            candle = candles[idx]
            if entry.direction == "bullish":
                if candle.low <= stop:
                    exit_time = candle.timestamp
                    exit_price = stop
                    pnl_r = -1.0
                    break
                if candle.high >= target:
                    exit_time = candle.timestamp
                    exit_price = target
                    pnl_r = (target - entry.entry_price) / (entry.entry_price - stop)
                    break
            else:
                if candle.high >= stop:
                    exit_time = candle.timestamp
                    exit_price = stop
                    pnl_r = -1.0
                    break
                if candle.low <= target:
                    exit_time = candle.timestamp
                    exit_price = target
                    pnl_r = (entry.entry_price - target) / (stop - entry.entry_price)
                    break

        if exit_time is None:
            last_candle = candles[min(entry.index + max_hold - 1, len(candles) - 1)]
            exit_time = last_candle.timestamp
            exit_price = last_candle.close
            if entry.direction == "bullish":
                pnl_r = (exit_price - entry.entry_price) / (entry.entry_price - stop)
            else:
                pnl_r = (entry.entry_price - exit_price) / (stop - entry.entry_price)

        trades.append(
            Trade(
                entry_time=entry_candle.timestamp,
                entry_price=entry.entry_price,
                direction=entry.direction,
                mmxm_phase=entry.mmxm_phase,
                entry_method=entry.method,
                ob_tradable=entry.ob_tradable,
                ob_id=entry.ob_id,
                exit_time=exit_time,
                exit_price=exit_price,
                pnl_r=pnl_r,
            )
        )

    return trades
