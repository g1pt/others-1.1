"""Deterministic exit simulation for paper trades."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ExitResult:
    exit_price: float
    exit_time: str
    reason: str


def _candle_field(candle, field: str) -> float | str:
    if isinstance(candle, dict):
        return candle[field]
    return getattr(candle, field)


def determine_exit(
    trade,
    candles_after_entry: Iterable,
    tie_breaker: str = "SL",
) -> ExitResult | None:
    """Determine exit price/time/reason based on forward candles."""
    last_candle = None
    for candle in candles_after_entry:
        last_candle = candle
        high = float(_candle_field(candle, "high"))
        low = float(_candle_field(candle, "low"))
        if trade.direction == "buy":
            sl_hit = low <= trade.sl_price
            tp_hit = high >= trade.tp_price
        else:
            sl_hit = high >= trade.sl_price
            tp_hit = low <= trade.tp_price

        if sl_hit and tp_hit:
            exit_price = trade.sl_price if tie_breaker.upper() == "SL" else trade.tp_price
            reason = "SL" if tie_breaker.upper() == "SL" else "TP"
            return ExitResult(
                exit_price=exit_price,
                exit_time=str(_candle_field(candle, "timestamp")),
                reason=reason,
            )
        if sl_hit:
            return ExitResult(
                exit_price=trade.sl_price,
                exit_time=str(_candle_field(candle, "timestamp")),
                reason="SL",
            )
        if tp_hit:
            return ExitResult(
                exit_price=trade.tp_price,
                exit_time=str(_candle_field(candle, "timestamp")),
                reason="TP",
            )

    if last_candle is None:
        return None
    return ExitResult(
        exit_price=float(_candle_field(last_candle, "close")),
        exit_time=str(_candle_field(last_candle, "timestamp")),
        reason="TIME_STOP",
    )
