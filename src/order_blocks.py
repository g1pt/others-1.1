"""Order block detection and classification."""
from __future__ import annotations

from dataclasses import dataclass

from collections import Counter

from src.data import Candle
from src.mmxm import MmxmPhase


IMPULSE_MULTIPLIER = 2.0
SWING_LOOKBACK = 5
NEAR_LEVEL_PCT = 0.005


@dataclass(frozen=True)
class OrderBlock:
    ob_id: int
    index: int
    timestamp: str
    direction: str
    open: float
    high: float
    low: float
    close: float
    range: float
    impulse_end: int
    has_imbalance: bool
    has_bos: bool
    near_level: bool
    after_sweep: bool
    tradable: bool
    tradability_failures: tuple[str, ...]


def _swing_levels(
    candles: list[Candle], idx: int, lookback: int = SWING_LOOKBACK
) -> tuple[float, float]:
    window_start = max(0, idx - lookback)
    window = candles[window_start:idx]
    if not window:
        return candles[idx].high, candles[idx].low
    return max(c.high for c in window), min(c.low for c in window)


def _has_fvg(candles: list[Candle], idx: int, direction: str) -> bool:
    if idx + 2 >= len(candles):
        return False
    prev = candles[idx]
    next_candle = candles[idx + 1]
    if direction == "bullish":
        return next_candle.low > prev.high
    return next_candle.high < prev.low


def detect_order_blocks(candles: list[Candle], phases: list[MmxmPhase]) -> list[OrderBlock]:
    """Detect order blocks and classify tradability."""
    order_blocks: list[OrderBlock] = []
    ob_id = 0

    for idx in range(1, len(candles) - 4):
        candle = candles[idx]
        is_bullish_ob = candle.close < candle.open
        is_bearish_ob = candle.close > candle.open
        if not (is_bullish_ob or is_bearish_ob):
            continue

        impulse_window = candles[idx + 1 : idx + 4]
        impulse_high = max(c.high for c in impulse_window)
        impulse_low = min(c.low for c in impulse_window)
        ob_range = candle.high - candle.low
        impulse_size = impulse_high - impulse_low

        if is_bullish_ob and impulse_high <= candle.high:
            continue
        if is_bearish_ob and impulse_low >= candle.low:
            continue

        if impulse_size < IMPULSE_MULTIPLIER * ob_range:
            continue

        direction = "bullish" if is_bullish_ob else "bearish"
        has_imbalance = _has_fvg(candles, idx, direction)

        swing_high, swing_low = _swing_levels(candles, idx)
        has_bos = impulse_high > swing_high if direction == "bullish" else impulse_low < swing_low

        phase = phases[idx]
        after_sweep = phase.sweep is not None
        near_level = False
        if direction == "bullish":
            near_level = abs(candle.low - swing_low) / candle.close < NEAR_LEVEL_PCT
        else:
            near_level = abs(candle.high - swing_high) / candle.close < NEAR_LEVEL_PCT
        near_level = near_level or after_sweep

        tradability_failures = []
        if not has_imbalance:
            tradability_failures.append("failed_imbalance")
        if not has_bos:
            tradability_failures.append("failed_bos")
        if not near_level:
            tradability_failures.append("failed_location")
        tradability_failures_tuple = tuple(tradability_failures)
        tradable = not tradability_failures_tuple

        order_blocks.append(
            OrderBlock(
                ob_id=ob_id,
                index=idx,
                timestamp=candle.timestamp,
                direction=direction,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                range=ob_range,
                impulse_end=idx + 3,
                has_imbalance=has_imbalance,
                has_bos=has_bos,
                near_level=near_level,
                after_sweep=after_sweep,
                tradable=tradable,
                tradability_failures=tradability_failures_tuple,
            )
        )
        ob_id += 1

    return order_blocks


def tradability_failure_counts(order_blocks: list[OrderBlock]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for block in order_blocks:
        if block.tradable:
            continue
        counts.update(block.tradability_failures)
    return counts
