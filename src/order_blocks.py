"""Order block detection and classification."""
from __future__ import annotations

from src.models import Candle, MmxmPhase, OrderBlock


def _swing_levels(candles: list[Candle], idx: int, lookback: int = 5) -> tuple[float, float]:
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

        if impulse_size < 2 * ob_range:
            continue

        direction = "bullish" if is_bullish_ob else "bearish"
        has_imbalance = _has_fvg(candles, idx, direction)

        swing_high, swing_low = _swing_levels(candles, idx)
        has_bos = impulse_high > swing_high if direction == "bullish" else impulse_low < swing_low

        phase = phases[idx]
        after_sweep = phase.sweep is not None
        near_level = False
        if direction == "bullish":
            near_level = abs(candle.low - swing_low) / candle.close < 0.005
        else:
            near_level = abs(candle.high - swing_high) / candle.close < 0.005
        near_level = near_level or after_sweep

        fail_reasons = []
        if not has_imbalance:
            fail_reasons.append("no_fvg")
        if not has_bos:
            fail_reasons.append("no_bos")
        if not near_level:
            fail_reasons.append("not_near_level")
        tradable = len(fail_reasons) == 0

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
                fail_reasons=fail_reasons,
            )
        )
        ob_id += 1

    return order_blocks
