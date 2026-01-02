"""MMXM phase detection."""
from __future__ import annotations

from dataclasses import dataclass

from src.data import Candle


@dataclass(frozen=True)
class MmxmPhase:
    index: int
    timestamp: str
    phase: str
    range_high: float
    range_low: float
    sweep: str | None = None


def detect_mmxm_phases(candles: list[Candle], lookback: int = 20) -> list[MmxmPhase]:
    """Detect MMXM phases using range, sweep, and breakout logic."""
    phases: list[MmxmPhase] = []
    current_phase = "Accumulation"
    manipulation_countdown = 0
    distribution_countdown = 0

    for idx, candle in enumerate(candles):
        window_start = max(0, idx - lookback)
        window = candles[window_start:idx] or [candle]
        range_high = max(c.high for c in window)
        range_low = min(c.low for c in window)
        sweep = None

        if candle.high > range_high and candle.close < range_high:
            sweep = "SweepHigh"
        elif candle.low < range_low and candle.close > range_low:
            sweep = "SweepLow"

        if sweep:
            current_phase = "Manipulation"
            manipulation_countdown = 3
            distribution_countdown = 0
        elif manipulation_countdown > 0:
            manipulation_countdown -= 1
        elif distribution_countdown > 0:
            distribution_countdown -= 1
        else:
            current_phase = "Accumulation"

        if current_phase == "Manipulation":
            breakout_high = candle.close > range_high
            breakout_low = candle.close < range_low
            if breakout_high or breakout_low:
                current_phase = "Distribution"
                distribution_countdown = 10

        phases.append(
            MmxmPhase(
                index=idx,
                timestamp=candle.timestamp,
                phase=current_phase,
                range_high=range_high,
                range_low=range_low,
                sweep=sweep,
            )
        )

    return phases
