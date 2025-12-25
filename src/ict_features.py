"""ICT feature detection helpers.

Keep research-only logic here.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.data import Candle


@dataclass(frozen=True)
class SweepSignal:
    armed: bool
    swept: bool


@dataclass(frozen=True)
class BosSignal:
    bullish: bool
    bearish: bool


@dataclass(frozen=True)
class FvgSignal:
    bullish: bool
    bearish: bool
    size: float


@dataclass(frozen=True)
class FibContext:
    in_premium: bool
    in_discount: bool


def detect_sweep(previous: Candle, current: Candle) -> SweepSignal:
    """Placeholder sweep detection."""
    _ = previous
    _ = current
    return SweepSignal(armed=False, swept=False)


def detect_bos(previous: Candle, current: Candle) -> BosSignal:
    """Placeholder BOS/MSS detection."""
    _ = previous
    _ = current
    return BosSignal(bullish=False, bearish=False)


def detect_fvg(previous: Candle, current: Candle) -> FvgSignal:
    """Placeholder FVG detection."""
    _ = previous
    _ = current
    return FvgSignal(bullish=False, bearish=False, size=0.0)


def compute_fib_context(high: float, low: float, price: float) -> FibContext:
    """Return fib premium/discount context."""
    midpoint = (high + low) / 2
    return FibContext(in_premium=price > midpoint, in_discount=price <= midpoint)
