"""Take-profit strategy definitions."""
from __future__ import annotations

from abc import ABC, abstractmethod


class TPStrategy(ABC):
    """Abstract base class for TP strategy implementations."""

    @abstractmethod
    def get_tp_level(self, entry: float, sl: float, data, direction: str) -> float:
        """Return TP price level. direction: 'buy' or 'sell'."""


class TP_FixedR(TPStrategy):
    """Fixed R-multiple take-profit strategy."""

    def __init__(self, r_multiple: float = 2.0) -> None:
        self.r_multiple = r_multiple

    def get_tp_level(self, entry: float, sl: float, data, direction: str) -> float:
        """Return TP price based on a fixed R multiple."""
        _ = data
        dist = abs(entry - sl)
        if direction == "buy":
            return entry + self.r_multiple * dist
        if direction == "sell":
            return entry - self.r_multiple * dist
        raise ValueError("direction must be 'buy' or 'sell'")


class TP_EQH(TPStrategy):
    """Placeholder strategy for equal highs/lows TP selection.

    Intended to target the most recent equal high/low zone once logic is implemented.
    """

    def get_tp_level(self, entry: float, sl: float, data, direction: str) -> float:
        """Return a placeholder TP until EQH/EQL logic is added."""
        _ = data
        dist = abs(entry - sl)
        if direction == "buy":
            return entry + dist
        if direction == "sell":
            return entry - dist
        raise ValueError("direction must be 'buy' or 'sell'")


class TP_FVG(TPStrategy):
    """Placeholder strategy for fair value gap TP selection.

    Intended to target the nearest FVG boundary once logic is implemented.
    """

    def get_tp_level(self, entry: float, sl: float, data, direction: str) -> float:
        """Return a placeholder TP until FVG logic is added."""
        _ = data
        dist = abs(entry - sl)
        if direction == "buy":
            return entry + dist
        if direction == "sell":
            return entry - dist
        raise ValueError("direction must be 'buy' or 'sell'")


class TP_LiquiditySweep(TPStrategy):
    """Placeholder strategy for liquidity sweep TP selection.

    Intended to target above a swing high or below a swing low once logic is implemented.
    """

    def get_tp_level(self, entry: float, sl: float, data, direction: str) -> float:
        """Return a placeholder TP until liquidity sweep logic is added."""
        _ = data
        dist = abs(entry - sl)
        if direction == "buy":
            return entry + dist
        if direction == "sell":
            return entry - dist
        raise ValueError("direction must be 'buy' or 'sell'")
