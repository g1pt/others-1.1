"""Risk management and equity tracking."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EquityPoint:
    timestamp: str
    equity: float


@dataclass(frozen=True)
class DailyEquity:
    date: str
    equity: float


def update_equity(current_equity: float, pnl_r: float, risk_per_trade: float) -> float:
    """Update equity based on a trade result in R."""
    return current_equity + (pnl_r * risk_per_trade)
