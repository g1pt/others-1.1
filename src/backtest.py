"""Backtest loop for research runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.data import Candle
from src.ict_features import BosSignal, FvgSignal, SweepSignal


@dataclass(frozen=True)
class Trade:
    entry_time: str
    entry_price: float
    direction: str
    exit_time: str | None = None
    exit_price: float | None = None
    pnl_r: float | None = None


@dataclass(frozen=True)
class BacktestResult:
    trades: list[Trade]


def run_backtest(candles: Iterable[Candle]) -> BacktestResult:
    """Placeholder backtest runner.

    Implement strategy evaluation in future iterations.
    """
    _ = list(candles)
    return BacktestResult(trades=[])


def collect_signals(candles: Iterable[Candle]) -> list[tuple[SweepSignal, BosSignal, FvgSignal]]:
    """Collect feature signals for logging/debugging."""
    _ = list(candles)
    return []
