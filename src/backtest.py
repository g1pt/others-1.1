"""Backtest loop for research runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.data import Candle
from src.entries import EntrySignal, generate_entries
from src.mmxm import MmxmPhase, detect_mmxm_phases
from src.models import Trade
from src.order_blocks import OrderBlock, detect_order_blocks
from src.simulation import simulate_trades


@dataclass(frozen=True)
class BacktestResult:
    trades: list[Trade]
    phases: list[MmxmPhase]
    order_blocks: list[OrderBlock]
    entry_signals: list[EntrySignal]


def run_backtest(candles: Iterable[Candle]) -> BacktestResult:
    """Run research backtest pipeline."""
    candle_list = list(candles)
    phases = detect_mmxm_phases(candle_list)
    order_blocks = detect_order_blocks(candle_list, phases)
    entries = generate_entries(candle_list, order_blocks, phases)
    trades = simulate_trades(candle_list, entries)
    return BacktestResult(
        trades=trades,
        phases=phases,
        order_blocks=order_blocks,
        entry_signals=entries,
    )
