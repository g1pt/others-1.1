"""Backtest loop for research runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.models import Candle
from src.entries import EntrySignal, generate_entries
from src.filtering import (
    ComboFilter,
    EntryFilter,
    filter_entries,
    filter_entry_signals,
    infer_timeframe_minutes,
)
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


def run_backtest(
    candles: Iterable[Candle],
    combo_filter: ComboFilter | None = None,
    timeframe: str | None = None,
    entry_filter: EntryFilter | None = None,
) -> BacktestResult:
    """Run research backtest pipeline."""
    candle_list = list(candles)
    phases = detect_mmxm_phases(candle_list)
    order_blocks = detect_order_blocks(candle_list, phases)
    entries = generate_entries(candle_list, order_blocks, phases)
    if entry_filter is None:
        entry_filter = EntryFilter.refinement_only()
    entries = filter_entries(entries, entry_filter)
    if combo_filter is not None:
        if timeframe is None:
            inferred = infer_timeframe_minutes(candle_list)
            if inferred is None:
                raise ValueError("Unable to infer timeframe for combo filtering.")
            timeframe = f"{inferred}m"
        entries = filter_entry_signals(entries, timeframe, combo_filter)
    trades = simulate_trades(candle_list, entries)
    return BacktestResult(
        trades=trades,
        phases=phases,
        order_blocks=order_blocks,
        entry_signals=entries,
    )
