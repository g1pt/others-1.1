"""Entry detection for MMXM/SMC research."""
from __future__ import annotations

from dataclasses import dataclass

from src.models import Candle
from src.mmxm import MmxmPhase
from src.order_blocks import OrderBlock


@dataclass(frozen=True)
class EntrySignal:
    index: int
    timestamp: str
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    method: str
    mmxm_phase: str
    ob_tradable: bool
    ob_id: int


def _impulse_range(candles: list[Candle], ob: OrderBlock) -> tuple[float, float]:
    window = candles[ob.index + 1 : ob.impulse_end + 1]
    return max(c.high for c in window), min(c.low for c in window)


def _find_retest_index(candles: list[Candle], ob: OrderBlock, start: int) -> int | None:
    for idx in range(start, len(candles)):
        candle = candles[idx]
        if ob.direction == "bullish" and candle.low <= ob.high:
            return idx
        if ob.direction == "bearish" and candle.high >= ob.low:
            return idx
    return None


def generate_entries(
    candles: list[Candle],
    order_blocks: list[OrderBlock],
    phases: list[MmxmPhase],
    risk_reward: float = 2.0,
) -> list[EntrySignal]:
    """Generate entry signals across methods."""
    entries: list[EntrySignal] = []

    for ob in order_blocks:
        impulse_high, impulse_low = _impulse_range(candles, ob)
        direction = ob.direction
        phase = phases[ob.index].phase

        stop_price = ob.low if direction == "bullish" else ob.high
        risk = (ob.open - stop_price) if direction == "bullish" else (stop_price - ob.open)
        if risk <= 0:
            continue

        risk_entry_index = _find_retest_index(candles, ob, ob.impulse_end + 1)
        if risk_entry_index is not None:
            entry_price = ob.open
            target_price = entry_price + risk_reward * risk if direction == "bullish" else entry_price - risk_reward * risk
            entries.append(
                EntrySignal(
                    index=risk_entry_index,
                    timestamp=candles[risk_entry_index].timestamp,
                    direction=direction,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    method="Risk Entry",
                    mmxm_phase=phase,
                    ob_tradable=ob.tradable,
                    ob_id=ob.ob_id,
                )
            )

        if ob.has_bos:
            confirmation_index = _find_retest_index(candles, ob, ob.impulse_end + 1)
            if confirmation_index is not None:
                entry_price = (impulse_high + impulse_low) / 2
                risk = abs(entry_price - stop_price)
                if risk > 0:
                    target_price = (
                        entry_price + risk_reward * risk
                        if direction == "bullish"
                        else entry_price - risk_reward * risk
                    )
                    entries.append(
                        EntrySignal(
                            index=confirmation_index,
                            timestamp=candles[confirmation_index].timestamp,
                            direction=direction,
                            entry_price=entry_price,
                            stop_price=stop_price,
                            target_price=target_price,
                            method="Confirmation Entry",
                            mmxm_phase=phase,
                            ob_tradable=ob.tradable,
                            ob_id=ob.ob_id,
                        )
                    )

        if ob.has_imbalance:
            refinement_index = _find_retest_index(candles, ob, ob.impulse_end + 1)
            if refinement_index is not None:
                entry_price = (ob.open + ob.close) / 2
                risk = abs(entry_price - stop_price)
                if risk > 0:
                    target_price = (
                        entry_price + risk_reward * risk
                        if direction == "bullish"
                        else entry_price - risk_reward * risk
                    )
                    entries.append(
                        EntrySignal(
                            index=refinement_index,
                            timestamp=candles[refinement_index].timestamp,
                            direction=direction,
                            entry_price=entry_price,
                            stop_price=stop_price,
                            target_price=target_price,
                            method="Refinement Entry",
                            mmxm_phase=phase,
                            ob_tradable=ob.tradable,
                            ob_id=ob.ob_id,
                        )
                    )

        continuation_index = _find_retest_index(candles, ob, ob.impulse_end + 3)
        if continuation_index is not None:
            entry_price = candles[continuation_index].close
            risk = abs(entry_price - stop_price)
            if risk > 0:
                target_price = (
                    entry_price + risk_reward * risk
                    if direction == "bullish"
                    else entry_price - risk_reward * risk
                )
                entries.append(
                    EntrySignal(
                        index=continuation_index,
                        timestamp=candles[continuation_index].timestamp,
                        direction=direction,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        method="Continuation Entry",
                        mmxm_phase=phase,
                        ob_tradable=ob.tradable,
                        ob_id=ob.ob_id,
                    )
                )

    return entries
