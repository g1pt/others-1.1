"""Research stubs for refinement entry experimentation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.entries import EntrySignal
from src.models import Trade
from src.order_blocks import OrderBlock


class RefinementMethod(Enum):
    OB_OPEN = "ob_open"
    OB_MID = "ob_mid"
    FVG_50 = "fvg_50"


class TimingMethod(Enum):
    PRE_SWEEP = "pre_sweep"
    POST_BOS = "post_bos"
    FIRST_RETRACE = "first_retrace"
    SECOND_RETRACE = "second_retrace"


@dataclass(frozen=True)
class RefinementExperiment:
    method: RefinementMethod
    timing: TimingMethod
    entry_timeframe: str
    execution_timeframe: str


def refinement_entry_price(order_block: OrderBlock, method: RefinementMethod) -> float:
    """Return a candidate entry price for a refinement method."""
    raise NotImplementedError("Implement refinement price selection for OB/FVG variants.")


def timing_allows_entry(entry: EntrySignal, method: TimingMethod) -> bool:
    """Determine whether a timing rule allows the entry."""
    raise NotImplementedError("Implement timing filters for sweep/BOS/retrace timing.")


def choose_timeframe_combo(entry_timeframe: str, execution_timeframe: str) -> bool:
    """Validate or score an entry/execution timeframe combination."""
    raise NotImplementedError("Implement timeframe combination evaluation.")


def evaluate_rr_options(
    trades: list[Trade],
    *,
    rr_targets: list[float],
    partial_take_profit: bool = False,
) -> dict[str, float]:
    """Return evaluation metrics for different R:R targets or partial TP rules."""
    raise NotImplementedError("Implement R:R optimization evaluation.")
