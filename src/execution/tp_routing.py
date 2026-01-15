"""Routing for take-profit strategies."""
from __future__ import annotations

from .tp_strategies import TP_EQH, TP_FixedR, TPStrategy

TP_ROUTING: dict[str, TPStrategy] = {
    "SP500": TP_FixedR(r_multiple=2.0),
    "SP500_RR4": TP_FixedR(r_multiple=4.0),
    "EURUSD": TP_EQH(),
}


def resolve_tp_strategy(symbol: str, setup: str | None = None) -> TPStrategy:
    """Resolve a TP strategy for a symbol/setup pair."""
    if setup:
        setup_key = f"{symbol}_{setup}"
        if setup_key in TP_ROUTING:
            return TP_ROUTING[setup_key]
    if symbol in TP_ROUTING:
        return TP_ROUTING[symbol]
    return TP_FixedR(r_multiple=2.0)


def compute_tp(
    entry: float,
    sl: float,
    data,
    direction: str,
    symbol: str,
    setup: str | None = None,
) -> tuple[float, str]:
    """Compute TP level and label using the routed strategy."""
    strategy = resolve_tp_strategy(symbol, setup)
    tp_level = strategy.get_tp_level(entry, sl, data, direction)
    return tp_level, strategy.__class__.__name__
