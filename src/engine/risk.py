from __future__ import annotations


def calculate_sl_tp(
    entry_price: float,
    direction: str,
    *,
    sl_pct: float,
    rr: float,
) -> tuple[float, float]:
    if direction.lower() == "buy":
        sl = entry_price * (1 - sl_pct)
        tp = entry_price * (1 + sl_pct * rr)
    else:
        sl = entry_price * (1 + sl_pct)
        tp = entry_price * (1 - sl_pct * rr)
    return sl, tp


def position_size(
    *,
    entry_price: float,
    stop_loss: float,
    equity: float,
    risk_per_trade: float,
) -> float:
    risk_cash = equity * risk_per_trade
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance == 0:
        return 0.0
    return risk_cash / stop_distance
