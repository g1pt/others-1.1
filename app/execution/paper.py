import uuid
from datetime import datetime

from app.config import SPREAD_SLIPPAGE_R_PENALTY
from app.models import Order, TVWebhook
from app.utils.time import utc_now


def _normalize_side(side: str) -> str:
    normalized = {
        "buy": "buy",
        "long": "buy",
        "sell": "sell",
        "short": "sell",
    }.get(side.lower())
    if normalized is None:
        raise ValueError("Invalid side")
    return normalized


def _stop_distance(symbol: str, price: float) -> float:
    if symbol == "EURUSD":
        return price * 0.001
    return price * 0.002


def build_order_from_signal(
    payload: TVWebhook,
    state_equity: float,
    risk_pct: float,
    symbol: str,
    timeframe: str,
    side: str,
    now: datetime | None = None,
) -> Order:
    normalized_side = _normalize_side(side)
    entry_price = payload.price
    stop_distance = _stop_distance(symbol, entry_price)
    direction = 1 if normalized_side == "buy" else -1
    stop_price = entry_price - (direction * stop_distance)
    tp_price = entry_price + (direction * stop_distance * 2)

    risk_cash = state_equity * risk_pct
    stop_gap = abs(entry_price - stop_price)
    qty = risk_cash / stop_gap if stop_gap > 0 else 0.0

    opened_at = (now or utc_now()).isoformat()
    return Order(
        id=str(uuid.uuid4()),
        symbol=symbol,
        timeframe=timeframe,
        setup=payload.setup,
        side=normalized_side,
        entry_price=entry_price,
        stop_price=stop_price,
        tp_price=tp_price,
        risk_pct=risk_pct,
        qty=qty,
        status="OPEN",
        opened_at=opened_at,
        meta={
            "risk_cash": risk_cash,
            "entry_type": payload.entry_type,
            "phase": payload.phase,
            "ob_tradability": payload.ob_tradability,
            "htf_bias": payload.htf_bias,
            "session": payload.session,
            "entry_variant": payload.entry_variant,
            "level_type": payload.level_type,
            "near_level_dist": payload.near_level_dist,
            "sweep": payload.sweep,
            "bos": payload.bos,
            "fvg_size": payload.fvg_size,
        },
    )


def simulate_close(order: Order, payload: TVWebhook, now: datetime | None = None) -> tuple[float, float, str]:
    pnl_r = payload.sim_outcome_r if payload.sim_outcome_r is not None else 1.0
    pnl_r -= SPREAD_SLIPPAGE_R_PENALTY
    risk_cash = order.meta.get("risk_cash", 0.0)
    pnl_cash = pnl_r * risk_cash

    order.status = "CLOSED"
    order.closed_at = (now or utc_now()).isoformat()
    order.pnl_r = pnl_r
    order.pnl_cash = pnl_cash
    return pnl_r, pnl_cash, "simulated_exit"
