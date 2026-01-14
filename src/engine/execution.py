from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from src.engine.orders import Order
from src.engine.risk import calculate_sl_tp, position_size
from src.storage.file_log import append_jsonl


def paper_execute(
    validated_signal: Mapping[str, Any],
    *,
    equity: float = 10000.0,
    risk_per_trade: float = 0.005,
    rr: float = 2.0,
    sl_pct: float = 0.002,
) -> Order:
    entry_price = float(validated_signal["price"])
    direction = str(validated_signal["direction"]).lower()
    sl, tp = calculate_sl_tp(entry_price, direction, sl_pct=sl_pct, rr=rr)
    size = position_size(
        entry_price=entry_price,
        stop_loss=sl,
        equity=equity,
        risk_per_trade=risk_per_trade,
    )
    order = Order.create(
        received_utc=datetime.utcnow(),
        symbol=str(validated_signal["symbol"]),
        timeframe=str(validated_signal["timeframe"]),
        setup=str(validated_signal["setup"]),
        direction=direction,
        entry_price=entry_price,
        sl=sl,
        tp=tp,
        rr=rr,
        sl_pct=sl_pct,
        risk_per_trade=risk_per_trade,
        equity_snapshot=equity,
        size=size,
    )
    append_jsonl("logs/paper_orders.log", order.to_dict())
    return order
