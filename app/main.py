from fastapi import FastAPI

from app.candles import load_candles
from app.config import EXECUTION_MODE, WEBHOOK_SECRET
from app.models import TVWebhook
from app.paper_engine import (
    build_order,
    can_open_trade,
    compute_risk_pct,
    normalize_direction,
    normalize_timeframe,
    parse_timestamp,
    resolve_symbol,
    simulate_backfill,
    validate_signal,
)
from app.storage import (
    ensure_logs_dir,
    load_orders_sorted,
    load_state,
    log_order,
    log_paper_trade,
    save_state,
)
from app.utils.time import utc_now

app = FastAPI()


@app.on_event("startup")
def _startup() -> None:
    ensure_logs_dir()


def _response(accepted: bool, reason: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {"ok": True, "accepted": accepted, "reason": reason}
    payload.update(extra)
    return payload


@app.post("/webhook/tradingview")
def tradingview_webhook(payload: TVWebhook) -> dict[str, object]:
    now = utc_now()
    event = payload.model_dump()
    event["received_utc"] = now.isoformat()
    parsed_timestamp = parse_timestamp(payload.timestamp)
    event["timestamp_parsed"] = parsed_timestamp.isoformat() if parsed_timestamp else None

    if payload.secret != WEBHOOK_SECRET:
        event["blocked_reason"] = "blocked_bad_secret"
        log_paper_trade(event)
        return _response(False, "blocked_bad_secret")

    internal_symbol = resolve_symbol(payload.symbol)
    if internal_symbol is None:
        event["blocked_reason"] = "blocked_symbol_not_allowed"
        log_paper_trade(event)
        return _response(False, "blocked_symbol_not_allowed")

    try:
        timeframe_value = normalize_timeframe(payload.timeframe)
    except ValueError:
        event["blocked_reason"] = "blocked_bad_timeframe"
        log_paper_trade(event)
        return _response(False, "blocked_bad_timeframe")

    side_value = payload.direction or payload.side
    if not side_value:
        event["blocked_reason"] = "blocked_missing_direction"
        log_paper_trade(event)
        return _response(False, "blocked_missing_direction")
    try:
        normalized_side = normalize_direction(side_value)
    except ValueError:
        event["blocked_reason"] = "blocked_bad_direction"
        log_paper_trade(event)
        return _response(False, "blocked_bad_direction")

    event["symbol"] = internal_symbol
    event["timeframe"] = timeframe_value
    event["direction"] = normalized_side

    gate_ok, gate_reason = validate_signal(payload, internal_symbol, timeframe_value)
    if not gate_ok:
        event["blocked_reason"] = "blocked_gate_mismatch"
        event["blocked_detail"] = gate_reason
        log_paper_trade(event)
        return _response(False, "blocked_gate_mismatch", detail=gate_reason)

    log_paper_trade(event)

    if EXECUTION_MODE == "LIVE_BROKER":
        return _response(False, "blocked_live_broker_disabled")
    if EXECUTION_MODE == "LOG_ONLY":
        return _response(True, "log_only", trade=False)
    if EXECUTION_MODE != "PAPER_SIM":
        return _response(False, "blocked_unknown_execution_mode")

    state = load_state()
    can_trade, reason = can_open_trade(state, now)
    if not can_trade:
        return _response(False, reason)

    risk_pct = compute_risk_pct(state)
    order = build_order(
        payload=payload,
        equity=state.equity,
        risk_pct=risk_pct,
        symbol=internal_symbol,
        timeframe=timeframe_value,
        direction=normalized_side,
        now=now,
    )
    state.trades_today += 1
    log_order(order)
    save_state(state)

    return _response(True, "accepted", trade=True, order_id=order.id)


@app.get("/status")
def status() -> dict[str, object]:
    state = load_state()
    orders = load_orders_sorted()
    open_count = len([order for order in orders if order.status == "OPEN"])
    recent_orders = [order.model_dump() for order in orders[-5:]]
    return {
        "equity": state.equity,
        "high_watermark": state.high_watermark,
        "open_orders": open_count,
        "last_orders": recent_orders,
    }


@app.get("/orders")
def orders(limit: int = 50) -> dict[str, object]:
    orders_list = load_orders_sorted()
    trimmed = orders_list[-limit:] if limit > 0 else orders_list
    return {"orders": [order.model_dump() for order in trimmed]}


@app.post("/simulate/backfill")
def simulate_backfill_endpoint(symbol: str | None = None, timeframe: str | None = None) -> dict[str, object]:
    state = load_state()
    orders_list = load_orders_sorted()

    if symbol:
        orders_list = [order for order in orders_list if order.symbol == symbol]
    if timeframe:
        orders_list = [order for order in orders_list if order.timeframe == timeframe]

    open_orders = [order for order in orders_list if order.status == "OPEN"]
    if not open_orders:
        return _response(True, "no_open_orders", closed=0)

    closed_total = 0
    grouped: dict[tuple[str, str], list] = {}
    for order in open_orders:
        grouped.setdefault((order.symbol, order.timeframe), []).append(order)

    for (order_symbol, order_timeframe), group_orders in grouped.items():
        candles = load_candles(order_symbol, order_timeframe)
        if not candles:
            return _response(False, "blocked_no_candles")
        closed_orders = simulate_backfill(state, group_orders, candles)
        for order in closed_orders:
            log_order(order)
        closed_total += len(closed_orders)

    save_state(state)
    return _response(True, "backfill_complete", closed=closed_total, equity=state.equity)
