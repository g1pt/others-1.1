from fastapi import FastAPI, HTTPException

from app.config import EXECUTION_MODE, SYMBOL_MAP, SYMBOL_RULESETS, WEBHOOK_SECRET
from app.execution import ledger, paper, research, risk
from app.models import TVWebhook
from app.utils.time import utc_now

app = FastAPI()


@app.on_event("startup")
def _startup() -> None:
    ledger.ensure_logs_dir()


def normalize_timeframe(timeframe: int | str) -> str:
    if isinstance(timeframe, int):
        return str(timeframe)
    if isinstance(timeframe, str) and timeframe.isdigit():
        return timeframe
    raise ValueError("Invalid timeframe")


def normalize_side(direction: str) -> str:
    normalized = {
        "buy": "buy",
        "long": "buy",
        "sell": "sell",
        "short": "sell",
    }.get(direction.lower())
    if normalized is None:
        raise ValueError("Invalid direction")
    return normalized


@app.post("/webhook/tradingview")
def tradingview_webhook(payload: TVWebhook):
    if payload.secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=400, detail="Invalid secret")
    internal_symbol = SYMBOL_MAP.get(payload.symbol)
    if internal_symbol is None:
        raise HTTPException(status_code=400, detail="Symbol not allowed")
    ruleset = SYMBOL_RULESETS.get(internal_symbol)
    if not ruleset or not ruleset.get("enabled"):
        raise HTTPException(status_code=400, detail="Symbol not enabled")
    try:
        timeframe_value = normalize_timeframe(payload.timeframe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if timeframe_value not in ruleset["allowed_timeframes"]:
        raise HTTPException(status_code=400, detail="Timeframe not allowed")
    if payload.setup != ruleset["require_setup"]:
        raise HTTPException(status_code=400, detail="Setup not allowed")
    if payload.entry_type != ruleset["entry_type"]:
        raise HTTPException(status_code=400, detail="Entry type not allowed")
    if payload.phase != ruleset["phase"]:
        raise HTTPException(status_code=400, detail="Phase not allowed")
    if payload.ob_tradability != ruleset["ob_tradability"]:
        raise HTTPException(status_code=400, detail="OB tradability not allowed")
    side_value = payload.direction or payload.side
    if not side_value:
        raise HTTPException(status_code=400, detail="Direction not provided")
    try:
        normalized_side = normalize_side(side_value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    event = payload.model_dump()
    event["symbol_raw"] = payload.symbol
    event["symbol"] = internal_symbol
    event["timeframe"] = timeframe_value
    event["direction"] = normalized_side
    event["received_utc"] = utc_now().isoformat()

    ledger.log_paper_trade(event)

    state = ledger.load_state()
    state_before = state.model_dump()
    now = utc_now()
    can_trade, reason = risk.can_open_trade(state, now, internal_symbol)

    if not can_trade:
        research.log_research_event(
            {
                "type": "accepted_trigger_no_trade",
                "reason": reason,
                "payload": event,
                "state_before": state_before,
                "state_after": state.model_dump(),
            }
        )
        ledger.save_state(state)
        return {"ok": True, "accepted": True, "trade": False, "reason": reason}

    if EXECUTION_MODE == "LOG_ONLY":
        research.log_research_event(
            {
                "type": "accepted_trigger_no_trade",
                "reason": "log_only",
                "payload": event,
                "state_before": state_before,
                "state_after": state.model_dump(),
            }
        )
        ledger.save_state(state)
        return {"ok": True, "accepted": True, "trade": False, "reason": "log_only"}

    if EXECUTION_MODE == "LIVE_BROKER":
        raise HTTPException(status_code=400, detail="Live broker mode disabled")

    if EXECUTION_MODE != "PAPER_SIM":
        raise HTTPException(status_code=400, detail="Unknown execution mode")

    risk_pct = risk.compute_risk_pct(state)
    order = paper.build_order_from_signal(
        payload=payload,
        state_equity=state.equity,
        risk_pct=risk_pct,
        symbol=internal_symbol,
        timeframe=timeframe_value,
        side=normalized_side,
        now=now,
    )
    state.trades_today += 1

    pnl_r, pnl_cash, closed_reason = paper.simulate_close(order, payload, now=now)
    risk.update_after_close(state, pnl_cash, won=pnl_r > 0)

    ledger.log_order(order)
    ledger.log_fill(order, closed_reason)
    ledger.save_state(state)

    research.log_research_event(
        {
            "type": "paper_simulated_trade",
            "payload": event,
            "state_before": state_before,
            "state_after": state.model_dump(),
            "risk_pct": risk_pct,
            "order_id": order.id,
            "pnl_r": pnl_r,
            "pnl_cash": pnl_cash,
            "equity_before": state_before["equity"],
            "equity_after": state.equity,
            "closed_reason": closed_reason,
        }
    )

    return {
        "ok": True,
        "accepted": True,
        "trade": True,
        "order_id": order.id,
        "pnl_r": pnl_r,
        "pnl_cash": pnl_cash,
        "equity_after": state.equity,
    }
