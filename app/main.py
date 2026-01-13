import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from app.config import (
    ALLOWED_TIMEFRAMES,
    REQUIRE_SETUP,
    SYMBOL_MAP,
    WEBHOOK_SECRET,
)
from app.models import TVWebhook

app = FastAPI()
logger = logging.getLogger("paper_trades")
logging.basicConfig(level=logging.INFO)


def normalize_timeframe(timeframe: int | str) -> int:
    if isinstance(timeframe, int):
        return timeframe
    if isinstance(timeframe, str) and timeframe.isdigit():
        return int(timeframe)
    raise ValueError("Invalid timeframe")


def normalize_direction(direction: str) -> str:
    normalized = {
        "buy": "LONG",
        "long": "LONG",
        "sell": "SHORT",
        "short": "SHORT",
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
    try:
        timeframe_value = normalize_timeframe(payload.timeframe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if timeframe_value not in ALLOWED_TIMEFRAMES:
        raise HTTPException(status_code=400, detail="Timeframe not allowed")
    if payload.setup != REQUIRE_SETUP:
        raise HTTPException(status_code=400, detail="Setup not allowed")
    if payload.entry_type != "Refinement":
        raise HTTPException(status_code=400, detail="Entry type not allowed")
    if payload.phase != "Manipulation":
        raise HTTPException(status_code=400, detail="Phase not allowed")
    if payload.ob_tradability != "Tradable":
        raise HTTPException(status_code=400, detail="OB tradability not allowed")
    try:
        normalized_direction = normalize_direction(payload.direction)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    event = payload.model_dump()
    event["symbol_raw"] = payload.symbol
    event["symbol"] = internal_symbol
    event["timeframe"] = str(timeframe_value)
    event["direction"] = normalized_direction
    event["received_utc"] = datetime.now(timezone.utc).isoformat()

    logger.info("paper_trade_event=%s", json.dumps(event, ensure_ascii=False))

    return {"ok": True, "accepted": True, "paper": True, "event": event}
