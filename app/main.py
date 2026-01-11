import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.config import (
    ALLOWED_SYMBOLS,
    ALLOWED_TIMEFRAMES,
    REQUIRE_SETUP,
    WEBHOOK_SECRET,
)
from app.models import TVWebhook

app = FastAPI()


@app.post("/webhook/tradingview")
def tradingview_webhook(payload: TVWebhook):
    if payload.secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=400, detail="Invalid secret")
    if payload.symbol not in ALLOWED_SYMBOLS:
        raise HTTPException(status_code=400, detail="Symbol not allowed")
    if payload.timeframe not in ALLOWED_TIMEFRAMES:
        raise HTTPException(status_code=400, detail="Timeframe not allowed")
    if payload.setup != REQUIRE_SETUP:
        raise HTTPException(status_code=400, detail="Setup not allowed")
    if payload.entry_type != "Refinement":
        raise HTTPException(status_code=400, detail="Entry type not allowed")
    if payload.phase != "Manipulation":
        raise HTTPException(status_code=400, detail="Phase not allowed")
    if payload.ob_tradability != "Tradable":
        raise HTTPException(status_code=400, detail="OB tradability not allowed")

    event = payload.model_dump()
    event["received_utc"] = datetime.now(timezone.utc).isoformat()

    logs_path = Path(__file__).resolve().parent.parent / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    log_file = logs_path / "paper_trades.log"
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    return {"ok": True, "accepted": True, "paper": True, "event": event}
