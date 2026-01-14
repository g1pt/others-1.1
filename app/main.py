from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from app.config import (
    DEFAULT_EQUITY,
    DEFAULT_RISK_PER_TRADE,
    DEFAULT_RR,
    DEFAULT_SL_PCT,
    LOGS_DIR,
    SYMBOL_MAP,
    WEBHOOK_SECRET,
)
from app.models import TVWebhook
from src.engine import build_default_spec, paper_execute, validate
from src.storage.file_log import append_jsonl

app = FastAPI()
_SPEC = build_default_spec(SYMBOL_MAP)


@app.on_event("startup")
def _startup() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _tail_jsonl(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-limit:]


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/webhook/tradingview")
def tradingview_webhook(payload: TVWebhook) -> dict[str, Any]:
    received = datetime.utcnow().isoformat()
    event = payload.model_dump()
    event["received_utc"] = received

    ok, reason, normalized = validate(event, spec=_SPEC, secret=WEBHOOK_SECRET)
    if not ok:
        event["rejected_reason"] = reason
        append_jsonl("logs/paper_trades.log", event)
        raise HTTPException(status_code=400, detail=reason)

    append_jsonl("logs/paper_trades.log", normalized)
    order = paper_execute(
        normalized,
        equity=DEFAULT_EQUITY,
        risk_per_trade=DEFAULT_RISK_PER_TRADE,
        rr=DEFAULT_RR,
        sl_pct=DEFAULT_SL_PCT,
    )
    return {
        "ok": True,
        "accepted": True,
        "paper": True,
        "order_id": order.id,
        "order": order.to_dict(),
    }


@app.get("/status")
def status() -> dict[str, Any]:
    trades = _tail_jsonl(LOGS_DIR / "paper_trades.log")
    orders = _tail_jsonl(LOGS_DIR / "paper_orders.log")
    return {
        "trades_logged": len(trades),
        "orders_logged": len(orders),
        "last_trades": trades,
        "last_orders": orders,
    }
