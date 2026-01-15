from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from app.agent_rules import SYMBOL_MAP, SYMBOL_RULESETS
from app.config import (
    DD_DAILY_STOP,
    DD_HARD_SYSTEM,
    EXECUTION_MODE,
    INITIAL_EQUITY,
    LOGS_DIR,
    MAX_CONSEC_LOSSES,
    MAX_TRADES_PER_DAY,
    RISK_PCT_DEFAULT,
    SL_FIXED_PCT,
    WEBHOOK_SECRET,
)
from app.models import TVWebhook
from src.execution.config import ExecutionMode, PaperEngineConfig, RulesetConfig
from src.execution.ledger import Ledger
from src.execution.paper_engine import PaperEngine
from src.execution.risk import RiskLimits

app = FastAPI()


def _resolve_mode(mode: str) -> ExecutionMode:
    try:
        return ExecutionMode(mode)
    except ValueError:
        return ExecutionMode.LOG_ONLY


def _build_rulesets() -> dict[str, RulesetConfig]:
    rulesets: dict[str, RulesetConfig] = {}
    for symbol, config in SYMBOL_RULESETS.items():
        timeframes = sorted(config.get("allowed_timeframes", []))
        timeframe = timeframes[0] if timeframes else ""
        rulesets[symbol] = RulesetConfig(
            setup_id=config["setup_id"],
            timeframe=timeframe,
            entry_type=config["entry_type"],
            phase=config["phase"],
            ob_tradability=config["ob_tradability"],
            enabled=bool(config.get("enabled", True)),
        )
    return rulesets


_ENGINE = PaperEngine(
    PaperEngineConfig(
        mode=_resolve_mode(EXECUTION_MODE),
        risk_limits=RiskLimits(
            max_trades_per_day=MAX_TRADES_PER_DAY,
            stop_after_consecutive_losses=MAX_CONSEC_LOSSES,
            daily_drawdown_stop_pct=DD_DAILY_STOP,
            hard_max_drawdown_pct=DD_HARD_SYSTEM,
        ),
        risk_per_trade_pct=RISK_PCT_DEFAULT,
        sl_pct=SL_FIXED_PCT,
        symbol_map=SYMBOL_MAP,
        rulesets=_build_rulesets(),
        log_dir=LOGS_DIR,
    ),
    Ledger.load_or_init(LOGS_DIR, initial_equity=INITIAL_EQUITY),
)


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

    if payload.secret != WEBHOOK_SECRET:
        _ENGINE.ledger.record_rejection(event, "bad_secret")
        _ENGINE.ledger.log_event(
            {
                "timestamp": received,
                "accepted": False,
                "reason": "bad_secret",
                "event": event,
            }
        )
        raise HTTPException(status_code=400, detail="secret mismatch")

    accepted, reason, trade_id = _ENGINE.process_signal(event)
    return {
        "ok": True,
        "accepted": accepted,
        "reason": None if accepted else reason,
        "paper": True,
        "trade_id": trade_id,
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
