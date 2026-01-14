import json
from pathlib import Path
from typing import Any

from app.config import HARD_MAX_DD_PCT, INITIAL_EQUITY, LOGS_DIR
from app.models import LedgerState, Order
from app.utils.time import utc_date_str

LEDGER_STATE_PATH = LOGS_DIR / "ledger_state.json"
PAPER_ORDERS_LOG = LOGS_DIR / "paper_orders.log"
PAPER_FILLS_LOG = LOGS_DIR / "paper_fills.log"
PAPER_TRADES_LOG = LOGS_DIR / "paper_trades.log"


def ensure_logs_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _default_state() -> LedgerState:
    today = utc_date_str()
    return LedgerState(
        equity=INITIAL_EQUITY,
        high_watermark=INITIAL_EQUITY,
        max_dd_pct=0.0,
        daily_date=today,
        daily_start_equity=INITIAL_EQUITY,
        daily_dd_pct=0.0,
        trades_today=0,
        consec_losses=0,
    )


def load_state() -> LedgerState:
    ensure_logs_dir()
    if not LEDGER_STATE_PATH.exists():
        return _default_state()
    try:
        data = json.loads(LEDGER_STATE_PATH.read_text(encoding="utf-8"))
        return LedgerState(**data)
    except (json.JSONDecodeError, OSError, ValueError):
        return _default_state()


def save_state(state: LedgerState) -> None:
    ensure_logs_dir()
    temp_path = LEDGER_STATE_PATH.with_suffix(".tmp")
    payload = state.model_dump()
    payload["max_dd_pct"] = min(payload.get("max_dd_pct", 0.0), HARD_MAX_DD_PCT)
    temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(LEDGER_STATE_PATH)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    ensure_logs_dir()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False))
        handle.write("\n")


def log_paper_trade(event: dict[str, Any]) -> None:
    append_jsonl(PAPER_TRADES_LOG, event)


def log_order(order: Order) -> None:
    append_jsonl(PAPER_ORDERS_LOG, order.model_dump())


def log_fill(order: Order, closed_reason: str) -> None:
    data = order.model_dump()
    data["closed_reason"] = closed_reason
    append_jsonl(PAPER_FILLS_LOG, data)
