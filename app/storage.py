import json
from pathlib import Path
from typing import Any, Iterable

from app.config import INITIAL_EQUITY, LOGS_DIR
from app.models import EquityState, PaperOrder
from app.utils.time import utc_date_str

PAPER_TRADES_LOG = LOGS_DIR / "paper_trades.log"
PAPER_ORDERS_LOG = LOGS_DIR / "paper_orders.jsonl"
EQUITY_STATE_PATH = LOGS_DIR / "equity_state.json"


def ensure_logs_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _default_state() -> EquityState:
    today = utc_date_str()
    return EquityState(
        equity=INITIAL_EQUITY,
        high_watermark=INITIAL_EQUITY,
        max_dd_pct=0.0,
        daily_date=today,
        daily_start_equity=INITIAL_EQUITY,
        daily_dd_pct=0.0,
        trades_today=0,
        consec_losses=0,
    )


def load_state() -> EquityState:
    ensure_logs_dir()
    if not EQUITY_STATE_PATH.exists():
        return _default_state()
    try:
        data = json.loads(EQUITY_STATE_PATH.read_text(encoding="utf-8"))
        return EquityState(**data)
    except (json.JSONDecodeError, OSError, ValueError):
        return _default_state()


def save_state(state: EquityState) -> None:
    ensure_logs_dir()
    temp_path = EQUITY_STATE_PATH.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(state.model_dump(), ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(EQUITY_STATE_PATH)


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    ensure_logs_dir()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False))
        handle.write("\n")


def log_paper_trade(event: dict[str, Any]) -> None:
    append_jsonl(PAPER_TRADES_LOG, event)


def log_order(order: PaperOrder) -> None:
    append_jsonl(PAPER_ORDERS_LOG, order.model_dump())


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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
    return entries


def load_orders() -> list[PaperOrder]:
    entries = _read_jsonl(PAPER_ORDERS_LOG)
    latest: dict[str, PaperOrder] = {}
    for entry in entries:
        try:
            order = PaperOrder(**entry)
        except ValueError:
            continue
        latest[order.id] = order
    return list(latest.values())


def load_orders_sorted() -> list[PaperOrder]:
    orders = load_orders()
    return sorted(orders, key=lambda order: order.opened_utc)
