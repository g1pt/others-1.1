from typing import Any

from app.execution.ledger import append_jsonl
from app.config import LOGS_DIR

RESEARCH_LOG = LOGS_DIR / "research_events.log"


def log_research_event(event: dict[str, Any]) -> None:
    append_jsonl(RESEARCH_LOG, event)
