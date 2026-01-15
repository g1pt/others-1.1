"""Paper execution package."""

from .config import PaperEngineConfig
from .engine import EngineConfig, ExecutionEngine, ExecutionMode, run_paper_execute
from .ledger import EquityLedger, Ledger
from .models import (
    CloseReason,
    DailyLedger,
    DailySnapshot,
    PaperTrade,
    SignalEvent,
    SignalPayload,
    TradeDecision,
    TradeSignal,
    TradeStatus,
)
from .risk import (
    RiskConfig,
    RiskLimits,
    can_take_trade,
    can_open_trade,
    compute_position_size,
    compute_qty,
    compute_sl,
    daily_key,
)
from .tp_routing import compute_tp, resolve_tp_strategy
from .state_machine import TradeStateMachine

__all__ = [
    "ExecutionEngine",
    "ExecutionMode",
    "EngineConfig",
    "run_paper_execute",
    "PaperEngineConfig",
    "EquityLedger",
    "Ledger",
    "PaperTrade",
    "SignalEvent",
    "SignalPayload",
    "TradeSignal",
    "DailySnapshot",
    "DailyLedger",
    "CloseReason",
    "TradeDecision",
    "TradeStatus",
    "RiskConfig",
    "RiskLimits",
    "TradeStateMachine",
    "can_take_trade",
    "can_open_trade",
    "compute_position_size",
    "compute_qty",
    "compute_sl",
    "daily_key",
    "compute_tp",
    "resolve_tp_strategy",
]
