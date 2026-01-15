"""Execution models for paper trading."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TradeStatus(str, Enum):
    NEW = "NEW"
    OPEN = "OPEN"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_TIME = "CLOSED_TIME"
    INVALID = "INVALID"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class RejectionReason:
    code: str
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalPayload:
    symbol: str
    timeframe: str
    ruleset_id: str
    entry_time: str
    entry_price: float
    direction: str
    entry_type: str
    phase: str
    ob_tradable: bool


@dataclass
class PaperTrade:
    trade_id: str
    symbol: str
    timeframe: str
    ruleset_id: str
    entry_time: str
    entry_price: float
    direction: str
    sl_price: float
    tp_price: float
    size: float
    status: TradeStatus
    exit_time: str | None = None
    exit_price: float | None = None
    pnl_r: float | None = None
    pnl_cash: float | None = None
    rejection_reason: RejectionReason | None = None


@dataclass(frozen=True)
class TradeDecision:
    status: TradeStatus
    trade: PaperTrade | None = None
    reason: RejectionReason | None = None
