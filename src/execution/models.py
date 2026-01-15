"""Execution models for paper trading."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TradeStatus(str, Enum):
    NEW = "NEW"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
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


@dataclass(frozen=True)
class SignalEvent:
    symbol: str
    timeframe: str
    setup: str
    direction: str
    entry_time: str
    entry_price: float
    entry_type: str
    phase: str
    ob_tradability: str


@dataclass
class PaperTrade:
    trade_id: str
    symbol: str
    timeframe: str
    ruleset_id: str | None
    entry_time: str
    entry_price: float
    direction: str
    sl_price: float
    tp_price: float
    size: float
    status: TradeStatus
    setup: str | None = None
    exit_time: str | None = None
    exit_price: float | None = None
    close_reason: str | None = None
    pnl_r: float | None = None
    pnl_cash: float | None = None
    tp_label: str | None = None
    created_utc: str | None = None
    risk_cash: float | None = None
    rejection_reason: RejectionReason | None = None

    @property
    def entry(self) -> float:
        return self.entry_price

    @property
    def sl(self) -> float:
        return self.sl_price

    @property
    def tp(self) -> float:
        return self.tp_price

    @property
    def qty(self) -> float:
        return self.size

    @property
    def pnl_R(self) -> float | None:
        return self.pnl_r


@dataclass(frozen=True)
class DailySnapshot:
    date: str
    start_equity: float
    end_equity: float
    realized_pnl: float
    max_dd_pct: float
    trades_count: int
    blocked_count: int


@dataclass(frozen=True)
class TradeDecision:
    status: TradeStatus
    trade: PaperTrade | None = None
    reason: RejectionReason | None = None
