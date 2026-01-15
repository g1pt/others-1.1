"""Paper execution engine."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable

from .ledger import EquityLedger
from .models import PaperTrade, RejectionReason, SignalPayload, TradeDecision, TradeStatus
from .risk import RiskConfig, can_take_trade
from .state_machine import TradeStateMachine


class ExecutionMode(str, Enum):
    LOG_ONLY = "LOG_ONLY"
    PAPER_SIM = "PAPER_SIM"
    LIVE_BROKER = "LIVE_BROKER"


@dataclass(frozen=True)
class EngineConfig:
    mode: ExecutionMode = ExecutionMode.LOG_ONLY
    risk_per_trade: float = 0.02
    sl_pct: float = 0.002
    rr_default: float = 2.0
    symbol_whitelist: frozenset[str] = frozenset({"SP500"})
    ruleset_whitelist: frozenset[str] = frozenset({"MMXM_4C_D"})
    timeframe_whitelist: frozenset[str] | None = None
    entry_type_required: str = "Refinement"
    phase_required: str = "Manipulation"
    require_tradable_ob: bool = True
    log_dir: Path = Path("logs")


class ExecutionEngine:
    """Paper execution engine for MMXM signals."""

    def __init__(
        self,
        config: EngineConfig,
        ledger: EquityLedger,
        risk_config: RiskConfig,
        candles: Iterable | None = None,
        candle_index: dict[str, int] | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.risk_config = risk_config
        self.candles = list(candles) if candles is not None else []
        self.candle_index = candle_index or {
            candle.timestamp: idx for idx, candle in enumerate(self.candles)
        }
        self.state_machine = TradeStateMachine()
        self._trade_counter = 0
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

    def _next_trade_id(self) -> str:
        self._trade_counter += 1
        return f"paper-{self._trade_counter:06d}"

    def _normalize_text(self, value: str) -> str:
        return str(value or "").strip()

    def _validate_payload(self, payload: SignalPayload) -> RejectionReason | None:
        if self.config.mode == ExecutionMode.LIVE_BROKER:
            return RejectionReason(
                code="live_broker_disabled",
                message="LIVE_BROKER mode is disabled",
            )
        if payload.symbol not in self.config.symbol_whitelist:
            return RejectionReason(
                code="symbol_not_allowed",
                message="Symbol not whitelisted",
                details={"symbol": payload.symbol},
            )
        if payload.ruleset_id not in self.config.ruleset_whitelist:
            return RejectionReason(
                code="ruleset_not_allowed",
                message="Ruleset not whitelisted",
                details={"ruleset_id": payload.ruleset_id},
            )
        if self.config.timeframe_whitelist and payload.timeframe not in self.config.timeframe_whitelist:
            return RejectionReason(
                code="timeframe_not_allowed",
                message="Timeframe not whitelisted",
                details={"timeframe": payload.timeframe},
            )
        entry_type = self._normalize_text(payload.entry_type)
        if self.config.entry_type_required.lower() not in entry_type.lower():
            return RejectionReason(
                code="entry_type_gate",
                message="Entry type gate failed",
                details={"entry_type": payload.entry_type},
            )
        if self._normalize_text(payload.phase).lower() != self.config.phase_required.lower():
            return RejectionReason(
                code="phase_gate",
                message="Phase gate failed",
                details={"phase": payload.phase},
            )
        if self.config.require_tradable_ob and not payload.ob_tradable:
            return RejectionReason(
                code="ob_tradability_gate",
                message="Order block not tradable",
                details={"ob_tradable": payload.ob_tradable},
            )
        return None

    def _risk_snapshot(self) -> dict[str, float | int]:
        return {
            "trades_today_count": self.ledger.trades_today_count,
            "consecutive_losses": self.ledger.consecutive_losses,
            "daily_drawdown_pct": self.ledger.daily_drawdown_pct,
            "overall_drawdown_pct": self.ledger.overall_drawdown_pct,
        }

    def _log_rejection(self, payload: SignalPayload, reason: RejectionReason) -> None:
        rejection_path = self.config.log_dir / "rejections.log"
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "payload": payload.__dict__,
            "reason": {
                "code": reason.code,
                "message": reason.message,
                "details": reason.details,
            },
        }
        with rejection_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_entry) + "\n")

    def _compute_sl_tp(self, entry_price: float, direction: str) -> tuple[float, float]:
        stop_distance = entry_price * self.config.sl_pct
        if direction == "bullish":
            sl_price = entry_price - stop_distance
            tp_price = entry_price + (stop_distance * self.config.rr_default)
        else:
            sl_price = entry_price + stop_distance
            tp_price = entry_price - (stop_distance * self.config.rr_default)
        return sl_price, tp_price

    def _compute_size(self, entry_price: float, sl_price: float) -> float:
        stop_distance = abs(entry_price - sl_price)
        if stop_distance <= 0:
            return 0.0
        return (self.ledger.equity_current * self.config.risk_per_trade) / stop_distance

    def _simulate_exit(self, trade: PaperTrade) -> PaperTrade:
        entry_idx = self.candle_index.get(trade.entry_time)
        if entry_idx is None:
            return trade
        exit_time = None
        exit_price = None
        status = trade.status
        start_idx = min(entry_idx + 1, len(self.candles))
        for candle in self.candles[start_idx:]:
            if trade.direction == "bullish":
                hit_sl = candle.low <= trade.sl_price
                hit_tp = candle.high >= trade.tp_price
                if hit_sl and hit_tp:
                    exit_time = candle.timestamp
                    exit_price = trade.sl_price
                    status = self.state_machine.transition(trade.status, TradeStatus.CLOSED_SL)
                    break
                if hit_sl:
                    exit_time = candle.timestamp
                    exit_price = trade.sl_price
                    status = self.state_machine.transition(trade.status, TradeStatus.CLOSED_SL)
                    break
                if hit_tp:
                    exit_time = candle.timestamp
                    exit_price = trade.tp_price
                    status = self.state_machine.transition(trade.status, TradeStatus.CLOSED_TP)
                    break
            else:
                hit_sl = candle.high >= trade.sl_price
                hit_tp = candle.low <= trade.tp_price
                if hit_sl and hit_tp:
                    exit_time = candle.timestamp
                    exit_price = trade.sl_price
                    status = self.state_machine.transition(trade.status, TradeStatus.CLOSED_SL)
                    break
                if hit_sl:
                    exit_time = candle.timestamp
                    exit_price = trade.sl_price
                    status = self.state_machine.transition(trade.status, TradeStatus.CLOSED_SL)
                    break
                if hit_tp:
                    exit_time = candle.timestamp
                    exit_price = trade.tp_price
                    status = self.state_machine.transition(trade.status, TradeStatus.CLOSED_TP)
                    break
        if exit_time is None and self.candles:
            last_candle = self.candles[-1]
            exit_time = last_candle.timestamp
            exit_price = last_candle.close
            status = self.state_machine.transition(trade.status, TradeStatus.CLOSED_TIME)
        if exit_time is None or exit_price is None:
            return trade
        stop_distance = abs(trade.entry_price - trade.sl_price)
        pnl_r = None
        if stop_distance > 0:
            if trade.direction == "bullish":
                pnl_r = (exit_price - trade.entry_price) / stop_distance
            else:
                pnl_r = (trade.entry_price - exit_price) / stop_distance
        pnl_cash = pnl_r * self.ledger.equity_current * self.config.risk_per_trade if pnl_r is not None else None
        return PaperTrade(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            timeframe=trade.timeframe,
            ruleset_id=trade.ruleset_id,
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            direction=trade.direction,
            sl_price=trade.sl_price,
            tp_price=trade.tp_price,
            size=trade.size,
            status=status,
            exit_time=exit_time,
            exit_price=exit_price,
            pnl_r=pnl_r,
            pnl_cash=pnl_cash,
        )

    def accept_signal(self, payload: SignalPayload) -> TradeDecision:
        rejection = self._validate_payload(payload)
        if rejection:
            decision = TradeDecision(status=TradeStatus.REJECTED, reason=rejection)
            self._log_rejection(payload, rejection)
            return decision

        can_take, reason = can_take_trade(self._risk_snapshot(), self.risk_config)
        if not can_take and reason:
            decision = TradeDecision(status=TradeStatus.REJECTED, reason=reason)
            self._log_rejection(payload, reason)
            return decision

        if self.config.mode == ExecutionMode.LOG_ONLY:
            return TradeDecision(status=TradeStatus.OPEN)

        sl_price, tp_price = self._compute_sl_tp(payload.entry_price, payload.direction)
        size = self._compute_size(payload.entry_price, sl_price)
        trade = PaperTrade(
            trade_id=self._next_trade_id(),
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            ruleset_id=payload.ruleset_id,
            entry_time=payload.entry_time,
            entry_price=payload.entry_price,
            direction=payload.direction,
            sl_price=sl_price,
            tp_price=tp_price,
            size=size,
            status=TradeStatus.NEW,
        )
        trade.status = self.state_machine.transition(trade.status, TradeStatus.OPEN)

        if self.candles:
            trade = self._simulate_exit(trade)
            if trade.status in {
                TradeStatus.CLOSED_TP,
                TradeStatus.CLOSED_SL,
                TradeStatus.CLOSED_TIME,
            }:
                self.ledger.record_trade(trade)
        return TradeDecision(status=trade.status, trade=trade)

    def finalize(self) -> None:
        self.ledger.finalize()
