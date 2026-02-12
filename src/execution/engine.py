"""Paper execution engine."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable

from .config import default_symbol_map
from .logging_utils import log_line, write_state
from .models import CloseReason, DailyLedger, TradeSignal
from .risk import RiskLimits, calc_position_size, can_open_trade, daily_key
from .simulator import determine_exit
from .tp_sl import compute_sl as compute_sl_price
from .tp_sl import compute_tp as compute_tp_price

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


def _normalize_timeframe(value: str) -> str:
    cleaned = str(value or "").strip().lower().replace("m", "")
    return cleaned


def _normalize_direction(value: str) -> str:
    mapping = {
        "buy": "buy",
        "long": "buy",
        "bullish": "buy",
        "sell": "sell",
        "short": "sell",
        "bearish": "sell",
    }
    return mapping.get(value.strip().lower(), value.strip().lower())


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _iter_candles(candles: Iterable) -> list:
    if hasattr(candles, "iterrows"):
        rows = []
        for _, row in candles.iterrows():
            rows.append(
                {
                    "timestamp": str(row["timestamp"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
        return rows
    return list(candles)


def _candle_timestamp(candle) -> str | None:
    if isinstance(candle, dict):
        return str(candle.get("timestamp"))
    return str(getattr(candle, "timestamp", None))


def _candles_after_entry(
    entry_time: str,
    candles: list,
    candle_index: dict[str, int],
) -> list:
    entry_idx = candle_index.get(entry_time)
    if entry_idx is not None:
        return candles[min(entry_idx + 1, len(candles)) :]
    entry_dt = _parse_datetime(entry_time)
    if entry_dt is None:
        return candles
    result = []
    for candle in candles:
        candle_dt = _parse_datetime(_candle_timestamp(candle) or "")
        if candle_dt is None or candle_dt <= entry_dt:
            continue
        result.append(candle)
    return result


def _compute_pnl_r(entry: float, sl: float, exit_price: float, direction: str) -> float | None:
    stop_distance = abs(entry - sl)
    if stop_distance <= 0:
        return None
    if direction == "buy":
        return (exit_price - entry) / stop_distance
    return (entry - exit_price) / stop_distance


def run_paper_execute(
    signals: Iterable[TradeSignal],
    candles_df: Iterable,
    config: dict,
) -> dict:
    """Run paper execution for MMXM_4C_D signals on 30m data."""
    defaults = {
        "initial_equity": 10000.0,
        "risk_per_trade_pct": 0.005,
        "risk_mode": "fixed_per_trade",
        "daily_risk_budget_pct": 0.02,
        "min_risk_per_trade_pct": 0.003,
        "max_risk_per_trade_pct": 0.01,
        "max_trades_per_day": 1,
        "stop_after_consecutive_losses": 2,
        "daily_drawdown_stop_pct": 0.02,
        "hard_max_drawdown_pct": 0.03,
        "rr": 2.0,
        "st_pct": 0.002,
        "timeframe": "30",
        "setup_id": "MMXM_4C_D",
        "entry_type": "Refinement",
        "phase": "Manipulation",
        "ob_tradability": "Tradable",
        "tie_breaker": "SL",
        "log_dir": Path("logs"),
        "symbol_map": default_symbol_map(),
    }
    settings = {**defaults, **(config or {})}
    log_dir: Path = settings["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    trades_log_path = log_dir / "paper_exec_trades.jsonl"
    equity_log_path = log_dir / "equity_by_day.csv"
    rejection_log_path = log_dir / "rejections.jsonl"
    stream_log_path = log_dir / "paper_trades.log"
    state_path = log_dir / "paper_state.json"

    risk_limits = RiskLimits(
        max_trades_per_day=settings["max_trades_per_day"],
        stop_after_consecutive_losses=settings["stop_after_consecutive_losses"],
        daily_drawdown_stop_pct=settings["daily_drawdown_stop_pct"],
        hard_max_drawdown_pct=settings["hard_max_drawdown_pct"],
    )

    candles = _iter_candles(candles_df)
    candle_index = {
        _candle_timestamp(candle): idx for idx, candle in enumerate(candles) if _candle_timestamp(candle)
    }

    equity_current = float(settings["initial_equity"])
    equity_high = equity_current
    max_drawdown_pct = 0.0
    loss_streak = 0
    loss_streak_max = 0
    trades_received = 0
    trades_rejected = 0
    trades_opened = 0
    trades_closed = 0
    rejection_counts: dict[str, int] = {
        "max_trades_per_day": 0,
        "loss_streak_stop": 0,
        "daily_drawdown_stop": 0,
        "hard_drawdown_stop": 0,
        "invalid_payload": 0,
        "gate_fail": 0,
    }
    trades: list[PaperTrade] = []
    daily_ledgers: list[DailyLedger] = []

    current_day: str | None = None
    day_start_equity = equity_current
    day_realized_pnl = 0.0
    day_drawdown_pct = 0.0
    trades_today = 0

    def _write_state(last_event_time: str) -> None:
        drawdown_pct = ((equity_high - equity_current) / equity_high) if equity_high else 0.0
        write_state(
            {
                "equity": equity_current,
                "drawdown_pct": drawdown_pct,
                "loss_streak": loss_streak,
                "trades_taken_today": trades_today,
                "rejection_counts": rejection_counts,
                "last_event_time": last_event_time,
            },
            state_path,
        )

    def finalize_day(day_key: str) -> None:
        daily_ledgers.append(
            DailyLedger(
                date=day_key,
                start_equity=day_start_equity,
                end_equity=equity_current,
                daily_low_equity=min(day_start_equity, equity_current),
                daily_realized_pnl=day_realized_pnl,
                daily_drawdown_pct=day_drawdown_pct,
                trades_taken=trades_today,
                consecutive_losses=loss_streak,
            )
        )

    def log_rejection(signal: TradeSignal, reason: str, detail: str | None = None) -> None:
        nonlocal trades_rejected
        trades_rejected += 1
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        payload = {
            "timestamp": signal.entry_time,
            "reason": reason,
            "detail": detail,
            "signal": signal.__dict__,
        }
        with rejection_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        extra = f":{detail}" if detail else ""
        log_line(f"[paper] REJECT reason={reason}{extra} t={signal.entry_time}", stream_log_path)
        _write_state(signal.entry_time)

    def log_trade(trade: PaperTrade) -> None:
        payload = {
            "trade_id": trade.trade_id,
            "state": trade.status.value,
            "external_symbol": trade.external_symbol,
            "internal_symbol": trade.internal_symbol,
            "timeframe": trade.timeframe,
            "setup_id": trade.ruleset_id,
            "entry_time": trade.entry_time,
            "entry_price": trade.entry_price,
            "direction": trade.direction,
            "sl": trade.sl_price,
            "tp": trade.tp_price,
            "exit_time": trade.exit_time,
            "exit_price": trade.exit_price,
            "close_reason": trade.close_reason,
            "pnl_r": trade.pnl_r,
            "pnl_cash": trade.pnl_cash,
            "risk_cash": trade.risk_cash,
        }
        with trades_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def gate_signal(signal: TradeSignal) -> str | None:
        if _normalize_timeframe(signal.timeframe) != _normalize_timeframe(settings["timeframe"]):
            return "session"
        if signal.setup_id != settings["setup_id"]:
            return "setup"
        if settings["entry_type"].lower() not in signal.entry_type.lower():
            return "entry_type"
        if signal.phase.lower() != settings["phase"].lower():
            return "phase"
        if str(signal.ob_tradability).lower() != settings["ob_tradability"].lower():
            return "ob_tradability"
        return None

    sorted_signals = sorted(signals, key=lambda s: _parse_datetime(s.entry_time) or datetime.min)

    for idx, signal in enumerate(sorted_signals, start=1):
        trades_received += 1
        signal_internal = settings["symbol_map"].get(signal.external_symbol, signal.internal_symbol)
        signal = TradeSignal(
            external_symbol=signal.external_symbol,
            internal_symbol=signal_internal,
            timeframe=signal.timeframe,
            setup_id=signal.setup_id,
            entry_time=signal.entry_time,
            entry_price=signal.entry_price,
            direction=signal.direction,
            entry_type=signal.entry_type,
            phase=signal.phase,
            ob_tradability=signal.ob_tradability,
        )
        day_key = daily_key(signal.entry_time)
        if current_day is None:
            current_day = day_key
            day_start_equity = equity_current
        if day_key != current_day:
            finalize_day(current_day)
            current_day = day_key
            day_start_equity = equity_current
            day_realized_pnl = 0.0
            day_drawdown_pct = 0.0
            trades_today = 0

        gate_label = gate_signal(signal)
        if gate_label:
            log_rejection(signal, "gate_fail", gate_label)
            continue

        day_drawdown_pct = ((day_start_equity - equity_current) / day_start_equity) if day_start_equity else 0.0
        overall_drawdown_pct = ((equity_high - equity_current) / equity_high) if equity_high else 0.0
        can_open, reason = can_open_trade(
            {
                "trades_today_count": trades_today,
                "consecutive_losses": loss_streak,
                "daily_drawdown_pct": day_drawdown_pct,
                "overall_drawdown_pct": overall_drawdown_pct,
            },
            risk_limits,
        )
        if not can_open and reason:
            log_rejection(signal, reason)
            continue

        direction = _normalize_direction(signal.direction)
        if direction not in {"buy", "sell"}:
            log_rejection(signal, "invalid_payload")
            continue

        sl_price = compute_sl_price(signal.entry_price, direction, settings["st_pct"])
        signal_rr = getattr(signal, "rr", None)
        rr_value = float(signal_rr) if isinstance(signal_rr, (int, float)) and signal_rr > 0 else settings["rr"]
        tp_price = compute_tp_price(signal.entry_price, sl_price, direction, rr_value)

        if settings["risk_mode"] == "daily_budget":
            daily_budget_cash = day_start_equity * settings["daily_risk_budget_pct"]
            remaining_slots = max(1, risk_limits.max_trades_per_day - trades_today)
            risk_cash = daily_budget_cash / remaining_slots
            risk_pct = risk_cash / equity_current if equity_current else 0.0
            risk_pct = max(settings["min_risk_per_trade_pct"], min(settings["max_risk_per_trade_pct"], risk_pct))
            risk_cash = equity_current * risk_pct
            log_line(
                f"[paper] risk_mode=daily_budget risk_pct_used={risk_pct:.6f} remaining_slots={remaining_slots} daily_budget_cash={daily_budget_cash:.2f}",
                stream_log_path,
            )
        else:
            risk_pct = settings["risk_per_trade_pct"]
            risk_cash = equity_current * risk_pct

        size = calc_position_size(signal.entry_price, sl_price, risk_pct, equity_current)
        if size <= 0:
            log_rejection(signal, "invalid_payload")
            continue

        trade = PaperTrade(
            trade_id=f"paper-{idx:06d}",
            symbol=signal_internal,
            timeframe=signal.timeframe,
            ruleset_id=signal.setup_id,
            entry_time=signal.entry_time,
            entry_price=signal.entry_price,
            direction=direction,
            sl_price=sl_price,
            tp_price=tp_price,
            size=size,
            status=TradeStatus.OPEN,
            risk_cash=risk_cash,
            external_symbol=signal.external_symbol,
            internal_symbol=signal_internal,
        )
        trades_opened += 1
        log_line(
            f"[paper] OPEN id={trade.trade_id} t={trade.entry_time} sym={trade.symbol} dir={trade.direction} entry={trade.entry_price:.5f} sl={trade.sl_price:.5f} tp={trade.tp_price:.5f}",
            stream_log_path,
        )

        candles_after = _candles_after_entry(signal.entry_time, candles, candle_index)
        exit_result = determine_exit(trade, candles_after, tie_breaker=settings["tie_breaker"])
        if exit_result is None:
            trade.status = TradeStatus.REJECTED
            trade.close_reason = CloseReason.INVALID.value
            log_trade(trade)
            log_rejection(signal, "invalid_payload")
            continue

        trade.exit_time = exit_result.exit_time
        trade.exit_price = exit_result.exit_price
        trade.close_reason = exit_result.reason
        trade.status = TradeStatus.CLOSED
        trade.pnl_r = _compute_pnl_r(trade.entry_price, trade.sl_price, trade.exit_price, trade.direction)
        if trade.pnl_r is None:
            trade.close_reason = CloseReason.INVALID.value
        trade.pnl_cash = trade.pnl_r * risk_cash if trade.pnl_r is not None else 0.0

        equity_current += trade.pnl_cash or 0.0
        day_realized_pnl += trade.pnl_cash or 0.0
        trades_today += 1
        day_drawdown_pct = ((day_start_equity - equity_current) / day_start_equity) if day_start_equity else 0.0

        if trade.pnl_cash is not None and trade.pnl_cash < 0:
            loss_streak += 1
        elif trade.pnl_cash is not None:
            loss_streak = 0
        loss_streak_max = max(loss_streak_max, loss_streak)
        equity_high = max(equity_high, equity_current)
        drawdown_pct = ((equity_high - equity_current) / equity_high) if equity_high else 0.0
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

        trades_closed += 1
        trades.append(trade)
        log_trade(trade)
        log_line(
            f"[paper] CLOSE id={trade.trade_id} t={trade.exit_time} reason={trade.close_reason} pnl_r={trade.pnl_r:.4f} pnl_cash={trade.pnl_cash:.2f} equity={equity_current:.2f}",
            stream_log_path,
        )
        _write_state(trade.exit_time or signal.entry_time)

    if current_day is not None:
        finalize_day(current_day)

    with equity_log_path.open("w", encoding="utf-8") as handle:
        handle.write("date,start_equity,end_equity,dd_pct,trades_taken,pnl\n")
        for ledger in daily_ledgers:
            handle.write(
                f"{ledger.date},{ledger.start_equity:.2f},{ledger.end_equity:.2f},"
                f"{ledger.daily_drawdown_pct:.4f},{ledger.trades_taken},"
                f"{ledger.daily_realized_pnl:.2f}\n"
            )

    total_return_pct = (((equity_current - settings["initial_equity"]) / settings["initial_equity"]) * 100) if settings["initial_equity"] else 0.0
    _write_state(current_day or datetime.utcnow().isoformat())

    return {
        "trades_received": trades_received,
        "trades_rejected": trades_rejected,
        "trades_opened": trades_opened,
        "trades_closed": trades_closed,
        "end_equity": equity_current,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "loss_streak_max": loss_streak_max,
        "rejection_reasons": rejection_counts,
    }
