"""Paper execution engine for deterministic simulation."""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.data import load_candles_csv
from src.models import Candle

from .config import ExecutionMode, PaperEngineConfig
from .ledger import Ledger
from .models import PaperTrade, SignalEvent, TradeStatus
from .risk import RiskLimits, can_open_trade, compute_qty, compute_sl
from .tp_routing import compute_tp


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _normalize_direction(direction: str) -> str | None:
    mapping = {
        "buy": "buy",
        "long": "buy",
        "bullish": "buy",
        "sell": "sell",
        "short": "sell",
        "bearish": "sell",
    }
    return mapping.get(direction.lower()) if direction else None


def _default_loader(symbol: str, timeframe: str) -> list[Candle]:
    data_dir = Path("data")
    candidates = [
        data_dir / f"{symbol}_{timeframe}.csv",
        data_dir / f"{symbol}{timeframe}.csv",
        data_dir / f"{symbol}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_candles_csv(candidate)
    return []


class PaperEngine:
    """Deterministic paper execution engine for webhook signals."""

    def __init__(
        self,
        config: PaperEngineConfig,
        ledger: Ledger,
        data_loader: Callable[[str, str], Iterable[Candle]] | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.data_loader = data_loader or _default_loader

    def _normalize_symbol(self, symbol: str) -> str:
        return self.config.symbol_map.get(symbol, symbol)

    def _parse_event(self, payload: Mapping[str, Any]) -> tuple[SignalEvent | None, str | None]:
        try:
            symbol = self._normalize_symbol(str(payload.get("symbol") or ""))
            timeframe = str(payload.get("timeframe") or "")
            setup = str(payload.get("setup") or "")
            entry_type = str(payload.get("entry_type") or "")
            phase = str(payload.get("phase") or "")
            ob_tradability = str(payload.get("ob_tradability") or "")
            direction = _normalize_direction(str(payload.get("direction") or ""))
            entry_price = float(payload.get("price"))
            entry_time = str(payload.get("timestamp") or payload.get("received_utc") or "")
        except (TypeError, ValueError):
            return None, "invalid_payload"

        if not all([symbol, timeframe, setup, entry_type, phase, ob_tradability, direction, entry_time]):
            return None, "invalid_payload"

        return (
            SignalEvent(
                symbol=symbol,
                timeframe=timeframe,
                setup=setup,
                direction=direction,
                entry_time=entry_time,
                entry_price=entry_price,
                entry_type=entry_type,
                phase=phase,
                ob_tradability=ob_tradability,
            ),
            None,
        )

    def _log_rejection(self, payload: Mapping[str, Any], reason: str) -> None:
        self.ledger.record_rejection(dict(payload), reason)
        self.ledger.log_event(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "accepted": False,
                "reason": reason,
                "event": dict(payload),
            }
        )

    def _log_accept(self, payload: Mapping[str, Any], trade: PaperTrade | None = None) -> None:
        event: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "accepted": True,
            "event": dict(payload),
        }
        if trade:
            # Attach tp_level/tp_label here if paper_trades.log needs TP metadata later.
            event.update(
                {
                    "trade_id": trade.trade_id,
                    "tp_level": trade.tp_price,
                    "tp_label": trade.tp_label,
                    "sl_level": trade.sl_price,
                    "qty": trade.size,
                    "risk_cash": trade.risk_cash,
                    "status": trade.status.value,
                    "close_reason": trade.close_reason,
                }
            )
        self.ledger.log_event(event)

    def _simulate_exit(self, trade: PaperTrade, candles: list[Candle]) -> PaperTrade:
        entry_time = _parse_datetime(trade.entry_time)
        if entry_time is None:
            trade.status = TradeStatus.INVALID
            trade.close_reason = "INVALID"
            return trade

        exit_time = None
        exit_price = None
        close_reason = None

        for candle in candles:
            candle_time = _parse_datetime(candle.timestamp)
            if candle_time is None or candle_time <= entry_time:
                continue
            if trade.direction == "buy":
                sl_hit = candle.low <= trade.sl_price
                tp_hit = candle.high >= trade.tp_price
            else:
                sl_hit = candle.high >= trade.sl_price
                tp_hit = candle.low <= trade.tp_price

            if sl_hit and tp_hit:
                exit_time = candle.timestamp
                exit_price = trade.sl_price
                close_reason = "SL"
                break
            if sl_hit:
                exit_time = candle.timestamp
                exit_price = trade.sl_price
                close_reason = "SL"
                break
            if tp_hit:
                exit_time = candle.timestamp
                exit_price = trade.tp_price
                close_reason = "TP"
                break

        if exit_time is None and candles:
            last_candle = candles[-1]
            exit_time = last_candle.timestamp
            exit_price = last_candle.close
            close_reason = "TIME"

        if exit_time is None or exit_price is None:
            trade.status = TradeStatus.INVALID
            trade.close_reason = "INVALID"
            return trade

        stop_distance = abs(trade.entry_price - trade.sl_price)
        pnl_r = None
        if stop_distance > 0:
            if trade.direction == "buy":
                pnl_r = (exit_price - trade.entry_price) / stop_distance
            else:
                pnl_r = (trade.entry_price - exit_price) / stop_distance
        trade.status = TradeStatus.CLOSED
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.close_reason = close_reason
        trade.pnl_r = pnl_r
        if pnl_r is not None and trade.risk_cash is not None:
            trade.pnl_cash = trade.risk_cash * pnl_r
        return trade

    def process_signal(self, payload: Mapping[str, Any]) -> tuple[bool, str, str | None]:
        event, error = self._parse_event(payload)
        if error:
            self._log_rejection(payload, error)
            return False, error, None

        if self.config.mode == ExecutionMode.LIVE_BROKER:
            self._log_rejection(payload, "live_broker_disabled")
            return False, "live_broker_disabled", None

        ruleset = self.config.rulesets.get(event.symbol)
        if not ruleset or not ruleset.enabled:
            self._log_rejection(payload, "symbol_not_allowed")
            return False, "symbol_not_allowed", None
        if event.timeframe != ruleset.timeframe:
            self._log_rejection(payload, "timeframe_not_allowed")
            return False, "timeframe_not_allowed", None
        if event.setup != ruleset.setup_id:
            self._log_rejection(payload, "setup_not_allowed")
            return False, "setup_not_allowed", None
        if event.entry_type != ruleset.entry_type:
            self._log_rejection(payload, "gate_failed:entry_type")
            return False, "gate_failed:entry_type", None
        if event.phase != ruleset.phase:
            self._log_rejection(payload, "gate_failed:phase")
            return False, "gate_failed:phase", None
        if event.ob_tradability != ruleset.ob_tradability:
            self._log_rejection(payload, "gate_failed:ob_tradability")
            return False, "gate_failed:ob_tradability", None

        if self.config.mode == ExecutionMode.LOG_ONLY:
            self._log_accept(payload)
            return True, "ok", None

        can_open, reason = can_open_trade(self.ledger.state(), event.entry_time, self.config.risk_limits)
        if not can_open:
            self._log_rejection(payload, reason or "risk_blocked")
            return False, reason or "risk_blocked", None

        sl = compute_sl(event.entry_price, event.direction, self.config.sl_pct)
        tp_level, tp_label = compute_tp(
            event.entry_price,
            sl,
            None,
            event.direction,
            event.symbol,
            event.setup,
        )
        qty = compute_qty(
            self.ledger.current_equity,
            self.config.risk_limits.risk_per_trade_pct,
            event.entry_price,
            sl,
        )
        if qty <= 0:
            self._log_rejection(payload, "invalid_payload")
            return False, "invalid_payload", None

        risk_cash = self.ledger.current_equity * self.config.risk_limits.risk_per_trade_pct
        trade = PaperTrade(
            trade_id=str(uuid.uuid4()),
            symbol=event.symbol,
            timeframe=event.timeframe,
            ruleset_id=event.setup,
            setup=event.setup,
            entry_time=event.entry_time,
            entry_price=event.entry_price,
            direction=event.direction,
            sl_price=sl,
            tp_price=tp_level,
            size=qty,
            status=TradeStatus.OPEN,
            tp_label=tp_label,
            created_utc=datetime.utcnow().isoformat(),
            risk_cash=risk_cash,
        )

        candles = list(self.data_loader(event.symbol, event.timeframe))
        if not candles:
            self._log_rejection(payload, "no_candles")
            return False, "no_candles", None

        trade = self._simulate_exit(trade, candles)
        if trade.status == TradeStatus.INVALID:
            self._log_rejection(payload, "invalid_payload")
            return False, "invalid_payload", None

        self.ledger.apply_trade_close(trade)
        self.ledger.record_trade(trade)
        self._log_accept(payload, trade)
        return True, "ok", trade.trade_id


__all__ = ["PaperEngine"]
