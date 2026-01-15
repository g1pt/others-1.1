"""Equity ledger for paper execution."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from .models import PaperTrade


def _parse_date(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return datetime.min.date()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class EquityLedger:
    equity_current: float
    equity_start_day: float
    realized_pnl_today: float = 0.0
    daily_drawdown_pct: float = 0.0
    trades_today_count: int = 0
    consecutive_losses: int = 0
    equity_high_watermark: float = field(init=False)
    overall_drawdown_pct: float = 0.0
    current_day: date | None = None
    equity_log_path: Path = Path("logs/equity_by_day.csv")
    trades_log_path: Path = Path("logs/paper_trades.csv")

    def __post_init__(self) -> None:
        self.equity_high_watermark = self.equity_current

    def _update_drawdowns(self) -> None:
        if self.equity_start_day:
            self.daily_drawdown_pct = (
                (self.equity_start_day - self.equity_current) / self.equity_start_day
            ) * 100
        else:
            self.daily_drawdown_pct = 0.0
        self.equity_high_watermark = max(self.equity_high_watermark, self.equity_current)
        if self.equity_high_watermark:
            self.overall_drawdown_pct = (
                (self.equity_high_watermark - self.equity_current)
                / self.equity_high_watermark
            ) * 100
        else:
            self.overall_drawdown_pct = 0.0

    def rollover_day(self, trade_date: date) -> None:
        if self.current_day is None:
            self.current_day = trade_date
            self.equity_start_day = self.equity_current
            return
        if trade_date == self.current_day:
            return
        self._write_daily_snapshot(self.current_day)
        self.current_day = trade_date
        self.equity_start_day = self.equity_current
        self.realized_pnl_today = 0.0
        self.daily_drawdown_pct = 0.0
        self.trades_today_count = 0

    def record_trade(self, trade: PaperTrade) -> None:
        trade_date = _parse_date(trade.entry_time)
        self.rollover_day(trade_date)
        self.trades_today_count += 1
        if trade.pnl_cash is not None:
            self.realized_pnl_today += trade.pnl_cash
            self.equity_current += trade.pnl_cash
        if trade.pnl_cash is not None and trade.pnl_cash < 0:
            self.consecutive_losses += 1
        elif trade.pnl_cash is not None:
            self.consecutive_losses = 0
        self._update_drawdowns()
        self._append_trade(trade)

    def _append_trade(self, trade: PaperTrade) -> None:
        _ensure_parent(self.trades_log_path)
        write_header = not self.trades_log_path.exists() or self.trades_log_path.stat().st_size == 0
        with self.trades_log_path.open("a", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "trade_id",
                    "symbol",
                    "timeframe",
                    "ruleset_id",
                    "entry_time",
                    "entry_price",
                    "direction",
                    "sl_price",
                    "tp_price",
                    "size",
                    "status",
                    "exit_time",
                    "exit_price",
                    "pnl_r",
                    "pnl_cash",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "timeframe": trade.timeframe,
                    "ruleset_id": trade.ruleset_id,
                    "entry_time": trade.entry_time,
                    "entry_price": trade.entry_price,
                    "direction": trade.direction,
                    "sl_price": trade.sl_price,
                    "tp_price": trade.tp_price,
                    "size": trade.size,
                    "status": trade.status.value,
                    "exit_time": trade.exit_time,
                    "exit_price": trade.exit_price,
                    "pnl_r": trade.pnl_r,
                    "pnl_cash": trade.pnl_cash,
                }
            )

    def _write_daily_snapshot(self, snapshot_day: date) -> None:
        _ensure_parent(self.equity_log_path)
        write_header = not self.equity_log_path.exists() or self.equity_log_path.stat().st_size == 0
        with self.equity_log_path.open("a", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "date",
                    "equity_start_day",
                    "equity_end_day",
                    "realized_pnl_today",
                    "daily_drawdown_pct",
                    "trades_today_count",
                    "consecutive_losses",
                    "equity_high_watermark",
                    "overall_drawdown_pct",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "date": snapshot_day.isoformat(),
                    "equity_start_day": self.equity_start_day,
                    "equity_end_day": self.equity_current,
                    "realized_pnl_today": self.realized_pnl_today,
                    "daily_drawdown_pct": self.daily_drawdown_pct,
                    "trades_today_count": self.trades_today_count,
                    "consecutive_losses": self.consecutive_losses,
                    "equity_high_watermark": self.equity_high_watermark,
                    "overall_drawdown_pct": self.overall_drawdown_pct,
                }
            )

    def finalize(self) -> None:
        if self.current_day is None:
            return
        self._write_daily_snapshot(self.current_day)


@dataclass
class Ledger:
    log_dir: Path
    current_equity: float
    high_watermark: float
    daily_start_equity: float
    daily_realized_pnl: float = 0.0
    daily_max_drawdown_pct: float = 0.0
    trades_today_count: int = 0
    blocked_count: int = 0
    consecutive_losses: int = 0
    overall_drawdown_pct: float = 0.0
    current_day: date | None = None
    trades_log_path: Path = field(init=False)
    equity_log_path: Path = field(init=False)
    rejections_log_path: Path = field(init=False)
    events_log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.trades_log_path = self.log_dir / "trades.csv"
        self.equity_log_path = self.log_dir / "equity_by_day.csv"
        self.rejections_log_path = self.log_dir / "rejections.log"
        self.events_log_path = self.log_dir / "paper_trades.log"

    @classmethod
    def load_or_init(cls, log_dir: Path, initial_equity: float = 10000.0) -> "Ledger":
        log_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            log_dir=log_dir,
            current_equity=initial_equity,
            high_watermark=initial_equity,
            daily_start_equity=initial_equity,
        )

    def _rollover_day(self, trade_date: date) -> None:
        if self.current_day is None:
            self.current_day = trade_date
            self.daily_start_equity = self.current_equity
            return
        if trade_date == self.current_day:
            return
        self._write_daily_snapshot(self.current_day)
        self.current_day = trade_date
        self.daily_start_equity = self.current_equity
        self.daily_realized_pnl = 0.0
        self.daily_max_drawdown_pct = 0.0
        self.trades_today_count = 0
        self.blocked_count = 0

    def _update_drawdowns(self) -> None:
        if self.daily_start_equity:
            daily_drawdown = (self.daily_start_equity - self.current_equity) / self.daily_start_equity
        else:
            daily_drawdown = 0.0
        self.daily_max_drawdown_pct = max(self.daily_max_drawdown_pct, daily_drawdown)
        if self.high_watermark:
            self.overall_drawdown_pct = (self.high_watermark - self.current_equity) / self.high_watermark
        else:
            self.overall_drawdown_pct = 0.0

    def state(self) -> dict[str, float | int]:
        return {
            "current_equity": self.current_equity,
            "trades_today_count": self.trades_today_count,
            "consecutive_losses": self.consecutive_losses,
            "daily_drawdown_pct": self.daily_max_drawdown_pct,
            "overall_drawdown_pct": self.overall_drawdown_pct,
        }

    def log_event(self, event: dict[str, object]) -> None:
        _ensure_parent(self.events_log_path)
        with self.events_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def record_rejection(self, event: dict[str, object], reason: str, details: dict[str, object] | None = None) -> None:
        self.blocked_count += 1
        _ensure_parent(self.rejections_log_path)
        rejection = {
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
            "details": details or {},
            "event": event,
        }
        with self.rejections_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rejection) + "\n")

    def apply_trade_close(self, trade: PaperTrade) -> None:
        trade_date = _parse_date(trade.exit_time or trade.entry_time)
        self._rollover_day(trade_date)
        self.trades_today_count += 1
        if trade.pnl_cash is not None:
            self.daily_realized_pnl += trade.pnl_cash
            self.current_equity += trade.pnl_cash
        if self.current_equity > self.high_watermark:
            self.high_watermark = self.current_equity
        if trade.pnl_r is not None:
            if trade.pnl_r < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
        self._update_drawdowns()

    def record_trade(self, trade: PaperTrade) -> None:
        _ensure_parent(self.trades_log_path)
        write_header = not self.trades_log_path.exists() or self.trades_log_path.stat().st_size == 0
        with self.trades_log_path.open("a", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "trade_id",
                    "symbol",
                    "timeframe",
                    "setup",
                    "direction",
                    "entry_time",
                    "entry_price",
                    "sl_price",
                    "tp_price",
                    "size",
                    "status",
                    "exit_time",
                    "exit_price",
                    "close_reason",
                    "pnl_r",
                    "pnl_cash",
                    "tp_label",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "timeframe": trade.timeframe,
                    "setup": trade.setup,
                    "direction": trade.direction,
                    "entry_time": trade.entry_time,
                    "entry_price": trade.entry_price,
                    "sl_price": trade.sl_price,
                    "tp_price": trade.tp_price,
                    "size": trade.size,
                    "status": trade.status.value,
                    "exit_time": trade.exit_time,
                    "exit_price": trade.exit_price,
                    "close_reason": trade.close_reason,
                    "pnl_r": trade.pnl_r,
                    "pnl_cash": trade.pnl_cash,
                    "tp_label": trade.tp_label,
                }
            )

    def _write_daily_snapshot(self, snapshot_day: date) -> None:
        _ensure_parent(self.equity_log_path)
        write_header = not self.equity_log_path.exists() or self.equity_log_path.stat().st_size == 0
        with self.equity_log_path.open("a", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "date",
                    "start_equity",
                    "end_equity",
                    "realized_pnl",
                    "max_dd_pct",
                    "trades_count",
                    "blocked_count",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "date": snapshot_day.isoformat(),
                    "start_equity": self.daily_start_equity,
                    "end_equity": self.current_equity,
                    "realized_pnl": self.daily_realized_pnl,
                    "max_dd_pct": self.daily_max_drawdown_pct,
                    "trades_count": self.trades_today_count,
                    "blocked_count": self.blocked_count,
                }
            )

    def finalize(self) -> None:
        if self.current_day is None:
            return
        self._write_daily_snapshot(self.current_day)
