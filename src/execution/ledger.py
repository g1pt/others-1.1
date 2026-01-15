"""Equity ledger for paper execution."""
from __future__ import annotations

import csv
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
