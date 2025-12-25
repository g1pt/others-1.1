"""Reporting helpers for research runs."""
from __future__ import annotations

from pathlib import Path

from src.backtest import BacktestResult


def write_trades_csv(result: BacktestResult, path: str | Path) -> None:
    """Write trades to CSV."""
    lines = ["entry_time,entry_price,direction,exit_time,exit_price,pnl_r"]
    for trade in result.trades:
        lines.append(
            f"{trade.entry_time},{trade.entry_price},{trade.direction},"
            f"{trade.exit_time or ''},{trade.exit_price or ''},{trade.pnl_r or ''}"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")
