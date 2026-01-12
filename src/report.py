"""Reporting helpers for research runs."""
from __future__ import annotations

from pathlib import Path

from src.analysis import SummaryRow
from src.backtest import BacktestResult
from src.risk import EquitySimulation


def write_trades_csv(result: BacktestResult, path: str | Path) -> None:
    """Write trades to CSV."""
    lines = [
        "entry_time,entry_price,stop_price,direction,mmxm_phase,entry_method,ob_tradable,ob_id,"
        "exit_time,exit_price,pnl_r,day_label"
    ]
    for trade in result.trades:
        lines.append(
            f"{trade.entry_time},{trade.entry_price},{trade.stop_price or ''},{trade.direction},"
            f"{trade.mmxm_phase},{trade.entry_method},{trade.ob_tradable},{trade.ob_id},"
            f"{trade.exit_time or ''},{trade.exit_price or ''},{trade.pnl_r or ''},"
            f"{trade.day_label or ''}"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_summary_csv(rows: list[SummaryRow], path: str | Path) -> None:
    """Write summary table to CSV."""
    lines = ["key,trades,winrate,expectancy,max_drawdown,stability"]
    for row in rows:
        lines.append(
            f"{row.key},{row.trades},{row.winrate:.4f},{row.expectancy:.4f},"
            f"{row.max_drawdown:.4f},{row.stability:.4f}"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_equity_log_csv(simulation: EquitySimulation, path: str | Path) -> None:
    """Write equity/risk logs to CSV."""
    lines = [
        "entry_time,entry_price,stop_price,direction,mmxm_phase,entry_method,ob_tradable,ob_id,"
        "exit_time,exit_price,pnl_r,equity_before,equity_after,risk_pct,risk_amount,"
        "drawdown_pct,new_equity_high"
    ]
    for log in simulation.trade_logs:
        trade = log.trade
        lines.append(
            f"{trade.entry_time},{trade.entry_price},{trade.stop_price or ''},{trade.direction},"
            f"{trade.mmxm_phase},{trade.entry_method},{trade.ob_tradable},{trade.ob_id},"
            f"{trade.exit_time or ''},{trade.exit_price or ''},{trade.pnl_r or ''},"
            f"{log.equity_before:.2f},{log.equity_after:.2f},{log.risk_pct:.4f},"
            f"{log.risk_amount:.2f},{log.drawdown_pct:.4f},{log.new_equity_high}"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")
