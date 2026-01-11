"""Risk management and equity tracking."""
from __future__ import annotations

from dataclasses import dataclass

from src.models import Trade


@dataclass(frozen=True)
class EquityPoint:
    timestamp: str
    equity: float


@dataclass(frozen=True)
class DailyEquity:
    date: str
    equity: float


@dataclass(frozen=True)
class TradeResult:
    trade: Trade
    pnl: float
    pnl_r: float | None
    equity_after: float


@dataclass(frozen=True)
class SkippedTrade:
    trade: Trade
    reason: str


@dataclass(frozen=True)
class EquitySimulation:
    initial_equity: float
    final_equity: float
    return_pct: float
    max_drawdown_pct: float
    max_drawdown_currency: float
    equity_per_trade: list[float]
    daily_equity: list[DailyEquity]
    trade_results: list[TradeResult]
    skipped_trades: list[SkippedTrade]


def update_equity(current_equity: float, pnl_r: float, risk_per_trade: float) -> float:
    """Update equity based on a trade result in R."""
    return current_equity + (pnl_r * risk_per_trade)


def simulate_equity(
    trades: list[Trade],
    *,
    initial_equity: float = 10_000.0,
    risk_per_trade: float = 0.01,
) -> EquitySimulation:
    """Simulate an equity curve based on trade outcomes in currency."""
    equity = initial_equity
    equity_curve: list[float] = []
    trade_results: list[TradeResult] = []
    skipped_trades: list[SkippedTrade] = []
    daily_equity: list[DailyEquity] = []
    current_date: str | None = None
    latest_equity_for_date: float | None = None

    peak_equity = initial_equity
    max_drawdown = 0.0

    for trade in trades:
        stop_distance = _stop_distance(trade)
        if stop_distance <= 0:
            skipped_trades.append(SkippedTrade(trade=trade, reason="bad_stop"))
            continue
        if trade.exit_price is None:
            skipped_trades.append(SkippedTrade(trade=trade, reason="missing_exit"))
            continue

        direction = 1 if trade.direction == "bullish" else -1
        risk_amount = equity * risk_per_trade
        size = risk_amount / stop_distance
        pnl = size * (trade.exit_price - trade.entry_price) * direction
        equity += pnl

        equity_curve.append(equity)
        trade_results.append(
            TradeResult(
                trade=trade,
                pnl=pnl,
                pnl_r=trade.pnl_r,
                equity_after=equity,
            )
        )

        trade_date = _extract_date(trade.entry_time)
        if current_date is None:
            current_date = trade_date
        if trade_date != current_date and latest_equity_for_date is not None:
            daily_equity.append(
                DailyEquity(date=current_date, equity=latest_equity_for_date)
            )
            current_date = trade_date
        latest_equity_for_date = equity

        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, peak_equity - equity)

    if current_date is not None and latest_equity_for_date is not None:
        daily_equity.append(DailyEquity(date=current_date, equity=latest_equity_for_date))

    return_pct = (
        (equity - initial_equity) / initial_equity * 100
        if initial_equity
        else 0.0
    )
    max_drawdown_pct = (
        (max_drawdown / peak_equity) * 100 if peak_equity else 0.0
    )

    return EquitySimulation(
        initial_equity=initial_equity,
        final_equity=equity,
        return_pct=return_pct,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_currency=max_drawdown,
        equity_per_trade=equity_curve,
        daily_equity=daily_equity,
        trade_results=trade_results,
        skipped_trades=skipped_trades,
    )


def _stop_distance(trade: Trade) -> float:
    if trade.stop_price is None:
        return 0.0
    if trade.direction == "bullish":
        return trade.entry_price - trade.stop_price
    if trade.direction == "bearish":
        return trade.stop_price - trade.entry_price
    return 0.0


def _extract_date(timestamp: str) -> str:
    if "T" in timestamp:
        return timestamp.split("T", maxsplit=1)[0]
    return timestamp.split(" ", maxsplit=1)[0]
