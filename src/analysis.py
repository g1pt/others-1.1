"""Analysis utilities for summarizing research backtests."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable, Iterable

from src.models import Trade


@dataclass(frozen=True)
class SummaryRow:
    key: str
    trades: int
    winrate: float
    expectancy: float
    max_drawdown: float
    stability: float


def _compute_equity(trades: Iterable[Trade]) -> list[float]:
    equity = 0.0
    curve = []
    for trade in trades:
        equity += trade.pnl_r or 0.0
        curve.append(equity)
    return curve


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = peak - value
        max_dd = max(max_dd, drawdown)
    return max_dd


def _stability(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls)
    variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    return 1.0 / (1.0 + sqrt(variance))


def summarize(
    trades: list[Trade], key_fn: Callable[[Trade], str], label: str
) -> list[SummaryRow]:
    grouped: dict[str, list[Trade]] = {}
    for trade in trades:
        key = key_fn(trade)
        grouped.setdefault(key, []).append(trade)

    summaries: list[SummaryRow] = []
    for key, group in grouped.items():
        pnls = [trade.pnl_r or 0.0 for trade in group]
        wins = sum(1 for pnl in pnls if pnl > 0)
        winrate = wins / len(group) if group else 0.0
        expectancy = sum(pnls) / len(group) if group else 0.0
        equity_curve = _compute_equity(group)
        max_dd = _max_drawdown(equity_curve)
        stability = _stability(pnls)
        summaries.append(
            SummaryRow(
                key=f"{label}:{key}",
                trades=len(group),
                winrate=winrate,
                expectancy=expectancy,
                max_drawdown=max_dd,
                stability=stability,
            )
        )
    return sorted(summaries, key=lambda row: row.key)


def summarize_combinations(trades: list[Trade]) -> list[SummaryRow]:
    return summarize(
        trades,
        lambda t: f"{t.mmxm_phase}|{t.entry_method}|{'Tradable' if t.ob_tradable else 'NonTradable'}",
        "Combo",
    )


def summarize_day_labels(trades: list[Trade]) -> list[SummaryRow]:
    return summarize(trades, lambda t: t.day_label or "UNKNOWN", "DayLabel")


def select_candidates(
    rows: list[SummaryRow],
    min_trades: int = 25,
    min_expectancy: float = 0.0,
    max_dd: float | None = None,
) -> list[SummaryRow]:
    """Filter candidate summary rows by trade count, expectancy, and drawdown."""
    filtered = [
        row
        for row in rows
        if row.trades >= min_trades and row.expectancy >= min_expectancy
    ]
    if max_dd is not None:
        filtered = [row for row in filtered if row.max_drawdown <= max_dd]
    return filtered


def rank_summaries(rows: list[SummaryRow]) -> list[SummaryRow]:
    """Rank summaries by expectancy, drawdown, stability, and key."""
    return sorted(
        rows,
        key=lambda row: (-row.expectancy, row.max_drawdown, -row.stability, row.key),
    )
