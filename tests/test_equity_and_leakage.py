from __future__ import annotations

from math import isclose

from src.models import Trade
from src.reporting import leakage_contribution_stats
from src.risk import simulate_equity


def test_equity_simulation_drawdown() -> None:
    trades = [
        Trade(
            entry_time="2024-01-01 09:00",
            entry_price=100.0,
            stop_price=90.0,
            direction="bullish",
            mmxm_phase="Expansion",
            entry_method="Risk Entry",
            ob_tradable=True,
            ob_id=1,
            exit_time="2024-01-01 10:00",
            exit_price=110.0,
            pnl_r=1.0,
        ),
        Trade(
            entry_time="2024-01-01 12:00",
            entry_price=100.0,
            stop_price=95.0,
            direction="bullish",
            mmxm_phase="Expansion",
            entry_method="Risk Entry",
            ob_tradable=True,
            ob_id=2,
            exit_time="2024-01-01 13:00",
            exit_price=105.0,
            pnl_r=1.0,
        ),
        Trade(
            entry_time="2024-01-02 09:00",
            entry_price=200.0,
            stop_price=210.0,
            direction="bearish",
            mmxm_phase="Distribution",
            entry_method="Risk Entry",
            ob_tradable=False,
            ob_id=3,
            exit_time="2024-01-02 10:00",
            exit_price=215.0,
            pnl_r=-1.5,
        ),
    ]

    result = simulate_equity(trades, initial_equity=10_000.0, risk_per_trade=0.01)

    assert isclose(result.final_equity, 10047.985, rel_tol=1e-6)
    assert isclose(result.max_drawdown_currency, 153.015, rel_tol=1e-6)


def test_profit_contribution_totals() -> None:
    trades = [
        Trade(
            entry_time="2024-01-01 09:00",
            entry_price=100.0,
            stop_price=90.0,
            direction="bullish",
            mmxm_phase="Expansion",
            entry_method="Risk Entry",
            ob_tradable=True,
            ob_id=1,
            exit_time="2024-01-01 10:00",
            exit_price=110.0,
            pnl_r=1.0,
        ),
        Trade(
            entry_time="2024-01-01 12:00",
            entry_price=200.0,
            stop_price=210.0,
            direction="bearish",
            mmxm_phase="Distribution",
            entry_method="Risk Entry",
            ob_tradable=False,
            ob_id=2,
            exit_time="2024-01-01 13:00",
            exit_price=190.0,
            pnl_r=1.0,
        ),
    ]

    result = simulate_equity(trades, initial_equity=10_000.0, risk_per_trade=0.01)
    buckets = leakage_contribution_stats(result.trade_results)
    entry_bucket = next(bucket for bucket in buckets if bucket.title == "By EntryType")
    contribution_total = sum(row.contribution_pct for row in entry_bucket.rows)
    assert isclose(contribution_total, 100.0, rel_tol=1e-6)
