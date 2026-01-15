"""Risk guards for paper execution."""
from __future__ import annotations

from dataclasses import dataclass

from .models import RejectionReason


@dataclass(frozen=True)
class RiskConfig:
    max_trades_per_day: int = 1
    max_consecutive_losses: int = 2
    daily_drawdown_limit_pct: float = 2.0
    hard_drawdown_limit_pct: float = 3.0


def can_take_trade(today_stats: dict[str, float | int], config: RiskConfig) -> tuple[bool, RejectionReason | None]:
    """Return whether a trade can be taken based on current stats."""
    trades_today = int(today_stats.get("trades_today_count", 0))
    consecutive_losses = int(today_stats.get("consecutive_losses", 0))
    daily_drawdown_pct = float(today_stats.get("daily_drawdown_pct", 0.0))
    overall_drawdown_pct = float(today_stats.get("overall_drawdown_pct", 0.0))

    if config.hard_drawdown_limit_pct and overall_drawdown_pct >= config.hard_drawdown_limit_pct:
        return False, RejectionReason(
            code="hard_drawdown_stop",
            message="System drawdown limit reached",
            details={
                "overall_drawdown_pct": overall_drawdown_pct,
                "limit_pct": config.hard_drawdown_limit_pct,
            },
        )
    if config.daily_drawdown_limit_pct and daily_drawdown_pct >= config.daily_drawdown_limit_pct:
        return False, RejectionReason(
            code="daily_drawdown_limit",
            message="Daily drawdown limit reached",
            details={
                "daily_drawdown_pct": daily_drawdown_pct,
                "limit_pct": config.daily_drawdown_limit_pct,
            },
        )
    if config.max_consecutive_losses and consecutive_losses >= config.max_consecutive_losses:
        return False, RejectionReason(
            code="consecutive_losses",
            message="Consecutive loss limit reached",
            details={
                "consecutive_losses": consecutive_losses,
                "limit": config.max_consecutive_losses,
            },
        )
    if config.max_trades_per_day and trades_today >= config.max_trades_per_day:
        return False, RejectionReason(
            code="max_trades_per_day",
            message="Daily trade count limit reached",
            details={
                "trades_today_count": trades_today,
                "limit": config.max_trades_per_day,
            },
        )
    return True, None
