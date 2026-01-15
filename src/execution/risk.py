"""Risk guards for paper execution."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from .models import RejectionReason


@dataclass(frozen=True)
class RiskConfig:
    max_trades_per_day: int = 1
    max_consecutive_losses: int = 2
    daily_drawdown_limit_pct: float = 2.0
    hard_drawdown_limit_pct: float = 3.0


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade_pct: float = 0.005
    max_trades_per_day: int = 1
    stop_after_losses: int = 2
    daily_dd_stop_pct: float = 0.02
    hard_dd_cap_pct: float = 0.03


def compute_sl(entry: float, direction: str, st_pct: float = 0.002) -> float:
    """Compute a fixed-percentage stop loss level."""
    if direction == "buy":
        return entry * (1 - st_pct)
    if direction == "sell":
        return entry * (1 + st_pct)
    raise ValueError("direction must be 'buy' or 'sell'")


def compute_qty(equity: float, risk_pct: float, entry: float, sl: float) -> float:
    """Compute position size based on fixed risk percentage."""
    distance = abs(entry - sl)
    if distance <= 0:
        return 0.0
    return (equity * risk_pct) / distance


def compute_position_size(equity: float, risk_per_trade: float, entry: float, sl: float) -> float:
    """Alias for compute_qty with clarified naming for paper execution."""
    return compute_qty(equity, risk_per_trade, entry, sl)


def daily_key(value: datetime | date | str) -> str:
    """Return YYYY-MM-DD from datetime/date/ISO string."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(cleaned).date().isoformat()
        except ValueError:
            return cleaned[:10]
    raise TypeError("Unsupported date type")


def update_consecutive_losses(previous: int, trade_pnl_r: float) -> int:
    """Update consecutive losses counter based on trade outcome."""
    if trade_pnl_r < 0:
        return previous + 1
    return 0


def can_open_trade(
    ledger_state: dict[str, float | int],
    trade_date: str | RiskLimits | None,
    limits: RiskLimits | None = None,
) -> tuple[bool, str | None]:
    """Return whether a trade can be opened based on ledger stats."""
    if isinstance(trade_date, RiskLimits) and limits is None:
        limits = trade_date
    if limits is None:
        raise ValueError("Risk limits must be provided")
    _ = trade_date
    trades_today = int(ledger_state.get("trades_today_count", 0))
    consecutive_losses = int(ledger_state.get("consecutive_losses", 0))
    daily_dd = float(ledger_state.get("daily_drawdown_pct", 0.0))
    overall_dd = float(ledger_state.get("overall_drawdown_pct", 0.0))

    if limits.hard_dd_cap_pct and overall_dd >= limits.hard_dd_cap_pct:
        return False, "hard_drawdown_stop"
    if limits.daily_dd_stop_pct and daily_dd >= limits.daily_dd_stop_pct:
        return False, "daily_drawdown_stop"
    if limits.stop_after_losses and consecutive_losses >= limits.stop_after_losses:
        return False, "loss_streak_stop"
    if limits.max_trades_per_day and trades_today >= limits.max_trades_per_day:
        return False, "max_trades_per_day"
    return True, None

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
            code="daily_drawdown_stop",
            message="Daily drawdown limit reached",
            details={
                "daily_drawdown_pct": daily_drawdown_pct,
                "limit_pct": config.daily_drawdown_limit_pct,
            },
        )
    if config.max_consecutive_losses and consecutive_losses >= config.max_consecutive_losses:
        return False, RejectionReason(
            code="loss_streak_stop",
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
