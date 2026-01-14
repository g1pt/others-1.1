from datetime import datetime

from app.config import (
    DAILY_MAX_DD_PCT,
    HARD_MAX_DD_PCT,
    HIGH_WATERMARK_RISK_REDUCE_PCT,
    MAX_CONSEC_LOSSES,
    MAX_TRADES_PER_DAY,
    RISK_PCT_BASE,
    RISK_PCT_MAX,
)
from app.models import LedgerState
from app.utils.time import utc_date_str


def _reset_daily_if_new(state: LedgerState, now: datetime) -> None:
    today = utc_date_str(now)
    if state.daily_date != today:
        state.daily_date = today
        state.daily_start_equity = state.equity
        state.daily_dd_pct = 0.0
        state.trades_today = 0


def can_open_trade(state: LedgerState, now: datetime, symbol: str) -> tuple[bool, str]:
    _reset_daily_if_new(state, now)

    if state.trades_today >= MAX_TRADES_PER_DAY:
        return False, "max_trades_per_day"
    if state.daily_dd_pct >= DAILY_MAX_DD_PCT:
        return False, "daily_max_dd"
    if state.max_dd_pct >= HARD_MAX_DD_PCT:
        return False, "hard_max_dd"
    if state.consec_losses >= MAX_CONSEC_LOSSES:
        return False, "max_consec_losses"
    return True, "ok"


def compute_risk_pct(state: LedgerState) -> float:
    if state.equity < state.high_watermark:
        return min(HIGH_WATERMARK_RISK_REDUCE_PCT, RISK_PCT_MAX)
    return min(RISK_PCT_BASE, RISK_PCT_MAX)


def update_after_close(state: LedgerState, pnl_cash: float, won: bool) -> None:
    state.equity += pnl_cash
    if state.equity > state.high_watermark:
        state.high_watermark = state.equity

    if state.high_watermark > 0:
        state.max_dd_pct = max(
            0.0,
            (state.high_watermark - state.equity) / state.high_watermark,
        )

    if state.daily_start_equity > 0:
        state.daily_dd_pct = max(
            0.0,
            (state.daily_start_equity - state.equity) / state.daily_start_equity,
        )

    if won:
        state.consec_losses = 0
    else:
        state.consec_losses += 1
