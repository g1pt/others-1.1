from datetime import datetime, timezone

from app.config import (
    DD_DAILY_STOP,
    DD_HARD_SYSTEM,
    DD_SOFT,
    MAX_CONSEC_LOSSES,
    MAX_TRADES_PER_DAY,
    REDUCED_RISK_PCT,
    RISK_PCT_DEFAULT,
)
from app.models import EquityState
from app.paper_engine import can_open_trade, compute_risk_pct
from app.utils.time import utc_date_str


def _state() -> EquityState:
    today = utc_date_str(datetime(2024, 1, 1, tzinfo=timezone.utc))
    return EquityState(
        equity=10000.0,
        high_watermark=10000.0,
        max_dd_pct=0.0,
        daily_date=today,
        daily_start_equity=10000.0,
        daily_dd_pct=0.0,
        trades_today=0,
        consec_losses=0,
    )


def test_daily_limits_block():
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    state = _state()
    state.trades_today = MAX_TRADES_PER_DAY
    ok, reason = can_open_trade(state, now)
    assert ok is False
    assert reason == "blocked_max_trades_day"

    state = _state()
    state.daily_dd_pct = DD_DAILY_STOP
    ok, reason = can_open_trade(state, now)
    assert ok is False
    assert reason == "blocked_daily_dd"

    state = _state()
    state.max_dd_pct = DD_HARD_SYSTEM
    ok, reason = can_open_trade(state, now)
    assert ok is False
    assert reason == "blocked_hard_dd"

    state = _state()
    state.consec_losses = MAX_CONSEC_LOSSES
    ok, reason = can_open_trade(state, now)
    assert ok is False
    assert reason == "blocked_max_consec_losses"


def test_risk_pct_reduces_on_drawdown():
    state = _state()
    state.high_watermark = 10000.0
    state.equity = 10000.0 * (1 - DD_SOFT) - 1
    assert compute_risk_pct(state) == REDUCED_RISK_PCT

    state = _state()
    assert compute_risk_pct(state) == RISK_PCT_DEFAULT
