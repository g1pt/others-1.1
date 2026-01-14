import uuid
from datetime import datetime

from app.agent_rules import SYMBOL_MAP, SYMBOL_RULESETS
from app.config import (
    DD_DAILY_STOP,
    DD_HARD_SYSTEM,
    DD_SOFT,
    MAX_CONSEC_LOSSES,
    MAX_TRADES_PER_DAY,
    REDUCED_RISK_PCT,
    RISK_PCT_DEFAULT,
    RISK_PCT_MAX,
    RR_DEFAULT,
    SL_DISTANCE_MODE,
    SL_FIXED_PCT,
)
from app.models import Candle, EquityState, PaperOrder, TVWebhook
from app.utils.time import utc_date_str, utc_now


def normalize_timeframe(timeframe: int | str) -> str:
    if isinstance(timeframe, int):
        return str(timeframe)
    if isinstance(timeframe, str) and timeframe.isdigit():
        return timeframe
    raise ValueError("Invalid timeframe")


def normalize_direction(direction: str) -> str:
    normalized = {
        "buy": "buy",
        "long": "buy",
        "sell": "sell",
        "short": "sell",
    }.get(direction.lower())
    if normalized is None:
        raise ValueError("Invalid direction")
    return normalized


def resolve_symbol(raw_symbol: str) -> str | None:
    return SYMBOL_MAP.get(raw_symbol)


def validate_signal(payload: TVWebhook, symbol: str, timeframe: str) -> tuple[bool, str]:
    ruleset = SYMBOL_RULESETS.get(symbol)
    if not ruleset or not ruleset.get("enabled"):
        return False, "symbol_disabled"
    if timeframe not in ruleset["allowed_timeframes"]:
        return False, "timeframe_not_allowed"
    if payload.setup != ruleset["setup_id"]:
        return False, "setup_not_allowed"
    if payload.entry_type != ruleset["entry_type"]:
        return False, "entry_type_not_allowed"
    if payload.phase != ruleset["phase"]:
        return False, "phase_not_allowed"
    if payload.ob_tradability != ruleset["ob_tradability"]:
        return False, "ob_tradability_not_allowed"
    return True, "ok"


def _reset_daily_if_new(state: EquityState, now: datetime) -> None:
    today = utc_date_str(now)
    if state.daily_date != today:
        state.daily_date = today
        state.daily_start_equity = state.equity
        state.daily_dd_pct = 0.0
        state.trades_today = 0
        state.consec_losses = 0


def can_open_trade(state: EquityState, now: datetime) -> tuple[bool, str]:
    _reset_daily_if_new(state, now)
    if state.max_dd_pct >= DD_HARD_SYSTEM:
        return False, "blocked_hard_dd"
    if state.daily_dd_pct >= DD_DAILY_STOP:
        return False, "blocked_daily_dd"
    if state.trades_today >= MAX_TRADES_PER_DAY:
        return False, "blocked_max_trades_day"
    if state.consec_losses >= MAX_CONSEC_LOSSES:
        return False, "blocked_max_consec_losses"
    return True, "ok"


def compute_risk_pct(state: EquityState) -> float:
    threshold = state.high_watermark * (1 - DD_SOFT)
    if state.equity < threshold:
        return min(REDUCED_RISK_PCT, RISK_PCT_MAX)
    return min(RISK_PCT_DEFAULT, RISK_PCT_MAX)


def _stop_distance(entry_price: float) -> float:
    if SL_DISTANCE_MODE != "fixed_pct":
        return entry_price * SL_FIXED_PCT
    return entry_price * SL_FIXED_PCT


def build_order(
    payload: TVWebhook,
    equity: float,
    risk_pct: float,
    symbol: str,
    timeframe: str,
    direction: str,
    now: datetime | None = None,
) -> PaperOrder:
    entry_price = payload.price
    stop_distance = _stop_distance(entry_price)
    direction_multiplier = 1 if direction == "buy" else -1
    stop_loss = entry_price - (direction_multiplier * stop_distance)
    take_profit = entry_price + (direction_multiplier * stop_distance * RR_DEFAULT)
    risk_cash = equity * risk_pct
    position_size = risk_cash / stop_distance if stop_distance > 0 else 0.0

    opened_at = (now or utc_now()).isoformat()
    return PaperOrder(
        id=str(uuid.uuid4()),
        symbol=symbol,
        timeframe=timeframe,
        setup_id=payload.setup,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_pct=risk_pct,
        risk_cash=risk_cash,
        position_size=position_size,
        status="OPEN",
        opened_utc=opened_at,
    )


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def evaluate_candle_hit(order: PaperOrder, candle: Candle) -> tuple[bool, str, float, float]:
    sl_hit = False
    tp_hit = False
    if order.direction == "buy":
        sl_hit = candle.low <= order.stop_loss
        tp_hit = candle.high >= order.take_profit
    else:
        sl_hit = candle.high >= order.stop_loss
        tp_hit = candle.low <= order.take_profit

    if sl_hit and tp_hit:
        pnl_r = -1.0
        exit_price = order.stop_loss
        return True, "stop_loss", pnl_r, exit_price
    if sl_hit:
        pnl_r = -1.0
        exit_price = order.stop_loss
        return True, "stop_loss", pnl_r, exit_price
    if tp_hit:
        pnl_r = RR_DEFAULT
        exit_price = order.take_profit
        return True, "take_profit", pnl_r, exit_price
    return False, "", 0.0, 0.0


def close_order(
    state: EquityState,
    order: PaperOrder,
    close_time: datetime,
    pnl_r: float,
) -> None:
    _reset_daily_if_new(state, close_time)
    pnl_cash = order.risk_cash * pnl_r
    order.status = "CLOSED"
    order.closed_utc = close_time.isoformat()
    order.pnl_r = pnl_r
    order.pnl_cash = pnl_cash

    state.equity += pnl_cash
    if state.equity > state.high_watermark:
        state.high_watermark = state.equity

    if state.high_watermark > 0:
        state.max_dd_pct = max(
            state.max_dd_pct,
            (state.high_watermark - state.equity) / state.high_watermark,
        )

    if state.daily_start_equity > 0:
        state.daily_dd_pct = max(
            0.0,
            (state.daily_start_equity - state.equity) / state.daily_start_equity,
        )

    if pnl_cash > 0:
        state.consec_losses = 0
    elif pnl_cash < 0:
        state.consec_losses += 1


def simulate_backfill(
    state: EquityState,
    orders: list[PaperOrder],
    candles: list[Candle],
) -> list[PaperOrder]:
    closed_orders: list[PaperOrder] = []
    for order in orders:
        if order.status != "OPEN":
            continue
        opened_time = parse_timestamp(order.opened_utc)
        for candle in candles:
            if opened_time and candle.timestamp < opened_time:
                continue
            hit, _, pnl_r, _ = evaluate_candle_hit(order, candle)
            if not hit:
                continue
            close_order(state, order, candle.timestamp, pnl_r)
            closed_orders.append(order)
            break
    return closed_orders
