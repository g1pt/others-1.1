from datetime import datetime, timezone

from app.models import Candle, PaperOrder
from app.paper_engine import evaluate_candle_hit


def _order(direction: str) -> PaperOrder:
    return PaperOrder(
        id="1",
        symbol="SP500",
        timeframe="30",
        setup_id="MMXM_4C_D",
        direction=direction,
        entry_price=100.0,
        stop_loss=99.8,
        take_profit=100.4,
        risk_pct=0.005,
        risk_cash=50.0,
        position_size=250.0,
        status="OPEN",
        opened_utc="2024-01-01T00:00:00Z",
    )


def test_same_candle_tp_sl_prefers_sl_long():
    order = _order("buy")
    candle = Candle(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=100.0,
        high=100.5,
        low=99.7,
        close=100.1,
    )
    hit, reason, pnl_r, _ = evaluate_candle_hit(order, candle)
    assert hit is True
    assert reason == "stop_loss"
    assert pnl_r == -1.0


def test_same_candle_tp_sl_prefers_sl_short():
    order = _order("sell")
    candle = Candle(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=100.0,
        high=100.5,
        low=99.5,
        close=99.9,
    )
    hit, reason, pnl_r, _ = evaluate_candle_hit(order, candle)
    assert hit is True
    assert reason == "stop_loss"
    assert pnl_r == -1.0
