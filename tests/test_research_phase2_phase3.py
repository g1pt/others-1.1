from scripts import run_mmxm_research as mod
from src.models import Candle, Trade


def test_is_in_killzone_true_for_london_overlap() -> None:
    assert mod._is_in_killzone("2024-01-01T08:00:00+00:00", ((7, 11), (13, 16)))


def test_entry_quality_rejects_low_impulse() -> None:
    candles = [
        Candle("2024-01-01T07:59:00+00:00", 100, 100.02, 99.99, 100.01),
        Candle("2024-01-01T08:00:00+00:00", 100.01, 100.02, 100.00, 100.01),
    ]
    idx = {c.timestamp: i for i, c in enumerate(candles)}
    trade = Trade(
        entry_time="2024-01-01T08:00:00+00:00",
        entry_price=100.0,
        direction="bullish",
        mmxm_phase="Manipulation",
        entry_method="Refinement Entry",
        ob_tradable=True,
        ob_id=1,
        stop_price=99.0,
    )
    cfg = mod.EntryQualityConfig(min_impulse_pct=0.01, min_ob_range_pct=0.0001)
    assert mod._entry_quality_ok(trade, candles, idx, cfg) is False


def test_simulate_rr_trade_time_stop_applies() -> None:
    candles = [
        Candle("2024-01-01T08:00:00+00:00", 100, 100.2, 99.8, 100.0),
        Candle("2024-01-01T08:01:00+00:00", 100.0, 100.05, 99.95, 100.01),
        Candle("2024-01-01T08:02:00+00:00", 100.01, 100.03, 99.97, 100.02),
    ]
    idx = {c.timestamp: i for i, c in enumerate(candles)}
    trade = Trade(
        entry_time="2024-01-01T08:00:00+00:00",
        entry_price=100.0,
        direction="bullish",
        mmxm_phase="Manipulation",
        entry_method="Refinement Entry",
        ob_tradable=True,
        ob_id=1,
        stop_price=99.0,
    )
    out = mod._simulate_rr_trade(
        trade,
        candles,
        idx,
        sl_pct=0.01,
        rr=3.0,
        symbol="FX_SPX500",
        trade_management=mod.TradeManagementConfig(time_stop_candles=1, partial_close_fraction=0.0),
    )
    assert out is not None
    assert out.exit_time == "2024-01-01T08:01:00+00:00"
