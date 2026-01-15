from src.execution.risk import compute_qty, compute_sl
from src.execution.tp_routing import resolve_tp_strategy
from src.execution.tp_strategies import TP_FixedR


def test_fixedr_buy():
    strategy = TP_FixedR(r_multiple=2)
    tp = strategy.get_tp_level(entry=100, sl=90, data=None, direction="buy")
    assert tp == 120


def test_fixedr_sell():
    strategy = TP_FixedR(r_multiple=3)
    tp = strategy.get_tp_level(entry=100, sl=110, data=None, direction="sell")
    assert tp == 70


def test_resolve_default():
    strategy = resolve_tp_strategy("UNKNOWN")
    assert isinstance(strategy, TP_FixedR)
    tp = strategy.get_tp_level(entry=100, sl=90, data=None, direction="buy")
    assert tp == 120


def test_compute_sl_buy():
    sl = compute_sl(100, "buy", st_pct=0.002)
    assert sl == 99.8


def test_compute_sl_sell():
    sl = compute_sl(100, "sell", st_pct=0.002)
    assert sl == 100.2


def test_qty_nonzero():
    qty = compute_qty(10000, 0.005, 100, 99.8)
    assert qty > 0
