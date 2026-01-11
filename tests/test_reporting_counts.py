from src.models import OrderBlock
from src.research.reporting import ob_failure_counts


def test_ob_failure_counts() -> None:
    order_blocks = [
        OrderBlock(
            ob_id=1,
            index=10,
            timestamp="t1",
            direction="bullish",
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
            range=1.5,
            impulse_end=12,
            has_imbalance=True,
            has_bos=True,
            near_level=True,
            after_sweep=False,
            tradable=True,
            fail_reasons=[],
        ),
        OrderBlock(
            ob_id=2,
            index=20,
            timestamp="t2",
            direction="bearish",
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
            range=1.5,
            impulse_end=22,
            has_imbalance=False,
            has_bos=True,
            near_level=True,
            after_sweep=False,
            tradable=False,
            fail_reasons=["no_fvg"],
        ),
        OrderBlock(
            ob_id=3,
            index=30,
            timestamp="t3",
            direction="bullish",
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
            range=1.5,
            impulse_end=32,
            has_imbalance=True,
            has_bos=False,
            near_level=False,
            after_sweep=False,
            tradable=False,
            fail_reasons=["no_bos", "not_near_level"],
        ),
        OrderBlock(
            ob_id=4,
            index=40,
            timestamp="t4",
            direction="bearish",
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
            range=1.5,
            impulse_end=42,
            has_imbalance=True,
            has_bos=True,
            near_level=False,
            after_sweep=False,
            tradable=False,
            fail_reasons=["not_near_level"],
        ),
    ]

    counts = ob_failure_counts(order_blocks)

    assert counts["total"] == 4
    assert counts["tradable"] == 1
    assert counts["nontradable"] == 3
    assert counts["no_fvg"] == 1
    assert counts["no_bos"] == 1
    assert counts["not_near_level"] == 2
