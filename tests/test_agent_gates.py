from app.models import TVWebhook
from app.paper_engine import normalize_direction, normalize_timeframe, validate_signal


def _payload(**kwargs):
    data = {
        "secret": "secret",
        "symbol": "SP500",
        "timeframe": "30",
        "timestamp": "2024-01-01T00:00:00Z",
        "setup": "MMXM_4C_D",
        "entry_type": "Refinement",
        "phase": "Manipulation",
        "ob_tradability": "Tradable",
        "direction": "buy",
        "price": 100.0,
    }
    data.update(kwargs)
    return TVWebhook(**data)


def test_validate_signal_ok():
    payload = _payload()
    timeframe = normalize_timeframe(payload.timeframe)
    ok, reason = validate_signal(payload, "SP500", timeframe)
    assert ok is True
    assert reason == "ok"


def test_validate_signal_rejects_setup():
    payload = _payload(setup="WRONG")
    timeframe = normalize_timeframe(payload.timeframe)
    ok, reason = validate_signal(payload, "SP500", timeframe)
    assert ok is False
    assert reason == "setup_not_allowed"


def test_normalize_direction():
    assert normalize_direction("buy") == "buy"
    assert normalize_direction("short") == "sell"
