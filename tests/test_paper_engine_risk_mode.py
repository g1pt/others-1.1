import json

from src.execution.config import ExecutionMode, PaperEngineConfig, RulesetConfig
from src.execution.ledger import Ledger
from src.execution.paper_engine import PaperEngine
from src.execution.risk import RiskLimits
from src.models import Candle


def _payload() -> dict[str, object]:
    return {
        "symbol": "SP500",
        "timeframe": "30",
        "setup": "MMXM_4C_D",
        "entry_type": "Refinement",
        "phase": "Manipulation",
        "ob_tradability": "Tradable",
        "direction": "buy",
        "price": 100.0,
        "timestamp": "2024-01-01T09:00:00+00:00",
    }


def _rulesets() -> dict[str, RulesetConfig]:
    return {
        "SP500": RulesetConfig(
            setup_id="MMXM_4C_D",
            timeframe="30",
            entry_type="Refinement",
            phase="Manipulation",
            ob_tradability="Tradable",
            enabled=True,
        )
    }


def _loader(symbol: str, timeframe: str) -> list[Candle]:
    _ = (symbol, timeframe)
    return [
        Candle(timestamp="2024-01-01T09:00:00+00:00", open=100, high=100.1, low=99.9, close=100.0),
        Candle(timestamp="2024-01-01T09:30:00+00:00", open=100, high=100.1, low=99.9, close=100.0),
    ]


def _last_event(log_path):
    with log_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    return json.loads(lines[-1])


def test_fixed_per_trade_risk_cash(tmp_path):
    ledger = Ledger.load_or_init(tmp_path / "logs", initial_equity=10_000.0)
    config = PaperEngineConfig(
        mode=ExecutionMode.PAPER_SIM,
        risk_limits=RiskLimits(max_trades_per_day=2),
        risk_per_trade_pct=0.005,
        risk_mode="fixed_per_trade",
        rulesets=_rulesets(),
    )
    engine = PaperEngine(config, ledger, data_loader=_loader)

    ok, reason, trade_id = engine.process_signal(_payload())

    assert ok is True
    assert reason == "ok"
    assert trade_id is not None

    event = _last_event(ledger.events_log_path)
    assert event["risk_cash"] == 50.0


def test_daily_budget_risk_cash_uses_equity_budget(tmp_path):
    ledger = Ledger.load_or_init(tmp_path / "logs", initial_equity=10_000.0)
    config = PaperEngineConfig(
        mode=ExecutionMode.PAPER_SIM,
        risk_limits=RiskLimits(max_trades_per_day=2),
        risk_per_trade_pct=0.005,
        risk_mode="daily_budget",
        daily_risk_budget_pct=0.02,
        min_risk_per_trade_pct=0.001,
        max_risk_per_trade_pct=0.02,
        rulesets=_rulesets(),
    )
    engine = PaperEngine(config, ledger, data_loader=_loader)

    ok, reason, trade_id = engine.process_signal(_payload())

    assert ok is True
    assert reason == "ok"
    assert trade_id is not None

    event = _last_event(ledger.events_log_path)
    assert event["risk_cash"] == 100.0
