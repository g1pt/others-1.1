"""Configuration for paper execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from typing import Mapping

from .risk import RiskLimits


class ExecutionMode(str, Enum):
    LOG_ONLY = "LOG_ONLY"
    PAPER_SIM = "PAPER_SIM"
    LIVE_BROKER = "LIVE_BROKER"


@dataclass(frozen=True)
class RulesetConfig:
    setup_id: str
    timeframe: str
    entry_type: str
    phase: str
    ob_tradability: str
    enabled: bool = True


@dataclass(frozen=True)
class PaperEngineConfig:
    mode: ExecutionMode = ExecutionMode.LOG_ONLY
    risk_limits: RiskLimits = RiskLimits()
    risk_per_trade_pct: float = 0.005
    risk_mode: str = "fixed_per_trade"
    daily_risk_budget_pct: float = 0.02
    min_risk_per_trade_pct: float = 0.003
    max_risk_per_trade_pct: float = 0.01
    st_pct: float = 0.002
    sl_pct: float = 0.002
    symbol_map: Mapping[str, str] = field(default_factory=dict)
    rulesets: Mapping[str, RulesetConfig] = field(default_factory=dict)
    log_dir: Path = Path("logs")

    @classmethod
    def from_env(cls) -> "PaperEngineConfig":
        def _env(key: str, default: str) -> str:
            return os.getenv(key, default)

        rr_default = float(_env("RR_DEFAULT", "2.0"))
        _ = rr_default  # kept for STEP2 guard print parity
        st_pct = float(_env("ST_PCT", "0.002"))
        risk_per_trade = float(_env("RISK_PER_TRADE_PCT", str(cls.risk_per_trade_pct)))

        limits = RiskLimits(
            max_trades_per_day=int(_env("MAX_TRADES_PER_DAY", "1")),
            stop_after_consecutive_losses=int(_env("MAX_CONSEC_LOSSES", "2")),
            daily_drawdown_stop_pct=float(_env("DAILY_DD_STOP_PCT", "0.02")),
            hard_max_drawdown_pct=float(_env("HARD_DD_STOP_PCT", "0.03")),
        )

        return cls(
            risk_limits=limits,
            risk_per_trade_pct=risk_per_trade,
            risk_mode=_env("RISK_MODE", "fixed_per_trade"),
            daily_risk_budget_pct=float(_env("DAILY_RISK_BUDGET_PCT", "0.02")),
            min_risk_per_trade_pct=float(_env("MIN_RISK_PER_TRADE_PCT", "0.003")),
            max_risk_per_trade_pct=float(_env("MAX_RISK_PER_TRADE_PCT", "0.01")),
            st_pct=st_pct,
            sl_pct=st_pct,
        )


def default_rulesets() -> dict[str, RulesetConfig]:
    return {
        "SP500": RulesetConfig(
            setup_id="MMXM_4C_D",
            timeframe="30",
            entry_type="Refinement",
            phase="Manipulation",
            ob_tradability="Tradable",
            enabled=True,
        ),
        "EURUSD": RulesetConfig(
            setup_id="EURUSD_4C_D",
            timeframe="30",
            entry_type="Refinement",
            phase="Manipulation",
            ob_tradability="Tradable",
            enabled=False,
        ),
    }


def default_symbol_map() -> dict[str, str]:
    return {
        "FX_SPX500": "SP500",
        "SPX500": "SP500",
        "SP500": "SP500",
        "EURUSD": "EURUSD",
    }
