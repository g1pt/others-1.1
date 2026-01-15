"""Configuration for paper execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
    sl_pct: float = 0.002
    symbol_map: Mapping[str, str] = field(default_factory=dict)
    rulesets: Mapping[str, RulesetConfig] = field(default_factory=dict)
    log_dir: Path = Path("logs")


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
