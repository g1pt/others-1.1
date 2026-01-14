from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SetupGate:
    setup_id: str
    timeframe: str
    entry_type: str
    phase: str
    ob_tradability: str


@dataclass(frozen=True)
class AgentSpec:
    allowed_symbols: set[str]
    symbol_map: Mapping[str, str]
    setups: Mapping[str, SetupGate]

    def normalize_symbol(self, symbol: str) -> str:
        return self.symbol_map.get(symbol, symbol)

    def gate_for(self, symbol: str) -> SetupGate | None:
        return self.setups.get(symbol)


def build_default_spec(symbol_map: Mapping[str, str]) -> AgentSpec:
    sp500_gate = SetupGate(
        setup_id="MMXM_4C_D",
        timeframe="30",
        entry_type="Refinement",
        phase="Manipulation",
        ob_tradability="Tradable",
    )
    eurusd_gate = SetupGate(
        setup_id="MMXM_EU_4C_D",
        timeframe="30",
        entry_type="Refinement",
        phase="Manipulation",
        ob_tradability="Tradable",
    )
    allowed_symbols = {"SP500", "EURUSD"}
    setups = {"SP500": sp500_gate, "EURUSD": eurusd_gate}
    return AgentSpec(
        allowed_symbols=allowed_symbols,
        symbol_map=symbol_map,
        setups=setups,
    )
