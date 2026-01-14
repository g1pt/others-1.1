from __future__ import annotations

from typing import Any, Mapping

from src.engine.agent_spec import AgentSpec


def _normalize_timeframe(timeframe: str | int) -> str:
    return str(timeframe)


def validate(
    payload: Mapping[str, Any],
    *,
    spec: AgentSpec,
    secret: str,
) -> tuple[bool, str, dict[str, Any]]:
    normalized: dict[str, Any] = dict(payload)

    if normalized.get("secret") != secret:
        return False, "secret mismatch", normalized

    symbol = normalized.get("symbol")
    if not symbol:
        return False, "symbol missing", normalized
    normalized_symbol = spec.normalize_symbol(str(symbol))
    normalized["symbol"] = normalized_symbol

    if normalized_symbol not in spec.allowed_symbols:
        return False, "symbol not allowed", normalized

    gate = spec.gate_for(normalized_symbol)
    if gate is None:
        return False, "setup not configured", normalized

    timeframe_value = _normalize_timeframe(normalized.get("timeframe", ""))
    normalized["timeframe"] = timeframe_value
    if timeframe_value != gate.timeframe:
        return False, "timeframe not allowed", normalized

    if normalized.get("setup") != gate.setup_id:
        return False, "setup not allowed", normalized

    if normalized.get("entry_type") != gate.entry_type:
        return False, "entry_type not allowed", normalized

    if normalized.get("phase") != gate.phase:
        return False, "phase not allowed", normalized

    if normalized.get("ob_tradability") != gate.ob_tradability:
        return False, "ob_tradability not allowed", normalized

    direction = normalized.get("direction") or normalized.get("side")
    if not direction:
        return False, "direction missing", normalized
    normalized["direction"] = str(direction).lower()

    return True, "accepted", normalized
