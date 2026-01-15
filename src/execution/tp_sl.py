"""Stop-loss and take-profit helpers for paper execution."""
from __future__ import annotations


def _normalize_direction(direction: str) -> str:
    normalized = direction.strip().lower()
    if normalized in {"buy", "long"}:
        return "buy"
    if normalized in {"sell", "short"}:
        return "sell"
    raise ValueError("direction must be buy/long or sell/short")


def compute_sl(entry: float, direction: str, st_pct: float = 0.002) -> float:
    """Compute stop-loss price using a fixed percentage distance."""
    side = _normalize_direction(direction)
    if side == "buy":
        return entry * (1 - st_pct)
    return entry * (1 + st_pct)


def compute_tp(entry: float, sl: float, direction: str, rr: float = 2.0) -> float:
    """Compute take-profit price using fixed R:R multiple."""
    side = _normalize_direction(direction)
    stop_distance = abs(entry - sl)
    if side == "buy":
        return entry + (stop_distance * rr)
    return entry - (stop_distance * rr)
