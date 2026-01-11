"""Filtering utilities for combo-based trade setup screening."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from src.data import Candle
from src.entries import EntrySignal


GLOBAL_MAX_DRAWDOWN = 3.0
GLOBAL_EXPECTANCY_MIN = 0.0

TIMEFRAME_THRESHOLDS: dict[str, dict[str, float]] = {
    "1m": {"trades": 20, "stability": 0.40},
    "5m": {"trades": 20, "stability": 0.40},
    "15m": {"trades": 20, "stability": 0.40},
    "30m": {"trades": 20, "stability": 0.40},
    "60m": {"trades": 7, "stability": 0.38},
}


ComboKey = tuple[str, str, str]


@dataclass(frozen=True)
class ComboMetrics:
    timeframe: str
    phase: str
    entry_type: str
    ob_type: str
    expectancy: float
    max_drawdown: float
    stability: float
    trades: int


@dataclass(frozen=True)
class ComboFilter:
    allowed: dict[str, set[ComboKey]]
    rejection_reasons: dict[str, dict[ComboKey, str]]

    def is_allowed(self, timeframe: str, combo: ComboKey) -> bool:
        return combo in self.allowed.get(timeframe, set())

    def rejection_reason(self, timeframe: str, combo: ComboKey) -> str | None:
        return self.rejection_reasons.get(timeframe, {}).get(combo)


def normalize_timeframe(value: Any) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.endswith("m"):
            return trimmed
        if trimmed.isdigit():
            return f"{trimmed}m"
        return trimmed
    if isinstance(value, (int, float)):
        return f"{int(value)}m"
    return None


def infer_timeframe_minutes(candles: list[Candle]) -> int | None:
    if len(candles) < 2:
        return None
    timestamps = []
    for candle in candles:
        value = candle.timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        timestamps.append(dt)
    deltas = []
    for prev, current in zip(timestamps, timestamps[1:]):
        delta = current - prev
        deltas.append(delta.total_seconds() / 60)
    return int(round(median(deltas))) if deltas else None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_ob_type(record: dict[str, Any]) -> str | None:
    raw = record.get("ob_type")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    tradable = record.get("ob_tradable")
    if isinstance(tradable, str):
        tradable = tradable.strip().lower() in {"true", "1", "yes"}
    if isinstance(tradable, bool):
        return "Tradable" if tradable else "NonTradable"
    return None


def _parse_combo_string(value: str) -> tuple[str, str, str] | None:
    trimmed = value.strip()
    if ":" in trimmed:
        trimmed = trimmed.split(":", 1)[1]
    parts = [part.strip() for part in trimmed.split("|") if part.strip()]
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _normalize_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "rows", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if all(isinstance(value, list) for value in payload.values()):
            flattened = []
            for timeframe, items in payload.items():
                for item in items:
                    if isinstance(item, dict) and "timeframe" not in item:
                        item = {**item, "timeframe": timeframe}
                    flattened.append(item)
            return flattened
    raise ValueError("Expected a JSON array or an object containing a list of records.")


def _to_metrics(record: dict[str, Any]) -> ComboMetrics | None:
    timeframe = normalize_timeframe(record.get("timeframe"))
    phase = record.get("phase") or record.get("mmxm_phase")
    entry_type = record.get("entry_type") or record.get("entry_method")
    ob_type = _normalize_ob_type(record)
    if not (isinstance(phase, str) and isinstance(entry_type, str) and isinstance(ob_type, str)):
        combo_value = record.get("combo") or record.get("key")
        if isinstance(combo_value, str):
            parsed = _parse_combo_string(combo_value)
            if parsed is not None:
                phase, entry_type, ob_type = parsed
    expectancy = _coerce_float(record.get("expectancy"))
    max_drawdown = _coerce_float(record.get("max_drawdown") or record.get("drawdown"))
    stability = _coerce_float(record.get("stability"))
    trades = _coerce_int(record.get("trades"))

    if not all(
        [
            isinstance(timeframe, str),
            isinstance(phase, str),
            isinstance(entry_type, str),
            isinstance(ob_type, str),
            expectancy is not None,
            max_drawdown is not None,
            stability is not None,
            trades is not None,
        ]
    ):
        return None

    return ComboMetrics(
        timeframe=timeframe,
        phase=phase,
        entry_type=entry_type,
        ob_type=ob_type,
        expectancy=expectancy,
        max_drawdown=max_drawdown,
        stability=stability,
        trades=trades,
    )


def evaluate_combo(metrics: ComboMetrics) -> tuple[bool, str | None]:
    if metrics.expectancy <= GLOBAL_EXPECTANCY_MIN:
        return False, "expectancy"
    if metrics.max_drawdown > GLOBAL_MAX_DRAWDOWN:
        return False, "dd"
    thresholds = TIMEFRAME_THRESHOLDS.get(metrics.timeframe)
    if not thresholds:
        return False, "trades"
    if metrics.trades < int(thresholds["trades"]):
        return False, "trades"
    if metrics.stability < float(thresholds["stability"]):
        return False, "stability"
    return True, None


def build_combo_filter(metrics_rows: Iterable[ComboMetrics]) -> ComboFilter:
    allowed: dict[str, set[ComboKey]] = {}
    rejection_reasons: dict[str, dict[ComboKey, str]] = {}

    for metrics in metrics_rows:
        combo = (metrics.phase, metrics.entry_type, metrics.ob_type)
        allowed_combo, reason = evaluate_combo(metrics)
        if allowed_combo:
            allowed.setdefault(metrics.timeframe, set()).add(combo)
        else:
            rejection_reasons.setdefault(metrics.timeframe, {})[combo] = reason or "trades"

    return ComboFilter(allowed=allowed, rejection_reasons=rejection_reasons)


def load_combo_filter(path: Path) -> ComboFilter:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = _normalize_records(payload)
    metrics_rows = [row for record in records if (row := _to_metrics(record)) is not None]
    return build_combo_filter(metrics_rows)


def filter_entry_signals(
    entries: list[EntrySignal],
    timeframe: str,
    combo_filter: ComboFilter,
) -> list[EntrySignal]:
    filtered: list[EntrySignal] = []
    for entry in entries:
        ob_type = "Tradable" if entry.ob_tradable else "NonTradable"
        combo = (entry.mmxm_phase, entry.method, ob_type)
        if combo_filter.is_allowed(timeframe, combo):
            filtered.append(entry)
            continue
        reason = combo_filter.rejection_reason(timeframe, combo) or "trades"
        print(
            "filtered_out: "
            f"{reason} | timeframe={timeframe} | combo={combo[0]}|{combo[1]}|{combo[2]}",
        )
    return filtered
