"""Shared data models for research runs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


@dataclass(frozen=True)
class MmxmPhase:
    index: int
    timestamp: str
    phase: str
    range_high: float
    range_low: float
    sweep: str | None = None


@dataclass(frozen=True)
class OrderBlock:
    ob_id: int
    index: int
    timestamp: str
    direction: str
    open: float
    high: float
    low: float
    close: float
    range: float
    impulse_end: int
    has_imbalance: bool
    has_bos: bool
    near_level: bool
    after_sweep: bool
    tradable: bool


@dataclass(frozen=True)
class Trade:
    entry_time: str
    entry_price: float
    direction: str
    mmxm_phase: str
    entry_method: str
    ob_tradable: bool
    ob_id: int
    exit_time: str | None = None
    exit_price: float | None = None
    pnl_r: float | None = None
    day_label: str | None = None
