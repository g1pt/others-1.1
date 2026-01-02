"""Shared data models for research runs."""
from __future__ import annotations

from dataclasses import dataclass


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
