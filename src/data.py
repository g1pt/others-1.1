"""Data loading utilities for research runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


def load_candles_csv(path: str | Path) -> list[Candle]:
    """Load candles from a CSV file.

    Expected columns: timestamp, open, high, low, close, volume(optional).
    """
    candles: list[Candle] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        for line in handle:
            parts = line.strip().split(",")
            if not parts or len(parts) < 5:
                continue
            row = dict(zip(header, parts))
            candles.append(
                Candle(
                    timestamp=row["timestamp"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]) if "volume" in row and row["volume"] else None,
                )
            )
    return candles


def iter_timeframe(candles: Iterable[Candle], timeframe_minutes: int) -> list[Candle]:
    """Placeholder for timeframe aggregation."""
    _ = timeframe_minutes
    return list(candles)


def generate_synthetic_candles(length: int = 240) -> list[Candle]:
    """Generate deterministic synthetic candles for reproducible research runs."""
    candles: list[Candle] = []
    price = 100.0
    for idx in range(length):
        phase = idx % 60
        if phase < 20:
            drift = 0.02
        elif phase < 30:
            drift = -0.15
        elif phase < 50:
            drift = 0.25
        else:
            drift = -0.05
        open_price = price
        close = price + drift
        high = max(open_price, close) + 0.2
        low = min(open_price, close) - 0.2
        candles.append(
            Candle(
                timestamp=f"2023-01-01T{idx:02d}:00:00Z",
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=1000.0 + idx,
            )
        )
        price = close
    return candles
