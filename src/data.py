"""Data loading utilities for research runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Candle:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


def load_candles_csv(path: str | Path) -> list[Candle]:
    """Load candles from a CSV file."""
    frame = pd.read_csv(path)
    normalized = {column: column.strip().lower() for column in frame.columns}
    columns = list(normalized.values())

    timestamp_col = _find_column(
        columns,
        ["timestamp", "time", "date", "datetime"],
        label="timestamp",
    )
    open_col = _find_column(columns, ["open", "o"], label="open")
    high_col = _find_column(columns, ["high", "h"], label="high")
    low_col = _find_column(columns, ["low", "l"], label="low")
    close_col = _find_column(columns, ["close", "c"], label="close")
    volume_candidates = [name for name in columns if name in {"volume", "vol"}]
    volume_col = volume_candidates[0] if volume_candidates else None

    reverse_lookup = {value: key for key, value in normalized.items()}
    timestamp_values = frame[reverse_lookup[timestamp_col]]
    numeric_values = pd.to_numeric(timestamp_values, errors="coerce")
    if numeric_values.notna().sum() == len(frame):
        parsed_time = pd.to_datetime(numeric_values, unit="s", utc=True)
    else:
        parsed_time = pd.to_datetime(timestamp_values, utc=True, errors="coerce")

    frame = frame.copy()
    frame["__timestamp"] = parsed_time
    before_drop = len(frame)
    frame = frame.dropna(subset=["__timestamp"])
    if before_drop != len(frame):
        print(
            f"Filtered {before_drop - len(frame)} rows with invalid timestamps "
            f"from {path}."
        )

    frame = frame.sort_values("__timestamp").drop_duplicates("__timestamp")
    print(
        "Loaded CSV candles from "
        f"{path} | timestamp={timestamp_col}, open={open_col}, high={high_col}, "
        f"low={low_col}, close={close_col}, volume={volume_col or 'n/a'} | "
        f"rows={len(frame)}"
    )

    candles: list[Candle] = []
    for _, row in frame.iterrows():
        timestamp = row["__timestamp"].isoformat()
        volume = (
            float(row[reverse_lookup[volume_col]])
            if volume_col and pd.notna(row[reverse_lookup[volume_col]])
            else None
        )
        candles.append(
            Candle(
                timestamp=timestamp,
                open=float(row[reverse_lookup[open_col]]),
                high=float(row[reverse_lookup[high_col]]),
                low=float(row[reverse_lookup[low_col]]),
                close=float(row[reverse_lookup[close_col]]),
                volume=volume,
            )
        )
    return candles


def _find_column(columns: list[str], candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Missing required column for "
        f"{label}. Looked for: {candidates}. Available: {columns}"
    )


def load_candles_xlsx(path: str | Path) -> list[Candle]:
    """Load candles from a SAXO-style XLSX file."""
    frame = pd.read_excel(path)
    normalized = {column: column.strip().lower() for column in frame.columns}
    columns = list(normalized.values())

    timestamp_col = _find_column(
        columns,
        ["timestamp", "time", "date", "datetime"],
        label="timestamp",
    )
    open_col = _find_column(columns, ["open", "o"], label="open")
    high_col = _find_column(columns, ["high", "h"], label="high")
    low_col = _find_column(columns, ["low", "l"], label="low")
    close_col = _find_column(columns, ["close", "c"], label="close")
    volume_candidates = [name for name in columns if name in {"volume", "vol"}]
    volume_col = volume_candidates[0] if volume_candidates else None

    reverse_lookup = {value: key for key, value in normalized.items()}
    candles: list[Candle] = []
    print(
        "Loaded XLSX candles from "
        f"{path} | timestamp={timestamp_col}, open={open_col}, high={high_col}, "
        f"low={low_col}, close={close_col}, volume={volume_col or 'n/a'} | "
        f"rows={len(frame)}"
    )
    for _, row in frame.iterrows():
        timestamp_value = row[reverse_lookup[timestamp_col]]
        timestamp = (
            timestamp_value.isoformat()
            if hasattr(timestamp_value, "isoformat")
            else str(timestamp_value)
        )
        volume = (
            float(row[reverse_lookup[volume_col]])
            if volume_col and pd.notna(row[reverse_lookup[volume_col]])
            else None
        )
        candles.append(
            Candle(
                timestamp=timestamp,
                open=float(row[reverse_lookup[open_col]]),
                high=float(row[reverse_lookup[high_col]]),
                low=float(row[reverse_lookup[low_col]]),
                close=float(row[reverse_lookup[close_col]]),
                volume=volume,
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
