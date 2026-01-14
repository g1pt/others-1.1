import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

from app.config import DATA_DIR
from app.models import Candle


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _candidate_paths(symbol: str, timeframe: str) -> list[Path]:
    filename = f"{symbol}_{timeframe}.csv"
    return [
        DATA_DIR / filename,
        DATA_DIR / f"{symbol}-{timeframe}.csv",
        DATA_DIR / f"{symbol}{timeframe}.csv",
    ]


def load_candles(symbol: str, timeframe: str) -> list[Candle]:
    for path in _candidate_paths(symbol, timeframe):
        if path.exists():
            return list(_load_csv(path))
    return []


def _load_csv(path: Path) -> Iterable[Candle]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = _parse_timestamp(row.get("timestamp") or row.get("time") or "")
            if timestamp is None:
                continue
            try:
                candle = Candle(
                    timestamp=timestamp,
                    open=float(row.get("open", 0)),
                    high=float(row.get("high", 0)),
                    low=float(row.get("low", 0)),
                    close=float(row.get("close", 0)),
                )
            except ValueError:
                continue
            yield candle
