"""Filter strategy combinations for statistically profitable ICT/MMXM setups."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


ALLOWED_INSTRUMENTS = {"EURUSD", "SPX500"}
ALLOWED_TIMEFRAMES = {"15m", "30m", "60m"}
ALLOWED_PHASES = {"Accumulation", "Distribution"}
ALLOWED_ENTRY_TYPES = {"Risk Entry", "Continuation"}


@dataclass(frozen=True)
class StrategyRow:
    instrument: str
    timeframe: str
    phase: str
    entry_type: str
    expectancy: float
    max_drawdown: float
    stability: float
    trades: int


def _load_payload(path: Path | None) -> Any:
    if path is None:
        raw = sys.stdin.read()
        if not raw.strip():
            raise SystemExit("No input provided on stdin.")
    else:
        raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def _normalize_records(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "items", "rows", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise SystemExit("Expected a JSON array or an object containing a list of records.")


def _normalize_timeframe(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("m"):
            return value
        if value.isdigit():
            return f"{value}m"
        return value
    if isinstance(value, (int, float)):
        return f"{int(value)}m"
    return None


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


def _to_row(record: dict[str, Any]) -> StrategyRow | None:
    instrument = record.get("instrument")
    timeframe = _normalize_timeframe(record.get("timeframe"))
    phase = record.get("phase")
    entry_type = record.get("entry_type")
    expectancy = _coerce_float(record.get("expectancy"))
    max_drawdown = _coerce_float(record.get("max_drawdown"))
    stability = _coerce_float(record.get("stability"))
    trades = _coerce_int(record.get("trades"))

    if not all(
        [
            isinstance(instrument, str),
            isinstance(timeframe, str),
            isinstance(phase, str),
            isinstance(entry_type, str),
            expectancy is not None,
            max_drawdown is not None,
            stability is not None,
            trades is not None,
        ]
    ):
        return None

    return StrategyRow(
        instrument=instrument,
        timeframe=timeframe,
        phase=phase,
        entry_type=entry_type,
        expectancy=expectancy,
        max_drawdown=max_drawdown,
        stability=stability,
        trades=trades,
    )


def _is_allowed(row: StrategyRow) -> bool:
    if row.instrument not in ALLOWED_INSTRUMENTS:
        return False
    if row.timeframe not in ALLOWED_TIMEFRAMES:
        return False
    if row.phase not in ALLOWED_PHASES:
        return False
    if row.entry_type not in ALLOWED_ENTRY_TYPES:
        return False
    if row.expectancy <= 0:
        return False
    if row.max_drawdown > 3.0:
        return False
    if row.stability < 0.40:
        return False
    if row.trades < 20:
        return False
    return True


def _group_and_sort(rows: Iterable[StrategyRow]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[StrategyRow]] = {}
    for row in rows:
        grouped.setdefault(row.timeframe, []).append(row)

    output: dict[str, list[dict[str, Any]]] = {}
    for timeframe, items in grouped.items():
        sorted_rows = sorted(items, key=lambda item: item.expectancy, reverse=True)
        output[timeframe] = [asdict(row) for row in sorted_rows]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter strategy combinations for profitable ICT/MMXM setups."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a JSON file. Reads stdin when omitted.",
    )
    args = parser.parse_args()

    payload = _load_payload(args.input)
    records = _normalize_records(payload)
    rows = [row for record in records if (row := _to_row(record)) is not None]
    filtered = [row for row in rows if _is_allowed(row)]
    output = _group_and_sort(filtered)
    json.dump(output, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
