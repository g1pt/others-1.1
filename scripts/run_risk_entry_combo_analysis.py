"""Analyze Risk Entry combinations across EURUSD/SPX500 datasets."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.backtest import run_backtest  # noqa: E402
from src.data import Candle, load_candles_csv, load_candles_xlsx  # noqa: E402
from src.day_context import label_days, label_for_timestamp  # noqa: E402


@dataclass(frozen=True)
class ComboRecord:
    instrument: str
    timeframe: int
    entry_time: str
    mmxm_phase: str
    day_label: str
    pnl_r: float


@dataclass(frozen=True)
class ComboSummary:
    instrument: str
    timeframe: int
    mmxm_phase: str
    day_label: str
    trades: int
    winrate: float
    expectancy: float
    max_drawdown: float
    stability: float


@dataclass(frozen=True)
class ComboDecision:
    combo: str
    instrument: str
    timeframe: int
    mmxm_phase: str
    day_label: str
    trades: int
    winrate: float
    expectancy: float
    max_drawdown: float
    stability: float
    status: str


def _data_roots() -> list[Path]:
    roots = [Path("/data"), Path("data")]
    seen = set()
    result = []
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        if resolved.exists():
            result.append(resolved)
            seen.add(resolved)
    return result


def _find_data_files() -> list[Path]:
    files: list[Path] = []
    for root in _data_roots():
        files.extend(sorted(root.glob("*.csv")))
        files.extend(sorted(root.glob("*.xlsx")))
    return files


def _load_candles(path: Path) -> list[Candle]:
    if path.suffix.lower() == ".csv":
        return load_candles_csv(path)
    if path.suffix.lower() == ".xlsx":
        return load_candles_xlsx(path)
    raise ValueError(f"Unsupported file type: {path}")


def _infer_timeframe_minutes(candles: list[Candle]) -> int | None:
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


def _instrument_from_path(path: Path) -> str:
    stem = path.stem.upper()
    for token in ["EURUSD", "SPX500"]:
        if token in stem:
            return token
    return stem


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)
    return max_dd


def _stability(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls)
    variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    return 1.0 / (1.0 + variance**0.5)


def _summarize(records: Iterable[ComboRecord]) -> list[ComboSummary]:
    grouped: dict[tuple[str, int, str, str], list[ComboRecord]] = {}
    for record in records:
        key = (record.instrument, record.timeframe, record.mmxm_phase, record.day_label)
        grouped.setdefault(key, []).append(record)

    summaries: list[ComboSummary] = []
    for (instrument, timeframe, mmxm_phase, day_label), group in grouped.items():
        pnls = [record.pnl_r for record in group]
        wins = sum(1 for pnl in pnls if pnl > 0)
        winrate = wins / len(group) if group else 0.0
        expectancy = sum(pnls) / len(group) if group else 0.0
        equity = 0.0
        equity_curve = []
        for record in sorted(group, key=lambda r: r.entry_time):
            equity += record.pnl_r
            equity_curve.append(equity)
        max_dd = _max_drawdown(equity_curve)
        stability = _stability(pnls)
        summaries.append(
            ComboSummary(
                instrument=instrument,
                timeframe=timeframe,
                mmxm_phase=mmxm_phase,
                day_label=day_label,
                trades=len(group),
                winrate=winrate,
                expectancy=expectancy,
                max_drawdown=max_dd,
                stability=stability,
            )
        )
    return sorted(
        summaries,
        key=lambda row: (
            row.instrument,
            row.timeframe,
            row.mmxm_phase,
            row.day_label,
        ),
    )


def _write_summary(rows: list[ComboSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "instrument",
                "timeframe",
                "mmxm_phase",
                "day_label",
                "trades",
                "winrate",
                "expectancy",
                "max_drawdown",
                "stability",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.instrument,
                    row.timeframe,
                    row.mmxm_phase,
                    row.day_label,
                    row.trades,
                    f"{row.winrate:.4f}",
                    f"{row.expectancy:.4f}",
                    f"{row.max_drawdown:.4f}",
                    f"{row.stability:.4f}",
                ]
            )


def _write_decisions_csv(rows: list[ComboDecision], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "combo",
                "instrument",
                "timeframe",
                "mmxm_phase",
                "day_label",
                "trades",
                "winrate",
                "expectancy",
                "max_drawdown",
                "stability",
                "status",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.combo,
                    row.instrument,
                    row.timeframe,
                    row.mmxm_phase,
                    row.day_label,
                    row.trades,
                    f"{row.winrate:.4f}",
                    f"{row.expectancy:.4f}",
                    f"{row.max_drawdown:.4f}",
                    f"{row.stability:.4f}",
                    row.status,
                ]
            )


def _write_decisions_json(rows: list[ComboDecision], path: Path) -> None:
    payload = [
        {
            "combo": row.combo,
            "trades": row.trades,
            "expectancy": row.expectancy,
            "drawdown": row.max_drawdown,
            "stability": row.stability,
            "status": row.status,
        }
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_common_summary(rows: list[ComboSummary], path: Path) -> None:
    counts = {
        "phase": {},
        "instrument": {},
        "timeframe": {},
    }
    for row in rows:
        counts["phase"][row.mmxm_phase] = counts["phase"].get(row.mmxm_phase, 0) + 1
        counts["instrument"][row.instrument] = (
            counts["instrument"].get(row.instrument, 0) + 1
        )
        counts["timeframe"][row.timeframe] = counts["timeframe"].get(row.timeframe, 0) + 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dimension", "value", "count"])
        for dimension in ["phase", "instrument", "timeframe"]:
            for value, count in sorted(counts[dimension].items()):
                writer.writerow([dimension, value, count])


def _write_status_summary(rows: list[ComboDecision], path: Path) -> None:
    counts: dict[str, int] = {"Toegestaan": 0, "Voorzichtig": 0, "Afgekeurd": 0}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1

    allowed_expectancies = [row.expectancy for row in rows if row.status == "Toegestaan"]
    avg_expectancy = (
        sum(allowed_expectancies) / len(allowed_expectancies)
        if allowed_expectancies
        else None
    )

    highest_by_instrument: dict[str, dict[str, object]] = {}
    for row in rows:
        current = highest_by_instrument.get(row.instrument)
        if current is None or row.stability > float(current["stability"]):
            highest_by_instrument[row.instrument] = {
                "combo": row.combo,
                "stability": row.stability,
                "expectancy": row.expectancy,
                "trades": row.trades,
                "status": row.status,
            }

    summary = {
        "status_counts": counts,
        "avg_expectancy_toegestaan": avg_expectancy,
        "highest_stability_by_instrument": highest_by_instrument,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _status_for(row: ComboSummary) -> str:
    if row.max_drawdown < 10 and row.stability > 0.4:
        return "Toegestaan"
    if row.max_drawdown < 5 and row.stability > 0.39:
        return "Voorzichtig"
    return "Afgekeurd"


def _combo_label(row: ComboSummary) -> str:
    return f"{row.mmxm_phase} | Risk Entry | {row.instrument} | {row.timeframe}m"


def _plot_expectancy_drawdown(rows: list[ComboSummary], path: Path) -> None:
    if not rows:
        return
    expectancies = [row.expectancy for row in rows]
    drawdowns = [row.max_drawdown for row in rows]
    stability = [row.stability for row in rows]
    sizes = [max(30.0, row.trades * 2.0) for row in rows]

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        drawdowns,
        expectancies,
        c=stability,
        cmap="viridis",
        s=sizes,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.4,
    )
    plt.colorbar(scatter, label="Stabiliteit")
    plt.xlabel("Max drawdown (R)")
    plt.ylabel("Expectancy (R)")
    plt.title("Risk Entry combinaties: expectancy vs drawdown")
    plt.grid(True, linestyle="--", alpha=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Risk Entry combinations for EURUSD/SPX500 15m/60m datasets."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        type=Path,
        help="Optional CSV/XLSX data files. Defaults to /data and ./data.",
    )
    parser.add_argument(
        "--runs-dir",
        default=Path("runs"),
        type=Path,
        help="Output directory for summary tables and charts.",
    )
    parser.add_argument(
        "--tz",
        default="Europe/Amsterdam",
        help="Timezone for day labeling.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs_dir: Path = args.runs_dir

    files = args.data if args.data else _find_data_files()
    if not files:
        raise SystemExit("No CSV/XLSX files found in /data or ./data")

    records: list[ComboRecord] = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        instrument = _instrument_from_path(path)
        if instrument not in {"EURUSD", "SPX500"}:
            continue
        candles = _load_candles(path)
        timeframe = _infer_timeframe_minutes(candles)
        if timeframe not in {15, 60}:
            continue
        result = run_backtest(candles)
        day_context = label_days(candles, tz_name=args.tz)

        for trade in result.trades:
            if trade.entry_method != "Risk Entry":
                continue
            day_label = label_for_timestamp(trade.entry_time, day_context, tz_name=args.tz)
            records.append(
                ComboRecord(
                    instrument=instrument,
                    timeframe=timeframe,
                    entry_time=trade.entry_time,
                    mmxm_phase=trade.mmxm_phase,
                    day_label=day_label,
                    pnl_r=trade.pnl_r or 0.0,
                )
            )

    if not records:
        raise SystemExit("No Risk Entry records found for EURUSD/SPX500 15m/60m datasets.")

    all_summaries = _summarize(records)
    _write_summary(all_summaries, runs_dir / "risk_entry_combos_all.csv")

    filtered = [row for row in all_summaries if row.expectancy > 0 and row.trades >= 20]
    _write_summary(filtered, runs_dir / "risk_entry_combos_filtered.csv")

    decisions = [
        ComboDecision(
            combo=_combo_label(row),
            instrument=row.instrument,
            timeframe=row.timeframe,
            mmxm_phase=row.mmxm_phase,
            day_label=row.day_label,
            trades=row.trades,
            winrate=row.winrate,
            expectancy=row.expectancy,
            max_drawdown=row.max_drawdown,
            stability=row.stability,
            status=_status_for(row),
        )
        for row in filtered
    ]
    _write_decisions_csv(decisions, runs_dir / "risk_entry_combos_filtered_status.csv")
    _write_decisions_json(decisions, runs_dir / "risk_entry_combos_filtered_status.json")
    _write_status_summary(decisions, runs_dir / "risk_entry_combos_status_summary.json")
    allowed = [row for row in filtered if _status_for(row) == "Toegestaan"]
    _write_summary(allowed, runs_dir / "risk_entry_combos_toegestaan.csv")
    _write_common_summary(allowed, runs_dir / "risk_entry_combo_summary.csv")
    _plot_expectancy_drawdown(
        allowed,
        runs_dir / "risk_entry_expectancy_drawdown.png",
    )

    print(f"All combos: {len(all_summaries)}")
    print(f"Filtered (expectancy>0 & trades>=20): {len(filtered)}")
    print(f"Allowed (drawdown<10 & stability>0.4): {len(allowed)}")
    print(f"Outputs written to {runs_dir}")


if __name__ == "__main__":
    main()
