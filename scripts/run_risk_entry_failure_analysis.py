"""Run structured Risk Entry failure analysis for MMXM research questions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Callable, Iterable

from zoneinfo import ZoneInfo

from src.backtest import run_backtest
from src.data import Candle, load_candles_csv, load_candles_xlsx
from src.day_context import DayContextMetrics, label_days
from src.mmxm import MmxmPhase
from src.models import Trade


@dataclass(frozen=True)
class RiskEntryRecord:
    instrument: str
    entry_time: str
    pnl_r: float
    day_label: str
    mmxm_phase: str
    sweep_subcontext: str
    trend_relation: str
    trend_timing: str
    sweep_profile: str
    timing_bucket: str
    context_bucket: str
    prior_manipulation: str
    volatility_regime: str


@dataclass(frozen=True)
class SummaryRow:
    key: str
    trades: int
    winrate: float
    expectancy: float
    max_drawdown: float
    stability: float


@dataclass(frozen=True)
class DaySummary:
    label: str
    range_atr: float
    sweep_high_count: int
    sweep_low_count: int
    day_open: float
    day_close: float
    day_high: float
    day_low: float
    day_range: float
    direction: str
    indices: list[int]
    trend_timing: str


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
    for token in ["SPX500", "EURUSD"]:
        if token in stem:
            return token
    return stem


def _to_local_timestamp(timestamp: str, tz_name: str) -> datetime:
    value = timestamp.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo(tz_name))


def _build_day_summaries(
    candles: list[Candle],
    day_context: dict[str, DayContextMetrics],
    tz_name: str,
) -> dict[str, DaySummary]:
    local_times = [_to_local_timestamp(c.timestamp, tz_name) for c in candles]
    day_indices: dict[str, list[int]] = defaultdict(list)
    for idx, local_dt in enumerate(local_times):
        day_indices[local_dt.date().isoformat()].append(idx)

    summaries: dict[str, DaySummary] = {}
    for day_key, indices in day_indices.items():
        day_open = candles[indices[0]].open
        day_close = candles[indices[-1]].close
        day_high = max(candles[idx].high for idx in indices)
        day_low = min(candles[idx].low for idx in indices)
        day_range = day_high - day_low
        direction = "bullish" if day_close >= day_open else "bearish"

        timing_threshold = 0.7
        trend_timing = "UNKNOWN"
        if day_range > 0:
            target_found_index = None
            for idx in indices:
                candle = candles[idx]
                progress = abs(candle.close - day_open) / day_range
                if direction == "bullish" and candle.close >= day_open and progress >= timing_threshold:
                    target_found_index = idx
                    break
                if direction == "bearish" and candle.close <= day_open and progress >= timing_threshold:
                    target_found_index = idx
                    break
            if target_found_index is not None:
                midpoint = indices[len(indices) // 2]
                trend_timing = "EARLY" if target_found_index <= midpoint else "LATE"

        context_metrics = day_context.get(day_key)
        label = context_metrics.label if context_metrics else "UNKNOWN"
        range_atr = context_metrics.range_atr if context_metrics else 0.0
        sweep_high_count = context_metrics.sweep_high_count if context_metrics else 0
        sweep_low_count = context_metrics.sweep_low_count if context_metrics else 0

        summaries[day_key] = DaySummary(
            label=label,
            range_atr=range_atr,
            sweep_high_count=sweep_high_count,
            sweep_low_count=sweep_low_count,
            day_open=day_open,
            day_close=day_close,
            day_high=day_high,
            day_low=day_low,
            day_range=day_range,
            direction=direction,
            indices=indices,
            trend_timing=trend_timing,
        )

    return summaries


def _sweep_profile(summary: DaySummary) -> str:
    if summary.sweep_high_count >= 1 and summary.sweep_low_count >= 1:
        return "DOUBLE"
    if summary.sweep_high_count >= 1:
        return "SINGLE_HIGH"
    if summary.sweep_low_count >= 1:
        return "SINGLE_LOW"
    return "NONE"


def _timing_bucket(progress: float) -> str:
    if progress < 0.3:
        return "EARLY"
    if progress < 0.7:
        return "MID"
    return "LATE"


def _context_bucket(progress: float) -> str:
    return "DECIDED" if progress >= 0.8 else "OPEN"


def _volatility_regime(range_atr: float) -> str:
    if range_atr < 0.8:
        return "LOW"
    if range_atr <= 1.2:
        return "MID"
    return "HIGH"


def _trend_relation(direction: str, day_direction: str, progress: float) -> str:
    if direction == day_direction:
        return "PRE_TREND" if progress < 0.3 else "TREND"
    return "ANTI_TREND"


def _entry_index_map(candles: list[Candle]) -> dict[str, int]:
    return {candle.timestamp: idx for idx, candle in enumerate(candles)}


def _progress_at_entry(entry_index: int, day_summary: DaySummary, candles: list[Candle]) -> float:
    if day_summary.day_range <= 0:
        return 0.0
    candle = candles[entry_index]
    return abs(candle.close - day_summary.day_open) / day_summary.day_range


def _prior_manipulation(entry_index: int, day_summary: DaySummary, phases: list[MmxmPhase]) -> str:
    for idx in day_summary.indices:
        if idx >= entry_index:
            break
        if phases[idx].phase == "Manipulation":
            return "YES"
    return "NO"


def _summarize_records(
    records: Iterable[RiskEntryRecord],
    key_fn: Callable[[RiskEntryRecord], str],
) -> list[SummaryRow]:
    grouped: dict[str, list[RiskEntryRecord]] = defaultdict(list)
    for record in records:
        grouped[key_fn(record)].append(record)

    summaries: list[SummaryRow] = []
    for key, group in grouped.items():
        pnls = [record.pnl_r for record in group]
        wins = sum(1 for pnl in pnls if pnl > 0)
        winrate = wins / len(group) if group else 0.0
        expectancy = sum(pnls) / len(group) if group else 0.0
        equity_curve = []
        equity = 0.0
        for record in sorted(group, key=lambda r: r.entry_time):
            equity += record.pnl_r
            equity_curve.append(equity)
        peak = 0.0
        max_dd = 0.0
        for value in equity_curve:
            peak = max(peak, value)
            max_dd = max(max_dd, peak - value)
        if len(pnls) < 2:
            stability = 0.0
        else:
            mean = sum(pnls) / len(pnls)
            variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
            stability = 1.0 / (1.0 + variance**0.5)
        summaries.append(
            SummaryRow(
                key=key,
                trades=len(group),
                winrate=winrate,
                expectancy=expectancy,
                max_drawdown=max_dd,
                stability=stability,
            )
        )
    return sorted(summaries, key=lambda row: row.key)


def _write_summary(rows: list[SummaryRow], path: Path) -> None:
    lines = ["key,trades,winrate,expectancy,max_drawdown,stability"]
    for row in rows:
        lines.append(
            f"{row.key},{row.trades},{row.winrate:.4f},{row.expectancy:.4f},"
            f"{row.max_drawdown:.4f},{row.stability:.4f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _question_outputs(records: list[RiskEntryRecord], runs_dir: Path, suffix: str) -> None:
    questions: dict[str, tuple[str, Callable[[RiskEntryRecord], str], Callable[[RiskEntryRecord], bool]]] = {
        "q1_day_context": (
            "Q1 Day context failures by day label × phase",
            lambda r: f"{r.day_label}|{r.mmxm_phase}",
            lambda r: True,
        ),
        "q2_trend_relation": (
            "Q2 Trend relation (anti/pre) by day label × phase",
            lambda r: f"{r.trend_relation}|{r.day_label}|{r.mmxm_phase}",
            lambda r: True,
        ),
        "q3_instrument_trend": (
            "Q3 Instrument trend relation by day label × phase",
            lambda r: f"{r.instrument}|{r.trend_relation}|{r.day_label}|{r.mmxm_phase}",
            lambda r: True,
        ),
        "q4_trend_timing": (
            "Q4 TREND_DAY timing by day label × phase",
            lambda r: f"{r.trend_timing}|{r.day_label}|{r.mmxm_phase}",
            lambda r: r.day_label == "TREND_DAY",
        ),
        "q5_sweep_profile": (
            "Q5 Sweep profile by day label × phase",
            lambda r: f"{r.sweep_profile}|{r.day_label}|{r.mmxm_phase}",
            lambda r: True,
        ),
        "q6_timing_context": (
            "Q6 Timing/context buckets by day label × phase",
            lambda r: f"{r.timing_bucket}|{r.context_bucket}|{r.day_label}|{r.mmxm_phase}",
            lambda r: True,
        ),
        "q7_manipulation_subcontext": (
            "Q7 Manipulation subcontext by day label × phase",
            lambda r: f"{r.sweep_subcontext}|{r.day_label}|{r.mmxm_phase}",
            lambda r: r.mmxm_phase == "Manipulation",
        ),
        "q8_distribution_prior": (
            "Q8 Distribution with prior manipulation by day label × phase",
            lambda r: f"{r.prior_manipulation}|{r.day_label}|{r.mmxm_phase}",
            lambda r: r.mmxm_phase == "Distribution",
        ),
        "q9_unknown_range": (
            "Q9 UNKNOWN/RANGE by day label × phase",
            lambda r: f"{r.day_label}|{r.mmxm_phase}",
            lambda r: r.day_label in {"UNKNOWN", "RANGE_DAY"},
        ),
        "q10_instrument_context": (
            "Q10 Instrument context by day label × phase",
            lambda r: f"{r.instrument}|{r.day_label}|{r.mmxm_phase}",
            lambda r: True,
        ),
        "q11_volatility_regime": (
            "Q11 Volatility regime by day label × phase",
            lambda r: f"{r.instrument}|{r.volatility_regime}|{r.day_label}|{r.mmxm_phase}",
            lambda r: True,
        ),
        "q12_failure_filters": (
            "Q12 Failure filter combos by day label × phase",
            lambda r: f"{r.sweep_profile}|{r.timing_bucket}|{r.day_label}|{r.mmxm_phase}",
            lambda r: True,
        ),
        "q13_exclusion_context": (
            "Q13 Context exclusion review by day label × phase",
            lambda r: f"{r.trend_relation}|{r.sweep_profile}|{r.day_label}|{r.mmxm_phase}",
            lambda r: True,
        ),
    }

    for key, (title, key_fn, predicate) in questions.items():
        filtered = [record for record in records if predicate(record)]
        rows = _summarize_records(filtered, key_fn)
        output_path = runs_dir / f"risk_entry_{key}_{suffix}.csv"
        _write_summary(rows, output_path)
        print(f"{title}: {output_path}")


def _records_for_instrument(
    instrument: str,
    trades: list[Trade],
    candles: list[Candle],
    phases: list[MmxmPhase],
    tz_name: str,
) -> list[RiskEntryRecord]:
    day_context = label_days(candles, tz_name=tz_name)
    day_summaries = _build_day_summaries(candles, day_context, tz_name)
    index_map = _entry_index_map(candles)

    records: list[RiskEntryRecord] = []
    for trade in trades:
        if trade.entry_method != "Risk Entry":
            continue
        if trade.ob_tradable:
            continue
        entry_index = index_map.get(trade.entry_time)
        if entry_index is None:
            continue
        day_key = _to_local_timestamp(trade.entry_time, tz_name).date().isoformat()
        summary = day_summaries.get(day_key)
        if summary is None:
            continue
        progress = _progress_at_entry(entry_index, summary, candles)
        sweep_profile = _sweep_profile(summary)
        timing_bucket = _timing_bucket(progress)
        context_bucket = _context_bucket(progress)
        trend_relation = _trend_relation(trade.direction, summary.direction, progress)
        sweep_subcontext = phases[entry_index].sweep or "NONE"
        prior_manipulation = _prior_manipulation(entry_index, summary, phases)
        volatility_regime = _volatility_regime(summary.range_atr)

        records.append(
            RiskEntryRecord(
                instrument=instrument,
                entry_time=trade.entry_time,
                pnl_r=trade.pnl_r or 0.0,
                day_label=summary.label,
                mmxm_phase=trade.mmxm_phase,
                sweep_subcontext=sweep_subcontext,
                trend_relation=trend_relation,
                trend_timing=summary.trend_timing,
                sweep_profile=sweep_profile,
                timing_bucket=timing_bucket,
                context_bucket=context_bucket,
                prior_manipulation=prior_manipulation,
                volatility_regime=volatility_regime,
            )
        )

    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Risk Entry failure analysis for MMXM research questions."
    )
    parser.add_argument(
        "--data",
        required=True,
        nargs="+",
        type=Path,
        help="CSV/XLSX data files (single timeframe).",
    )
    parser.add_argument(
        "--timeframe",
        required=True,
        type=int,
        choices=[15, 60],
        help="Dataset timeframe in minutes (15 or 60).",
    )
    parser.add_argument(
        "--runs-dir",
        default=Path("runs"),
        type=Path,
        help="Output directory for summary tables.",
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
    runs_dir.mkdir(exist_ok=True)

    all_records: list[RiskEntryRecord] = []
    for path in args.data:
        if not path.exists():
            raise FileNotFoundError(path)
        candles = _load_candles(path)
        inferred = _infer_timeframe_minutes(candles)
        if inferred is None:
            raise ValueError(f"Unable to infer timeframe for {path}")
        if inferred != args.timeframe:
            raise ValueError(
                f"Timeframe mismatch for {path}: expected {args.timeframe}m, got {inferred}m"
            )
        instrument = _instrument_from_path(path)
        result = run_backtest(candles)
        records = _records_for_instrument(
            instrument,
            result.trades,
            candles,
            result.phases,
            tz_name=args.tz,
        )
        all_records.extend(records)

    if not all_records:
        raise SystemExit("No Risk Entry NonTradable trades found.")

    suffix = f"{args.timeframe}m"
    _question_outputs(all_records, runs_dir, suffix)


if __name__ == "__main__":
    main()
