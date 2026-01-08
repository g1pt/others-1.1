"""Day context labeling for SPX500 datasets."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from statistics import mean
from zoneinfo import ZoneInfo

from src.config import (
    ATR_PERIOD,
    CHAOS_MIN_OPEN_REVISITS,
    CHAOS_REQUIRE_BOTH_SWEEPS,
    LC_TIME_LOCAL,
    NYO_TIME_LOCAL,
    OPEN_REVISIT_TOL_ATR,
    RANGE_MAX_RANGE_ATR,
    REV_WINDOW_MIN,
    TREND_MIN_RANGE_ATR,
)
from src.data import Candle

SWEEP_LOOKBACK = 3
CHAOS_MAX_RANGE_ATR = 1.2


@dataclass(frozen=True)
class DayContextMetrics:
    date: str
    label: str
    range_atr: float
    close_to_open_atr: float
    open_revisit_count: int
    sweep_high_count: int
    sweep_low_count: int


def label_days(
    candles: list[Candle], tz_name: str = "Europe/Amsterdam"
) -> dict[str, DayContextMetrics]:
    """Compute day context labels and metrics for a candle series."""
    if not candles:
        return {}

    local_times = [_to_local_timestamp(c.timestamp, tz_name) for c in candles]
    atr_values = _atr_series(candles, ATR_PERIOD)
    sweep_high_flags, sweep_low_flags = _sweep_flags(candles, SWEEP_LOOKBACK)

    day_indices: dict[str, list[int]] = {}
    for idx, local_dt in enumerate(local_times):
        day_key = local_dt.date().isoformat()
        day_indices.setdefault(day_key, []).append(idx)

    results: dict[str, DayContextMetrics] = {}
    for day, indices in day_indices.items():
        day_open = candles[indices[0]].open
        day_high = max(candles[idx].high for idx in indices)
        day_low = min(candles[idx].low for idx in indices)
        day_close = candles[indices[-1]].close
        day_range = day_high - day_low

        atr_samples = [atr_values[idx] for idx in indices if atr_values[idx] > 0]
        atr_value = mean(atr_samples) if atr_samples else 0.0
        range_atr = day_range / atr_value if atr_value else 0.0
        close_to_open_atr = (
            abs(day_close - day_open) / atr_value if atr_value else 0.0
        )

        open_revisit_count = 0
        if atr_value:
            revisit_tol = OPEN_REVISIT_TOL_ATR * atr_value
            for idx in indices[2:]:
                if abs(candles[idx].close - day_open) <= revisit_tol:
                    open_revisit_count += 1

        sweep_high_count = sum(1 for idx in indices if sweep_high_flags[idx])
        sweep_low_count = sum(1 for idx in indices if sweep_low_flags[idx])

        is_reversal = _is_reversal_day(
            indices,
            local_times,
            candles,
            day_open,
            day_close,
            tz_name,
        )
        is_chaos = _is_chaos_day(
            range_atr,
            open_revisit_count,
            sweep_high_count,
            sweep_low_count,
        )
        is_trend = _is_trend_day(
            range_atr,
            open_revisit_count,
            day_close,
            day_low,
            day_range,
        )
        is_range = _is_range_day(
            range_atr,
            open_revisit_count,
            close_to_open_atr,
        )

        if is_reversal:
            label = "REVERSAL_DAY"
        elif is_chaos:
            label = "CHAOS_DAY"
        elif is_trend:
            label = "TREND_DAY"
        elif is_range:
            label = "RANGE_DAY"
        else:
            label = "UNKNOWN"

        results[day] = DayContextMetrics(
            date=day,
            label=label,
            range_atr=range_atr,
            close_to_open_atr=close_to_open_atr,
            open_revisit_count=open_revisit_count,
            sweep_high_count=sweep_high_count,
            sweep_low_count=sweep_low_count,
        )

    return results


def label_for_timestamp(
    timestamp: str, day_context: dict[str, DayContextMetrics], tz_name: str = "Europe/Amsterdam"
) -> str:
    """Return the day label for a timestamp."""
    day_key = _to_local_timestamp(timestamp, tz_name).date().isoformat()
    metrics = day_context.get(day_key)
    return metrics.label if metrics else "UNKNOWN"


def _to_local_timestamp(timestamp: str, tz_name: str) -> datetime:
    value = timestamp.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo(tz_name))


def _atr_series(candles: list[Candle], period: int) -> list[float]:
    trs: list[float] = []
    atrs: list[float] = []
    for idx, candle in enumerate(candles):
        if idx == 0:
            tr = candle.high - candle.low
        else:
            prev_close = candles[idx - 1].close
            tr = max(
                candle.high - candle.low,
                abs(candle.high - prev_close),
                abs(candle.low - prev_close),
            )
        trs.append(tr)
        window = trs[max(0, idx - period + 1) : idx + 1]
        atrs.append(sum(window) / len(window))
    return atrs


def _sweep_flags(
    candles: list[Candle], lookback: int
) -> tuple[list[bool], list[bool]]:
    sweep_high_flags = [False] * len(candles)
    sweep_low_flags = [False] * len(candles)
    for idx in range(lookback, len(candles)):
        window = candles[idx - lookback : idx]
        prev_high = max(c.high for c in window)
        prev_low = min(c.low for c in window)
        candle = candles[idx]
        if candle.high > prev_high and candle.close < candle.open:
            sweep_high_flags[idx] = True
        if candle.low < prev_low and candle.close > candle.open:
            sweep_low_flags[idx] = True
    return sweep_high_flags, sweep_low_flags


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def _is_reversal_day(
    indices: list[int],
    local_times: list[datetime],
    candles: list[Candle],
    day_open: float,
    day_close: float,
    tz_name: str,
) -> bool:
    local_tz = ZoneInfo(tz_name)
    day_date = local_times[indices[0]].date()
    anchors = (_parse_time(NYO_TIME_LOCAL), _parse_time(LC_TIME_LOCAL))
    for anchor in anchors:
        anchor_dt = datetime.combine(day_date, anchor, tzinfo=local_tz)
        window_start = anchor_dt - timedelta(minutes=REV_WINDOW_MIN)
        window_end = anchor_dt + timedelta(minutes=REV_WINDOW_MIN)

        pre_indices = [idx for idx in indices if local_times[idx] < window_start]
        post_indices = [idx for idx in indices if local_times[idx] > window_end]
        if not pre_indices or not post_indices:
            continue

        pre_high = max(candles[idx].high for idx in pre_indices)
        pre_low = min(candles[idx].low for idx in pre_indices)
        post_high = max(candles[idx].high for idx in post_indices)
        post_low = min(candles[idx].low for idx in post_indices)

        if post_high > pre_high and day_close < day_open:
            return True
        if post_low < pre_low and day_close > day_open:
            return True
    return False


def _is_chaos_day(
    range_atr: float,
    open_revisit_count: int,
    sweep_high_count: int,
    sweep_low_count: int,
) -> bool:
    if range_atr > CHAOS_MAX_RANGE_ATR:
        return False
    if open_revisit_count < CHAOS_MIN_OPEN_REVISITS:
        return False
    if CHAOS_REQUIRE_BOTH_SWEEPS and (sweep_high_count < 1 or sweep_low_count < 1):
        return False
    return True


def _is_trend_day(
    range_atr: float,
    open_revisit_count: int,
    day_close: float,
    day_low: float,
    day_range: float,
) -> bool:
    if range_atr < TREND_MIN_RANGE_ATR:
        return False
    if open_revisit_count > 1:
        return False
    if day_range <= 0:
        return False
    close_percent = (day_close - day_low) / day_range
    return close_percent <= 0.3 or close_percent >= 0.7


def _is_range_day(
    range_atr: float,
    open_revisit_count: int,
    close_to_open_atr: float,
) -> bool:
    return (
        range_atr <= RANGE_MAX_RANGE_ATR
        and open_revisit_count >= 2
        and close_to_open_atr <= OPEN_REVISIT_TOL_ATR
    )
