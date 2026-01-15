"""Run MMXM research over all CSV/XLSX files in /data."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import date, datetime
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import (  # noqa: E402
    rank_summaries,
    summarize,
    summarize_combinations,
    summarize_day_labels,
)
from src.backtest import run_backtest  # noqa: E402
from src.data import load_candles_csv, load_candles_xlsx  # noqa: E402
from src.models import Candle
from src.day_context import label_days, label_for_timestamp  # noqa: E402
from src.filtering import load_combo_filter  # noqa: E402
from src.report import write_summary_csv, write_trades_csv  # noqa: E402
from src.reporting import leakage_report  # noqa: E402
from src.research.reporting import ob_failure_counts, top_candidates  # noqa: E402
from src.execution import (  # noqa: E402
    EngineConfig,
    ExecutionEngine,
    ExecutionMode,
    EquityLedger,
    RiskConfig,
    SignalPayload,
    TradeStatus,
)

DEFAULT_SYMBOL = "FX_SPX500"
DEFAULT_TFS = [15, 30]
DEFAULT_DATASETS = {
    f"{DEFAULT_SYMBOL.lower()}, {tf}.csv" for tf in DEFAULT_TFS
}
INITIAL_EQUITY = 10000.0
RISK_PER_TRADE = 0.01
UNIT_PNL_MODE = "R"
ENTRY_TYPE_FIELD = "entry_method"
PHASE_FIELD = "mmxm_phase"
OB_TRADABLE_FIELD = "ob_tradable"


@dataclass
class SimulationResult:
    taken_trades: list
    trade_results: list[dict]
    equity_end: float
    total_return_pct: float
    max_drawdown_pct: float
    stopped_by: str
    skipped_by_daily_loss: int
    skipped_by_max_trades: int
    missing_pnl_trades: int


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


def _dataset_names(symbol: str, tfs: list[int]) -> set[str]:
    normalized = symbol.strip().lower()
    names = set()
    for tf in tfs:
        names.add(f"{normalized}, {tf}.csv")
        names.add(f"{normalized}, {tf}.xlsx")
    return names


def _find_data_files(include_all: bool, symbol: str, tfs: list[int]) -> list[Path]:
    files: list[Path] = []
    for root in _data_roots():
        files.extend(sorted(root.glob("*.csv")))
        files.extend(sorted(root.glob("*.xlsx")))
    if include_all:
        return files
    targets = _dataset_names(symbol, tfs)
    return [path for path in files if path.name.lower() in targets]


def _instrument_from_path(path: Path) -> str:
    match = re.search(r"([A-Z]{6})", path.stem.upper())
    if match:
        return match.group(1)
    return path.stem.upper()


def _load_candles(path: Path) -> list[Candle]:
    if path.suffix.lower() == ".csv":
        return load_candles_csv(path)
    if path.suffix.lower() == ".xlsx":
        return load_candles_xlsx(path)
    raise ValueError(f"Unsupported file type: {path}")


def _write_summaries(trades, runs_dir: Path, label: str) -> None:
    by_phase = summarize(trades, lambda t: t.mmxm_phase, "Phase")
    by_entry = summarize(trades, lambda t: t.entry_method, "Entry")
    by_ob = summarize(
        trades, lambda t: "Tradable" if t.ob_tradable else "NonTradable", "OB"
    )
    by_combo = summarize_combinations(trades)
    ranking = rank_summaries(by_combo)
    candidates = top_candidates(by_combo)

    write_summary_csv(by_phase, runs_dir / f"summary_by_phase_{label}.csv")
    write_summary_csv(by_entry, runs_dir / f"summary_by_entry_{label}.csv")
    write_summary_csv(by_ob, runs_dir / f"summary_by_ob_{label}.csv")
    write_summary_csv(by_combo, runs_dir / f"summary_by_combination_{label}.csv")
    write_summary_csv(ranking, runs_dir / f"summary_ranking_{label}.csv")

    print("Summary by Phase")
    for row in by_phase:
        print(row)
    print("\nSummary by Entry")
    for row in by_entry:
        print(row)
    print("\nSummary by OB")
    for row in by_ob:
        print(row)
    print("\nSummary by Combination")
    for row in by_combo:
        print(row)
    print("\nCombined Ranking")
    for row in ranking:
        print(row)
    print("\nTop Candidates (filtered)")
    for row in candidates:
        print(row)


def _is_spx500_dataset(path: Path) -> bool:
    return path.name.lower() in DEFAULT_DATASETS


def _apply_day_labels(trades, day_context) -> list:
    labeled = []
    for trade in trades:
        day_label = label_for_timestamp(trade.entry_time, day_context)
        labeled.append(replace(trade, day_label=day_label))
    return labeled


def _print_day_context(trades, day_context) -> None:
    label_counts: dict[str, int] = {}
    for metrics in day_context.values():
        label_counts[metrics.label] = label_counts.get(metrics.label, 0) + 1

    print("\nDay Label Counts")
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}")

    risk_trades = [trade for trade in trades if trade.entry_method == "Risk Entry"]
    if not risk_trades:
        print("\nNo Risk Entry trades found for day label summary.")
        return

    risk_summary = summarize_day_labels(risk_trades)
    print("\nRisk Entry Expectancy by Day Label")
    for row in risk_summary:
        print(row)

    worst = min(risk_summary, key=lambda row: row.expectancy)
    print(
        "\nWorst Risk Entry Label: "
        f"{worst.key} (trades={worst.trades}, expectancy={worst.expectancy:.4f})"
    )


def _print_risk_entry_failure_analysis(trades) -> None:
    risk_trades = [trade for trade in trades if trade.entry_method == "Risk Entry"]
    if not risk_trades:
        print("\nRisk Entry Failure Analysis: No Risk Entry trades.")
        return

    summary = summarize_day_labels(risk_trades)
    sorted_by_expectancy = sorted(summary, key=lambda row: row.expectancy)
    worst_labels = sorted_by_expectancy[:3]
    best_labels = list(reversed(sorted_by_expectancy[-3:]))

    print("\nRisk Entry Failure Analysis")
    for row in summary:
        print(row)
    print("\nTop 3 Worst Day Labels (Risk Entry Expectancy)")
    for row in worst_labels:
        print(row)
    print("\nTop 3 Best Day Labels (Risk Entry Expectancy)")
    for row in best_labels:
        print(row)


def _normalize_entry_type(trade) -> str:
    raw = str(getattr(trade, ENTRY_TYPE_FIELD, "") or "").strip()
    value = raw.replace("Entry:", "").strip()
    return value or "Unknown"


def _normalize_tradability(trade) -> str:
    raw = getattr(trade, OB_TRADABLE_FIELD, None)
    if raw is True:
        return "Tradable"
    if raw is False:
        return "NonTradable"
    return "Unknown"


def _normalize_phase(trade) -> str:
    raw = str(getattr(trade, PHASE_FIELD, "") or "").strip()
    if not raw:
        return "Unknown"
    mapping = {
        "accumulation": "Accumulation",
        "manipulation": "Manipulation",
        "distribution": "Distribution",
    }
    return mapping.get(raw.lower(), raw)


def _trade_return_units(trade) -> tuple[float | None, str]:
    if trade.pnl_r is not None:
        return trade.pnl_r, "R"
    if trade.stop_price is None or trade.exit_price is None:
        return None, "Unknown"
    if trade.direction == "bullish":
        stop_distance = trade.entry_price - trade.stop_price
        if stop_distance <= 0:
            return None, "Unknown"
        return (trade.exit_price - trade.entry_price) / stop_distance, "R"
    if trade.direction == "bearish":
        stop_distance = trade.stop_price - trade.entry_price
        if stop_distance <= 0:
            return None, "Unknown"
        return (trade.entry_price - trade.exit_price) / stop_distance, "R"
    return None, "Unknown"


def _trade_points_units(trade) -> float | None:
    if trade.exit_price is None:
        return None
    if trade.direction == "bullish":
        return trade.exit_price - trade.entry_price
    if trade.direction == "bearish":
        return trade.entry_price - trade.exit_price
    return None


def _ensure_datetime(value: datetime | date | str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.min
    return datetime.min


def filter_trades(
    trades,
    allow_entry_types: list[str],
    require_tradable_ob: bool,
    allow_phases: list[str] | None,
) -> list:
    allowed_entries = {_normalize_entry_type_label(entry) for entry in allow_entry_types}
    allowed_phases = (
        {_normalize_phase_label(phase) for phase in allow_phases}
        if allow_phases
        else None
    )
    filtered = []
    for trade in trades:
        entry_type = _normalize_entry_type(trade)
        if entry_type == "Continuation Entry":
            continue
        if allowed_entries and entry_type not in allowed_entries:
            continue
        if require_tradable_ob and _normalize_tradability(trade) != "Tradable":
            continue
        if allowed_phases and _normalize_phase(trade) not in allowed_phases:
            continue
        filtered.append(trade)
    return filtered


def simulate_equity(
    trades,
    initial_equity: float = 10000.0,
    base_risk_pct: float = 0.01,
    max_dd_pct: float = 0.03,
    max_trades_per_day: int = 1,
    cooldown_losses: int = 2,
    reduce_risk_under_hwm: bool = True,
    reduced_risk_pct: float = 0.005,
) -> SimulationResult:
    equity = initial_equity
    high_watermark = equity
    max_drawdown = 0.0
    trade_results: list[dict] = []
    taken_trades: list = []
    trades_per_day: dict[date, int] = {}
    skipped_by_daily_loss = 0
    skipped_by_max_trades = 0
    stopped_by = "none"
    missing_pnl_trades = 0
    consecutive_losses = 0

    for trade in sorted(trades, key=lambda t: _ensure_datetime(t.entry_time)):
        entry_time = _ensure_datetime(trade.entry_time)
        trade_date = entry_time.date()
        if stopped_by != "none":
            break
        if max_trades_per_day and trades_per_day.get(trade_date, 0) >= max_trades_per_day:
            skipped_by_max_trades += 1
            continue
        if trade_date not in trades_per_day:
            trades_per_day[trade_date] = 0

        trade_return, mode = _trade_return_units(trade)
        if trade_return is None or mode != "R":
            missing_pnl_trades += 1
            continue

        risk_pct = (
            reduced_risk_pct
            if reduce_risk_under_hwm and equity < high_watermark
            else base_risk_pct
        )
        risk_cash = equity * risk_pct
        pnl_cash = trade_return * risk_cash
        equity += pnl_cash
        high_watermark = max(high_watermark, equity)
        drawdown = (high_watermark - equity) / high_watermark if high_watermark else 0.0
        max_drawdown = max(max_drawdown, drawdown)

        trade_results.append(
            {"trade": trade, "return_units": trade_return, "pnl_cash": pnl_cash}
        )
        taken_trades.append(trade)
        trades_per_day[trade_date] += 1

        if trade_return < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        if max_dd_pct and drawdown >= max_dd_pct:
            stopped_by = "max_dd"
            break
        if cooldown_losses and consecutive_losses >= cooldown_losses:
            stopped_by = "cooldown_losses"
            break

    total_return_pct = (
        (equity - initial_equity) / initial_equity * 100 if initial_equity else 0.0
    )
    max_drawdown_pct = max_drawdown * 100

    return SimulationResult(
        taken_trades=taken_trades,
        trade_results=trade_results,
        equity_end=equity,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        stopped_by=stopped_by,
        skipped_by_daily_loss=skipped_by_daily_loss,
        skipped_by_max_trades=skipped_by_max_trades,
        missing_pnl_trades=missing_pnl_trades,
    )


def _normalize_entry_type_label(label: str) -> str:
    return str(label or "").strip() or "Unknown"


def _normalize_phase_label(label: str) -> str:
    raw = str(label or "").strip()
    if not raw:
        return "Unknown"
    mapping = {
        "accumulation": "Accumulation",
        "manipulation": "Manipulation",
        "distribution": "Distribution",
    }
    return mapping.get(raw.lower(), raw)


def _bucket_contribution(trade_results: list[dict], key_fn):
    grouped: dict[str, list[dict]] = {}
    for result in trade_results:
        key = key_fn(result["trade"])
        grouped.setdefault(key, []).append(result)
    rows = []
    total_pnl_cash = sum(result["pnl_cash"] for result in trade_results)
    for key, results in grouped.items():
        trades = len(results)
        total_pnl_units = sum(result["return_units"] for result in results)
        pnl_cash = sum(result["pnl_cash"] for result in results)
        contribution_pct = (pnl_cash / total_pnl_cash * 100) if total_pnl_cash else 0.0
        rows.append(
            {
                "key": key,
                "trades": trades,
                "total_pnl_units": total_pnl_units,
                "pnl_cash": pnl_cash,
                "contribution_pct": contribution_pct,
            }
        )
    rows.sort(key=lambda row: (-row["contribution_pct"], -row["trades"], row["key"]))
    return rows


def _print_leakage_contribution(trade_results: list[dict], missing_pnl: int = 0) -> None:
    total_pnl_cash = sum(result["pnl_cash"] for result in trade_results)
    print("\n=== Leakage Contribution (profit share) ===")
    print(f"Total pnl_cash={total_pnl_cash:.2f}")
    print("-- By EntryType")
    for row in _bucket_contribution(trade_results, _normalize_entry_type):
        print(
            f"{row['key']}: trades={row['trades']} pnl_R={row['total_pnl_units']:.2f} "
            f"pnl_cash={row['pnl_cash']:.2f} "
            f"contrib={row['contribution_pct']:.2f}%"
        )
    print("-- By OB Tradability")
    for row in _bucket_contribution(trade_results, _normalize_tradability):
        print(
            f"{row['key']}: trades={row['trades']} pnl_R={row['total_pnl_units']:.2f} "
            f"pnl_cash={row['pnl_cash']:.2f} "
            f"contrib={row['contribution_pct']:.2f}%"
        )
    print("-- By Phase")
    for row in _bucket_contribution(trade_results, _normalize_phase):
        print(
            f"{row['key']}: trades={row['trades']} pnl_R={row['total_pnl_units']:.2f} "
            f"pnl_cash={row['pnl_cash']:.2f} "
            f"contrib={row['contribution_pct']:.2f}%"
        )
    print("-- Cross: Phase x EntryType (top 10 only)")
    for row in _bucket_contribution(
        trade_results,
        lambda trade: f"{_normalize_phase(trade)} | {_normalize_entry_type(trade)}",
    )[:10]:
        print(
            f"{row['key']}: trades={row['trades']} pnl_R={row['total_pnl_units']:.2f} "
            f"pnl_cash={row['pnl_cash']:.2f} "
            f"contrib={row['contribution_pct']:.2f}%"
        )
    print("-- Cross: Tradable x EntryType")
    for row in _bucket_contribution(
        trade_results,
        lambda trade: f"{_normalize_tradability(trade)} | {_normalize_entry_type(trade)}",
    ):
        print(
            f"{row['key']}: trades={row['trades']} pnl_R={row['total_pnl_units']:.2f} "
            f"pnl_cash={row['pnl_cash']:.2f} "
            f"contrib={row['contribution_pct']:.2f}%"
        )
    print("-- Cross: Phase x Tradable")
    for row in _bucket_contribution(
        trade_results,
        lambda trade: f"{_normalize_phase(trade)} | {_normalize_tradability(trade)}",
    ):
        print(
            f"{row['key']}: trades={row['trades']} pnl_R={row['total_pnl_units']:.2f} "
            f"pnl_cash={row['pnl_cash']:.2f} "
            f"contrib={row['contribution_pct']:.2f}%"
        )
    print(f"missing_pnl_trades={missing_pnl}")


def _print_step4c_leakage_contribution(trade_results: list[dict]) -> None:
    print("\n--- Leakage Contribution ---")
    print("-- By EntryType")
    for row in _bucket_contribution(trade_results, _normalize_entry_type):
        print(
            f"{row['key']}: trades={row['trades']} pnl_R={row['total_pnl_units']:.2f} "
            f"pnl_cash={row['pnl_cash']:.2f} "
            f"contrib={row['contribution_pct']:.2f}%"
        )
    print("-- By Phase")
    for row in _bucket_contribution(trade_results, _normalize_phase):
        print(
            f"{row['key']}: trades={row['trades']} pnl_R={row['total_pnl_units']:.2f} "
            f"pnl_cash={row['pnl_cash']:.2f} "
            f"contrib={row['contribution_pct']:.2f}%"
        )
    print("-- By OB Tradability")
    for row in _bucket_contribution(trade_results, _normalize_tradability):
        print(
            f"{row['key']}: trades={row['trades']} pnl_R={row['total_pnl_units']:.2f} "
            f"pnl_cash={row['pnl_cash']:.2f} "
            f"contrib={row['contribution_pct']:.2f}%"
        )


def _simulate_equity_and_contribution(trades) -> None:
    # Step 4B: Equity + Contribution
    equity = INITIAL_EQUITY
    peak_equity = equity
    max_drawdown = 0.0
    trade_results: list[dict] = []
    missing_pnl = 0

    for trade in sorted(trades, key=lambda t: t.entry_time):
        if UNIT_PNL_MODE == "POINTS":
            trade_points = _trade_points_units(trade)
            if trade_points is None:
                missing_pnl += 1
                continue
            trade_return = trade_points
            mode = "POINTS"
        else:
            trade_return, mode = _trade_return_units(trade)
            if trade_return is None:
                trade_points = _trade_points_units(trade)
                if trade_points is None:
                    missing_pnl += 1
                    continue
                trade_return = trade_points
                mode = "POINTS"
        pnl_cash = equity * RISK_PER_TRADE * trade_return
        equity += pnl_cash
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, peak_equity - equity)
        trade_results.append(
            {
                "trade": trade,
                "return_units": trade_return,
                "pnl_cash": pnl_cash,
                "mode": mode,
            }
        )

    trades_count = len(trade_results)
    end_equity = equity
    total_return_pct = (
        (end_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100
        if INITIAL_EQUITY
        else 0.0
    )
    max_drawdown_pct = (max_drawdown / peak_equity * 100) if peak_equity else 0.0

    print(
        f"=== Equity Simulation (initial={INITIAL_EQUITY:.0f}, risk={RISK_PER_TRADE:.0%}) ==="
    )
    print(f"trades={trades_count}")
    print(f"end_equity={end_equity:.2f}")
    print(f"total_return_pct={total_return_pct:.2f}")
    print(f"max_drawdown_pct={max_drawdown_pct:.2f}")

    _print_leakage_contribution(trade_results, missing_pnl=missing_pnl)


def _validate_risk_pct(risk_pct: float) -> None:
    if not 0.01 <= risk_pct <= 0.03:
        raise ValueError("risk must be between 0.01 and 0.03")


def compute_sl_tp(
    entry_price: float, direction: str, sl_pct: float = 0.002, rr: float = 2.0
) -> tuple[float, float]:
    stop_distance = entry_price * sl_pct
    if direction == "bullish":
        sl_price = entry_price - stop_distance
        tp_price = entry_price + (stop_distance * rr)
    else:
        sl_price = entry_price + stop_distance
        tp_price = entry_price - (stop_distance * rr)
    return sl_price, tp_price


def _simulate_rr_trade(
    trade,
    candles: list[Candle],
    candle_index: dict[str, int],
    sl_pct: float,
    rr: float,
) -> Trade | None:
    entry_idx = candle_index.get(trade.entry_time)
    if entry_idx is None:
        return None
    sl_price, tp_price = compute_sl_tp(
        trade.entry_price, trade.direction, sl_pct=sl_pct, rr=rr
    )
    exit_time = None
    exit_price = None
    pnl_r = None

    start_idx = min(entry_idx + 1, len(candles))
    for candle in candles[start_idx:]:
        if trade.direction == "bullish":
            hit_sl = candle.low <= sl_price
            hit_tp = candle.high >= tp_price
            if hit_sl and hit_tp:
                exit_time = candle.timestamp
                exit_price = sl_price
                pnl_r = -1.0
                break
            if hit_sl:
                exit_time = candle.timestamp
                exit_price = sl_price
                pnl_r = -1.0
                break
            if hit_tp:
                exit_time = candle.timestamp
                exit_price = tp_price
                pnl_r = rr
                break
        else:
            hit_sl = candle.high >= sl_price
            hit_tp = candle.low <= tp_price
            if hit_sl and hit_tp:
                exit_time = candle.timestamp
                exit_price = sl_price
                pnl_r = -1.0
                break
            if hit_sl:
                exit_time = candle.timestamp
                exit_price = sl_price
                pnl_r = -1.0
                break
            if hit_tp:
                exit_time = candle.timestamp
                exit_price = tp_price
                pnl_r = rr
                break

    if exit_time is None:
        if not candles:
            return None
        last_candle = candles[-1]
        exit_time = last_candle.timestamp
        exit_price = last_candle.close
        stop_distance = abs(trade.entry_price - sl_price)
        if stop_distance > 0:
            if trade.direction == "bullish":
                pnl_r = (exit_price - trade.entry_price) / stop_distance
            else:
                pnl_r = (trade.entry_price - exit_price) / stop_distance

    return replace(
        trade,
        stop_price=sl_price,
        exit_time=exit_time,
        exit_price=exit_price,
        pnl_r=pnl_r,
    )


def _compute_rr_metrics(
    trades: list[Trade],
    initial_equity: float,
    risk_per_trade: float,
) -> dict:
    pnl_rs = [trade.pnl_r for trade in trades if trade.pnl_r is not None]
    trades_count = len(pnl_rs)
    wins = sum(1 for pnl in pnl_rs if pnl > 0)
    losses = sum(1 for pnl in pnl_rs if pnl < 0)
    winrate = (wins / trades_count * 100) if trades_count else 0.0
    avg_r = sum(pnl_rs) / trades_count if trades_count else 0.0
    median_r = statistics.median(pnl_rs) if trades_count else 0.0
    expectancy_r = avg_r

    gross_profit = sum(pnl for pnl in pnl_rs if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in pnl_rs if pnl < 0))
    if gross_loss == 0:
        profit_factor = float("inf") if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    equity = initial_equity
    peak_equity = initial_equity
    max_drawdown_pct = 0.0
    for pnl_r in pnl_rs:
        equity += equity * risk_per_trade * pnl_r
        peak_equity = max(peak_equity, equity)
        drawdown_pct = (
            (peak_equity - equity) / peak_equity * 100 if peak_equity else 0.0
        )
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
    total_return_pct = (
        (equity - initial_equity) / initial_equity * 100
        if initial_equity
        else 0.0
    )

    longest_loss_streak = 0
    current_streak = 0
    for pnl_r in pnl_rs:
        if pnl_r < 0:
            current_streak += 1
        else:
            longest_loss_streak = max(longest_loss_streak, current_streak)
            current_streak = 0
    longest_loss_streak = max(longest_loss_streak, current_streak)

    return {
        "trades_count": trades_count,
        "winrate": winrate,
        "expectancy_R": expectancy_r,
        "end_equity": equity,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "profit_factor": profit_factor,
        "avg_R": avg_r,
        "median_R": median_r,
        "longest_loss_streak": longest_loss_streak,
        "wins": wins,
        "losses": losses,
    }


def _format_rr_table(rr_rows: list[dict]) -> None:
    headers = [
        "RR",
        "Trades",
        "Win%",
        "ExpectR",
        "EndEq",
        "Ret%",
        "MaxDD%",
        "PF",
        "AvgR",
        "MedR",
        "LossStk",
    ]
    print(" | ".join(headers))
    print("-" * 110)
    for row in rr_rows:
        print(
            " | ".join(
                [
                    f"{row['rr']:.1f}",
                    f"{row['trades_count']}",
                    f"{row['winrate']:.2f}",
                    f"{row['expectancy_R']:.3f}",
                    f"{row['end_equity']:.2f}",
                    f"{row['total_return_pct']:.2f}",
                    f"{row['max_drawdown_pct']:.2f}",
                    f"{row['profit_factor']:.2f}"
                    if row["profit_factor"] != float("inf")
                    else "inf",
                    f"{row['avg_R']:.3f}",
                    f"{row['median_R']:.3f}",
                    f"{row['longest_loss_streak']}",
                ]
            )
        )


def run_rr_sweep(
    existing_trades: list[Trade],
    rr_list: list[float],
    config: dict,
    tf_label: str,
) -> None:
    candles: list[Candle] = config["candles"]
    candle_index = {candle.timestamp: idx for idx, candle in enumerate(candles)}
    initial_equity = config["initial_equity"]
    risk_per_trade = config["risk_per_trade"]
    symbol = config["symbol"]
    sl_pct = config.get("sl_pct", 0.002)

    rr_results: list[dict] = []
    rr_splits: dict[float, list[dict]] = {}

    for rr in rr_list:
        sweep_trades = []
        for trade in existing_trades:
            updated = _simulate_rr_trade(
                trade,
                candles,
                candle_index,
                sl_pct=sl_pct,
                rr=rr,
            )
            if updated is None or updated.pnl_r is None:
                continue
            sweep_trades.append(updated)

        metrics = _compute_rr_metrics(
            sweep_trades,
            initial_equity=initial_equity,
            risk_per_trade=risk_per_trade,
        )
        metrics["rr"] = rr
        rr_results.append(metrics)

        split_metrics = []
        for trade_slice in _split_trades_into_slices(sweep_trades):
            split_metrics.append(
                _compute_rr_metrics(
                    trade_slice,
                    initial_equity=initial_equity,
                    risk_per_trade=risk_per_trade,
                )
            )
        rr_splits[rr] = split_metrics

    rr_results.sort(
        key=lambda row: (row["max_drawdown_pct"], -row["total_return_pct"])
    )

    print(f"\n=== RR SWEEP ({symbol} {tf_label}) ===")
    _format_rr_table(rr_results)

    print("\n--- RR Sweep Robustness (early/mid/late) ---")
    slice_labels = ["early", "mid", "late"]
    for rr in rr_list:
        print(f"RR {rr:.1f}")
        for label, metrics in zip(slice_labels, rr_splits.get(rr, [])):
            print(
                f"{label}: trades={metrics['trades_count']} "
                f"end_equity={metrics['end_equity']:.2f} "
                f"total_return_pct={metrics['total_return_pct']:.2f} "
                f"max_drawdown_pct={metrics['max_drawdown_pct']:.2f}"
            )

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    payload = {
        "symbol": symbol,
        "timeframe": tf_label,
        "sl_pct": sl_pct,
        "rr_list": rr_list,
        "results": rr_results,
        "robustness": {
            f"{rr:.1f}": rr_splits.get(rr, []) for rr in rr_list
        },
    }
    out_path = logs_dir / f"rr_sweep_{symbol.lower()}_{tf_label}.json"
    out_path.write_text(json.dumps(payload, indent=2))


def _run_step4c_matrix(
    trades,
    initial_equity: float,
    risk_pct: float,
    max_dd_pct: float,
    max_trades_per_day: int,
) -> None:
    variants = [
        (
            "[4C-A] BASELINE",
            "Baseline",
            lambda trade: True,
        ),
        (
            "[4C-B] Refinement only",
            "Refinement Entry",
            lambda trade: _normalize_entry_type(trade) == "Refinement Entry",
        ),
        (
            "[4C-C] Refinement + Tradable OB",
            "Refinement Entry + Tradable",
            lambda trade: _normalize_entry_type(trade) == "Refinement Entry"
            and _normalize_tradability(trade) == "Tradable",
        ),
        (
            "[4C-D] Refinement + Tradable + Manipulation",
            "Refinement Entry + Tradable + Manipulation",
            lambda trade: _normalize_entry_type(trade) == "Refinement Entry"
            and _normalize_tradability(trade) == "Tradable"
            and _normalize_phase(trade) == "Manipulation",
        ),
    ]

    for label, _, predicate in variants:
        filtered = [trade for trade in trades if predicate(trade)]
        sim_result = simulate_equity(
            filtered,
            initial_equity=initial_equity,
            base_risk_pct=risk_pct,
            max_dd_pct=max_dd_pct,
            max_trades_per_day=max_trades_per_day,
        )
        print("\n==============================")
        print(label)
        print("==============================")
        if not sim_result.taken_trades:
            print("no trades")
        print(f"trades={len(sim_result.taken_trades)}")
        print(f"end_equity={sim_result.equity_end:.2f}")
        print(f"total_return_pct={sim_result.total_return_pct:.2f}")
        print(f"max_drawdown_pct={sim_result.max_drawdown_pct:.2f}")
        _print_step4c_leakage_contribution(sim_result.trade_results)


def _split_trades_into_slices(trades, slices: int = 3) -> list[list]:
    if not trades:
        return [[] for _ in range(slices)]
    total = len(trades)
    base = total // slices
    remainder = total % slices
    result = []
    start = 0
    for idx in range(slices):
        size = base + (1 if idx < remainder else 0)
        end = start + size
        result.append(list(trades[start:end]))
        start = end
    return result


def _run_step4d_robustness(
    trades,
    initial_equity: float,
    risk_pct: float,
    max_dd_pct: float,
    max_trades_per_day: int,
) -> None:
    print("\n=== [4D] ROBUSTNESS (MMXM_4C_D) ===")
    slice_labels = ["[4D-1] early", "[4D-2] mid", "[4D-3] late"]
    for label, trade_slice in zip(slice_labels, _split_trades_into_slices(trades)):
        sim_result = simulate_equity(
            trade_slice,
            initial_equity=initial_equity,
            base_risk_pct=risk_pct,
            max_dd_pct=max_dd_pct,
            max_trades_per_day=max_trades_per_day,
        )
        print(label)
        if not sim_result.taken_trades:
            print("no trades")
        print(f"trades={len(sim_result.taken_trades)}")
        print(f"end_equity={sim_result.equity_end:.2f}")
        print(f"total_return_pct={sim_result.total_return_pct:.2f}")
        print(f"max_drawdown_pct={sim_result.max_drawdown_pct:.2f}")


def _timeframe_label_from_path(path: Path) -> str:
    match = re.search(r",\s*(\d+)", path.stem)
    if match:
        return f"{match.group(1)}m"
    return "unknown"


def _run_paper_execution(
    trades,
    candles: list[Candle],
    instrument: str,
    timeframe_label: str,
    initial_equity: float,
    risk_pct: float,
) -> None:
    print("\n=== Paper Execution (MMXM_4C_D) ===")
    ledger = EquityLedger(
        equity_current=initial_equity,
        equity_start_day=initial_equity,
    )
    engine = ExecutionEngine(
        config=EngineConfig(
            mode=ExecutionMode.PAPER_SIM,
            risk_per_trade=risk_pct,
            sl_pct=0.002,
            rr_default=2.0,
            symbol_whitelist=frozenset({instrument, "SP500"}),
            ruleset_whitelist=frozenset({"MMXM_4C_D"}),
            timeframe_whitelist=frozenset({timeframe_label}),
        ),
        ledger=ledger,
        risk_config=RiskConfig(
            max_trades_per_day=1,
            max_consecutive_losses=2,
            daily_drawdown_limit_pct=2.0,
            hard_drawdown_limit_pct=3.0,
        ),
        candles=candles,
    )
    accepted = 0
    rejected = 0
    for trade in sorted(trades, key=lambda t: t.entry_time):
        decision = engine.accept_signal(
            SignalPayload(
                symbol=instrument,
                timeframe=timeframe_label,
                ruleset_id="MMXM_4C_D",
                entry_time=trade.entry_time,
                entry_price=trade.entry_price,
                direction=trade.direction,
                entry_type=trade.entry_method,
                phase=trade.mmxm_phase,
                ob_tradable=trade.ob_tradable,
            )
        )
        if decision.status == TradeStatus.REJECTED:
            rejected += 1
        else:
            accepted += 1
    engine.finalize()
    print(f"paper_trades_accepted={accepted}")
    print(f"paper_trades_rejected={rejected}")


def _run_instrument_baseline(path: Path, runs_dir: Path, combo_filter) -> None:
    instrument = _instrument_from_path(path)
    print(f"\n=== {instrument} ({path.name}) ===")
    candles = _load_candles(path)
    result = run_backtest(candles, combo_filter=combo_filter)
    label = instrument.lower()
    failure_counts = ob_failure_counts(result.order_blocks)
    true_count = sum(1 for ob in result.order_blocks if ob.has_imbalance)
    false_count = len(result.order_blocks) - true_count
    print(f"FVG sanity: true_count={true_count} false_count={false_count}")
    print("\nOB Tradability Failure Counts")
    print(f"total={failure_counts['total']}")
    print(f"tradable={failure_counts['tradable']}")
    print(f"nontradable={failure_counts['nontradable']}")
    print(f"no_fvg={failure_counts['no_fvg']}")
    print(f"no_bos={failure_counts['no_bos']}")
    print(f"not_near_level={failure_counts['not_near_level']}")

    if _is_spx500_dataset(path):
        day_context = label_days(candles)
        trades = _apply_day_labels(result.trades, day_context)
        result = replace(result, trades=trades)
        write_summary_csv(
            summarize_day_labels(trades),
            runs_dir / f"summary_by_daylabel_{label}.csv",
        )
        risk_trades = [trade for trade in trades if trade.entry_method == "Risk Entry"]
        write_summary_csv(
            summarize_day_labels(risk_trades),
            runs_dir / f"risk_entry_by_daylabel_{label}.csv",
        )
        _print_day_context(trades, day_context)
        _print_risk_entry_failure_analysis(trades)

    write_trades_csv(result, runs_dir / f"trades_{label}.csv")
    _write_summaries(result.trades, runs_dir, label)
    print()
    print(leakage_report(result.trades))
    print()
    _simulate_equity_and_contribution(result.trades)


def _run_instrument_step4c(
    path: Path,
    combo_filter,
    initial_equity: float,
    risk_pct: float,
    max_dd_pct: float,
    max_trades_per_day: int,
    paper_execute: bool,
) -> None:
    instrument = _instrument_from_path(path)
    print(f"\n=== {instrument} ({path.name}) ===")
    candles = _load_candles(path)
    result = run_backtest(candles, combo_filter=combo_filter)
    _run_step4c_matrix(
        result.trades,
        initial_equity,
        risk_pct,
        max_dd_pct,
        max_trades_per_day,
    )
    step4d_trades = [
        trade
        for trade in result.trades
        if _normalize_entry_type(trade) == "Refinement Entry"
        and _normalize_tradability(trade) == "Tradable"
        and _normalize_phase(trade) == "Manipulation"
    ]
    _run_step4d_robustness(
        step4d_trades,
        initial_equity,
        risk_pct,
        max_dd_pct,
        max_trades_per_day,
    )
    run_rr_sweep(
        step4d_trades,
        [2.0, 3.0, 4.0, 5.0],
        {
            "candles": candles,
            "initial_equity": initial_equity,
            "risk_per_trade": risk_pct,
            "symbol": instrument,
            "sl_pct": 0.002,
        },
        _timeframe_label_from_path(path),
    )
    if paper_execute:
        _run_paper_execution(
            step4d_trades,
            candles,
            instrument,
            _timeframe_label_from_path(path),
            initial_equity,
            risk_pct,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMXM research over all CSV/XLSX files in /data."
    )
    parser.add_argument(
        "--combo-filter",
        type=Path,
        help="Path to a JSON file containing combo summaries for whitelist filtering.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run research on all CSV/XLSX files instead of the default SPX500 15/30 set.",
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Instrument symbol.")
    parser.add_argument(
        "--tfs",
        nargs="+",
        type=int,
        default=DEFAULT_TFS,
        help="Timeframes to run (e.g. 15 30).",
    )
    parser.add_argument("--initial-equity", type=float, default=INITIAL_EQUITY)
    parser.add_argument(
        "--risk",
        type=float,
        default=0.02,
        help="Risk per trade (0.01-0.03).",
    )
    parser.add_argument("--max-dd", type=float, default=0.03)
    parser.add_argument(
        "--daily-loss",
        type=float,
        default=0.02,
        help="Daily loss guard (0 disables).",
    )
    parser.add_argument(
        "--max-trades-day",
        type=int,
        default=3,
        help="Max trades per day (0 disables).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the legacy baseline output.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run basic sanity checks and exit.",
    )
    parser.add_argument(
        "--paper-execute",
        action="store_true",
        help="Run paper execution on MMXM_4C_D trades and write logs.",
    )
    return parser.parse_args()


def _run_self_test() -> None:
    class _TestTrade:
        def __init__(self, entry_time: datetime, pnl_r: float) -> None:
            self.entry_time = entry_time
            self.pnl_r = pnl_r
            self.entry_method = "Refinement Entry"
            self.mmxm_phase = "Manipulation"
            self.ob_tradable = True
            self.stop_price = None
            self.exit_price = None
            self.entry_price = None
            self.direction = None

    try:
        _validate_risk_pct(0.005)
        raise AssertionError("Expected risk validation to fail for 0.005")
    except ValueError:
        pass
    _validate_risk_pct(0.02)

    trades_dd = [
        _TestTrade(datetime(2024, 1, 1, 9), -2.0),
        _TestTrade(datetime(2024, 1, 1, 10), 1.0),
    ]
    dd_result = simulate_equity(
        trades_dd,
        initial_equity=10000,
        base_risk_pct=0.02,
        max_dd_pct=0.03,
        max_trades_per_day=0,
    )
    assert dd_result.stopped_by == "max_dd"

    trades_cooldown = [
        _TestTrade(datetime(2024, 1, 1, 9), -1.0),
        _TestTrade(datetime(2024, 1, 1, 10), -1.0),
        _TestTrade(datetime(2024, 1, 2, 9), 1.0),
    ]
    cooldown_result = simulate_equity(
        trades_cooldown,
        initial_equity=10000,
        base_risk_pct=0.02,
        max_dd_pct=0.5,
        max_trades_per_day=0,
        cooldown_losses=2,
    )
    assert len(cooldown_result.taken_trades) == 2
    assert cooldown_result.stopped_by == "cooldown_losses"


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _run_self_test()
        print("Self-test passed.")
        return

    _validate_risk_pct(args.risk)
    files = _find_data_files(args.all_datasets, args.symbol, args.tfs)
    if not files:
        raise SystemExit("No CSV/XLSX files found in /data or ./data")

    combo_filter = load_combo_filter(args.combo_filter) if args.combo_filter else None

    for path in files:
        if args.baseline:
            runs_dir = Path("runs")
            runs_dir.mkdir(exist_ok=True)
            _run_instrument_baseline(path, runs_dir, combo_filter)
        else:
            _run_instrument_step4c(
                path,
                combo_filter,
                args.initial_equity,
                args.risk,
                args.max_dd,
                args.max_trades_day,
                args.paper_execute,
            )


if __name__ == "__main__":
    main()

# Usage:
# python scripts/run_mmxm_research.py --symbol FX_SPX500 --tfs 15 30 --risk 0.02 \
#   --max-dd 0.03 --daily-loss 0.02 --max-trades-day 3
