"""Run MMXM research over all CSV/XLSX files in /data."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
import re
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
    initial_equity: float,
    risk_pct: float,
    max_dd_pct: float,
    daily_loss_pct: float,
    max_trades_per_day: int,
) -> SimulationResult:
    equity = initial_equity
    peak_equity = equity
    max_drawdown = 0.0
    trade_results: list[dict] = []
    taken_trades: list = []
    day_start_equity: dict[date, float] = {}
    trades_per_day: dict[date, int] = {}
    daily_loss_blocked: set[date] = set()
    skipped_by_daily_loss = 0
    skipped_by_max_trades = 0
    stopped_by = "none"

    for trade in sorted(trades, key=lambda t: t.entry_time):
        trade_date = trade.entry_time.date()
        if stopped_by == "max_dd":
            break
        if daily_loss_pct and trade_date in daily_loss_blocked:
            skipped_by_daily_loss += 1
            continue
        if max_trades_per_day and trades_per_day.get(trade_date, 0) >= max_trades_per_day:
            skipped_by_max_trades += 1
            continue
        if trade_date not in day_start_equity:
            day_start_equity[trade_date] = equity
            trades_per_day[trade_date] = 0

        trade_return, mode = _trade_return_units(trade)
        if trade_return is None or mode != "R":
            continue

        risk_cash = equity * risk_pct
        pnl_cash = trade_return * risk_cash
        equity += pnl_cash
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0.0
        max_drawdown = max(max_drawdown, drawdown)

        trade_results.append(
            {"trade": trade, "return_units": trade_return, "pnl_cash": pnl_cash}
        )
        taken_trades.append(trade)
        trades_per_day[trade_date] += 1

        if max_dd_pct and drawdown >= max_dd_pct:
            stopped_by = "max_dd"
            break
        if daily_loss_pct:
            day_start = day_start_equity[trade_date]
            if equity <= day_start * (1 - daily_loss_pct):
                daily_loss_blocked.add(trade_date)

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


def _print_variant_report(
    label: str,
    baseline_trades: int,
    filtered_trades: list,
    sim_result: SimulationResult,
) -> None:
    print(f"\n--- {label} ---")
    print(f"trades_in_baseline={baseline_trades}")
    print(f"trades_after_filter={len(filtered_trades)}")
    print(f"trades_taken_after_guards={len(sim_result.taken_trades)}")
    print(f"end_equity={sim_result.equity_end:.2f}")
    print(f"total_return_pct={sim_result.total_return_pct:.2f}")
    print(f"max_drawdown_pct={sim_result.max_drawdown_pct:.2f}")
    print(f"stopped_by={sim_result.stopped_by}")
    print(f"skipped_by_daily_loss={sim_result.skipped_by_daily_loss}")
    print(f"skipped_by_max_trades={sim_result.skipped_by_max_trades}")
    _print_leakage_contribution(sim_result.trade_results)


def _run_step4c_matrix(
    trades,
    initial_equity: float,
    risk_pct: float,
    max_dd_pct: float,
    daily_loss_pct: float,
    max_trades_per_day: int,
) -> None:
    variants = [
        (
            "Variant A: Refinement only + Tradable only + all phases",
            ["Refinement Entry"],
            True,
            None,
        ),
        (
            "Variant B: Refinement only + Tradable only + Manipulation only",
            ["Refinement Entry"],
            True,
            ["Manipulation"],
        ),
        (
            "Variant C: Refinement + Risk + Tradable only + all phases",
            ["Refinement Entry", "Risk Entry"],
            True,
            None,
        ),
        (
            "Variant D: Refinement + Risk + Tradable only + Manipulation only",
            ["Refinement Entry", "Risk Entry"],
            True,
            ["Manipulation"],
        ),
        (
            "Variant E (diagnostic): Refinement + Risk + Confirmation + Tradable only + Manipulation only",
            ["Refinement Entry", "Risk Entry", "Confirmation Entry"],
            True,
            ["Manipulation"],
        ),
    ]

    baseline_trades = len(trades)
    for label, allow_entries, require_tradable, allow_phases in variants:
        filtered = filter_trades(trades, allow_entries, require_tradable, allow_phases)
        sim_result = simulate_equity(
            filtered,
            initial_equity,
            risk_pct,
            max_dd_pct,
            daily_loss_pct,
            max_trades_per_day,
        )
        _print_variant_report(label, baseline_trades, filtered, sim_result)


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
    daily_loss_pct: float,
    max_trades_per_day: int,
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
        daily_loss_pct,
        max_trades_per_day,
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
    dd_result = simulate_equity(trades_dd, 10000, 0.02, 0.03, 0.0, 0)
    assert dd_result.stopped_by == "max_dd"

    trades_daily = [
        _TestTrade(datetime(2024, 1, 1, 9), -1.0),
        _TestTrade(datetime(2024, 1, 1, 10), -1.0),
        _TestTrade(datetime(2024, 1, 2, 9), 1.0),
    ]
    daily_result = simulate_equity(trades_daily, 10000, 0.02, 0.5, 0.02, 0)
    assert len(daily_result.taken_trades) == 2
    assert daily_result.skipped_by_daily_loss >= 1


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
                args.daily_loss,
                args.max_trades_day,
            )


if __name__ == "__main__":
    main()

# Usage:
# python scripts/run_mmxm_research.py --symbol FX_SPX500 --tfs 15 30 --risk 0.02 \
#   --max-dd 0.03 --daily-loss 0.02 --max-trades-day 3
