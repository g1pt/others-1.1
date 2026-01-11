"""Run MMXM research over all CSV/XLSX files in /data."""
from __future__ import annotations

import argparse
from dataclasses import replace
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


def _is_spx500_dataset(path: Path) -> bool:
    targets = {"forexcom_spx500, 15.csv", "fx_spx500, 60.csv"}
    return path.name.lower() in targets


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


def _run_instrument(path: Path, runs_dir: Path, combo_filter) -> None:
    instrument = _instrument_from_path(path)
    print(f"\n=== {instrument} ({path.name}) ===")
    candles = _load_candles(path)
    result = run_backtest(candles, combo_filter=combo_filter)
    label = instrument.lower()

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMXM research over all CSV/XLSX files in /data."
    )
    parser.add_argument(
        "--combo-filter",
        type=Path,
        help="Path to a JSON file containing combo summaries for whitelist filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    files = _find_data_files()
    if not files:
        raise SystemExit("No CSV/XLSX files found in /data or ./data")

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    combo_filter = load_combo_filter(args.combo_filter) if args.combo_filter else None

    for path in files:
        _run_instrument(path, runs_dir, combo_filter)


if __name__ == "__main__":
    main()
