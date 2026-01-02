"""Run MMXM research over all CSV/XLSX files in /data."""
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import (  # noqa: E402
    DEFAULT_MIN_TRADES,
    rank_summaries,
    summarize,
    summarize_combinations,
)
from src.backtest import run_backtest  # noqa: E402
from src.data import Candle, load_candles_csv, load_candles_xlsx  # noqa: E402
from src.order_blocks import tradability_failure_counts  # noqa: E402
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
    min_trades = DEFAULT_MIN_TRADES
    ranking_base = [row for row in by_combo if row.trades >= min_trades]
    small_sample = [row for row in by_combo if row.trades < min_trades]
    ranking = rank_summaries(ranking_base)

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
    if small_sample:
        print(f"\nSmall-sample combos (< {min_trades} trades)")
        for row in small_sample:
            print(row)


def _print_ob_debug(order_blocks, instrument: str) -> None:
    total = len(order_blocks)
    tradable = sum(1 for block in order_blocks if block.tradable)
    nontradable = total - tradable
    ratio = tradable / total if total else 0.0
    print(
        "\nOB Debug",
        f"{instrument}: ob_total={total}",
        f"ob_tradable={tradable}",
        f"ob_nontradable={nontradable}",
        f"tradable_ratio={ratio:.2%}",
    )
    if total and tradable == 0:
        failures = tradability_failure_counts(order_blocks)
        top_three = failures.most_common(3)
        if top_three:
            print("Top tradability failure reasons:")
            for reason, count in top_three:
                print(f"  - {reason}: {count}")


def _run_instrument(path: Path, runs_dir: Path) -> None:
    instrument = _instrument_from_path(path)
    print(f"\n=== {instrument} ({path.name}) ===")
    candles = _load_candles(path)
    result = run_backtest(candles)
    _print_ob_debug(result.order_blocks, instrument)
    label = instrument.lower()

    write_trades_csv(result, runs_dir / f"trades_{label}.csv")
    _write_summaries(result.trades, runs_dir, label)


def main() -> None:
    files = _find_data_files()
    if not files:
        raise SystemExit("No CSV/XLSX files found in /data or ./data")

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    for path in files:
        _run_instrument(path, runs_dir)


if __name__ == "__main__":
    main()
