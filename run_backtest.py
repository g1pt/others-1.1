"""Entry script to run the MMXM/Smart Money backtest."""
from __future__ import annotations

from pathlib import Path

from src.analysis import summarize, summarize_combinations
from src.backtest import run_backtest
from src.data import generate_synthetic_candles, load_candles_csv
from src.report import write_summary_csv, write_trades_csv


def _ensure_sample_data(path: Path) -> None:
    if path.exists():
        return
    candles = generate_synthetic_candles()
    lines = ["timestamp,open,high,low,close,volume"]
    for candle in candles:
        lines.append(
            f"{candle.timestamp},{candle.open:.5f},{candle.high:.5f},"
            f"{candle.low:.5f},{candle.close:.5f},{candle.volume or ''}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data_path = Path("data/synthetic_candles.csv")
    _ensure_sample_data(data_path)
    candles = load_candles_csv(data_path)
    result = run_backtest(candles)

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    write_trades_csv(result, runs_dir / "trades.csv")

    by_phase = summarize(result.trades, lambda t: t.mmxm_phase, "Phase")
    by_entry = summarize(result.trades, lambda t: t.entry_method, "Entry")
    by_ob = summarize(
        result.trades, lambda t: "Tradable" if t.ob_tradable else "NonTradable", "OB"
    )
    by_combo = summarize_combinations(result.trades)

    write_summary_csv(by_phase, runs_dir / "summary_by_phase.csv")
    write_summary_csv(by_entry, runs_dir / "summary_by_entry.csv")
    write_summary_csv(by_ob, runs_dir / "summary_by_ob.csv")
    write_summary_csv(by_combo, runs_dir / "summary_by_combination.csv")

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

