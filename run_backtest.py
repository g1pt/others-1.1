"""Entry script to run the MMXM/Smart Money backtest."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis import summarize, summarize_combinations
from src.backtest import run_backtest
from src.config import INITIAL_EQUITY, RISK_PER_TRADE
from src.data import generate_synthetic_candles, load_candles_csv
from src.filtering import load_combo_filter
from src.report import write_equity_log_csv, write_summary_csv, write_trades_csv
from src.reporting import leakage_contribution_report, leakage_report
from src.risk import simulate_equity


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
    parser = argparse.ArgumentParser(description="Run the MMXM/Smart Money backtest.")
    parser.add_argument(
        "--combo-filter",
        type=Path,
        help="Path to a JSON file containing combo summaries for whitelist filtering.",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=INITIAL_EQUITY,
        help="Starting equity for the equity curve simulation.",
    )
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=RISK_PER_TRADE,
        help="Risk per trade as a fraction of equity.",
    )
    args = parser.parse_args()

    data_path = Path("data/synthetic_candles.csv")
    _ensure_sample_data(data_path)
    candles = load_candles_csv(data_path)
    combo_filter = load_combo_filter(args.combo_filter) if args.combo_filter else None
    result = run_backtest(candles, combo_filter=combo_filter)

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

    print()
    print(leakage_report(result.trades))

    equity = simulate_equity(
        result.trades,
        initial_equity=args.initial_equity,
        risk_per_trade=args.risk_per_trade,
        risk_policy="throttle",
        drawdown_trigger=0.05,
        risk_floor=0.005,
        risk_ceiling=0.01,
    )
    write_equity_log_csv(equity, runs_dir / "equity_log.csv")
    print("\nEquity Curve Summary")
    print(f"initial_equity={equity.initial_equity:.2f}")
    print(f"final_equity={equity.final_equity:.2f}")
    print(f"return_pct={equity.return_pct:.2f}%")
    print(f"max_drawdown_pct={equity.max_drawdown_pct:.2f}%")
    print(f"max_drawdown_currency={equity.max_drawdown_currency:.2f}")
    if equity.skipped_trades:
        print(f"skipped_trades={len(equity.skipped_trades)}")

    print()
    print(leakage_contribution_report(equity.trade_results))
