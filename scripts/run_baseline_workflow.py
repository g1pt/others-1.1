"""Reproducible baseline workflow with KPI gates and OOS validation."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import summarize_combinations
from src.backtest import run_backtest
from src.data import load_candles_csv, load_candles_xlsx


@dataclass(frozen=True)
class KPI:
    trades: int
    winrate: float
    expectancy: float
    max_drawdown_r: float


@dataclass(frozen=True)
class KPIGates:
    min_trades: int
    min_winrate: float
    min_expectancy: float
    max_drawdown_r: float


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: float
    threshold: float
    comparator: str


def _load_allowlist(path: Path) -> set[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Allowlist JSON must be an array of dataset names.")
    return {Path(item).stem.strip().lower() for item in raw if isinstance(item, str) and item.strip()}


def _data_roots() -> list[Path]:
    roots = [Path("/data"), ROOT / "DATA", ROOT / "data", ROOT]
    return [p for p in roots if p.exists()]


def _discover_files(allowlist: set[str]) -> list[Path]:
    files: list[Path] = []
    for root in _data_roots():
        files.extend(sorted(root.glob("*.csv")))
        files.extend(sorted(root.glob("*.xlsx")))
    selected = [p for p in files if p.stem.strip().lower() in allowlist]
    return sorted(set(selected))


def _load_candles(path: Path):
    if path.suffix.lower() == ".csv":
        return load_candles_csv(path)
    if path.suffix.lower() == ".xlsx":
        return load_candles_xlsx(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def _pnls_from_trades(trades: Iterable) -> list[float]:
    return [float(t.pnl_r or 0.0) for t in trades]


def _max_drawdown_r(pnls: list[float]) -> float:
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        eq += pnl
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    return max_dd


def kpi_from_trades(trades: list) -> KPI:
    pnls = _pnls_from_trades(trades)
    trades_n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    winrate = (wins / trades_n) if trades_n else 0.0
    expectancy = (sum(pnls) / trades_n) if trades_n else 0.0
    return KPI(
        trades=trades_n,
        winrate=winrate,
        expectancy=expectancy,
        max_drawdown_r=_max_drawdown_r(pnls),
    )


def evaluate_gates(kpi: KPI, gates: KPIGates) -> list[GateResult]:
    results = [
        GateResult("min_trades", kpi.trades >= gates.min_trades, float(kpi.trades), float(gates.min_trades), ">="),
        GateResult("min_winrate", kpi.winrate >= gates.min_winrate, kpi.winrate, gates.min_winrate, ">="),
        GateResult("min_expectancy", kpi.expectancy >= gates.min_expectancy, kpi.expectancy, gates.min_expectancy, ">="),
        GateResult("max_drawdown_r", kpi.max_drawdown_r <= gates.max_drawdown_r, kpi.max_drawdown_r, gates.max_drawdown_r, "<="),
    ]
    return results


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def _fmt_r(value: float) -> str:
    return f"{value:.3f}R"


def _split_candles(candles: list, train_ratio: float) -> tuple[list, list]:
    split_idx = max(1, min(len(candles) - 1, int(len(candles) * train_ratio)))
    return candles[:split_idx], candles[split_idx:]


def _dataset_row(path: Path, trades: list) -> str:
    combo = summarize_combinations(trades)
    top = combo[0] if combo else None
    kpi = kpi_from_trades(trades)
    top_key = top.key.replace("Combo:", "") if top else "n/a"
    return (
        f"| {path.name} | {kpi.trades} | {_fmt_pct(kpi.winrate)} | {_fmt_r(kpi.expectancy)} | "
        f"{kpi.max_drawdown_r:.2f} | {top_key} |"
    )


def run(args: argparse.Namespace) -> Path:
    allowlist = _load_allowlist(Path(args.allowlist))
    datasets = _discover_files(allowlist)
    if not datasets:
        raise SystemExit("No datasets matched allowlist.")

    all_trades = []
    in_sample_trades = []
    out_sample_trades = []
    dataset_rows = []

    for path in datasets:
        candles = _load_candles(path)
        full_result = run_backtest(candles)
        all_trades.extend(full_result.trades)
        dataset_rows.append(_dataset_row(path, full_result.trades))

        train_candles, test_candles = _split_candles(candles, args.train_ratio)
        in_sample_trades.extend(run_backtest(train_candles).trades)
        out_sample_trades.extend(run_backtest(test_candles).trades)

    baseline_kpi = kpi_from_trades(all_trades)
    in_kpi = kpi_from_trades(in_sample_trades)
    out_kpi = kpi_from_trades(out_sample_trades)

    gates = KPIGates(
        min_trades=args.min_trades,
        min_winrate=args.min_winrate,
        min_expectancy=args.min_expectancy,
        max_drawdown_r=args.max_drawdown_r,
    )
    gate_results = evaluate_gates(baseline_kpi, gates)
    passed = all(r.passed for r in gate_results)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d")
    out_path = Path(args.output or f"logs/baseline_v2_report_{timestamp}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Baseline V2 report ({timestamp})",
        "",
        "## 1) Reproduceerbare baseline",
        f"- Allowlist: `{args.allowlist}`",
        f"- Datasets gevonden: **{len(datasets)}**",
        "",
        "## 2) KPI-gates",
        f"- Min trades: **{gates.min_trades}**",
        f"- Min winrate: **{_fmt_pct(gates.min_winrate)}**",
        f"- Min expectancy: **{_fmt_r(gates.min_expectancy)}**",
        f"- Max drawdown: **{gates.max_drawdown_r:.2f}R**",
        "",
        "### Baseline KPI (alle trades)",
        f"- Trades: **{baseline_kpi.trades}**",
        f"- Winrate: **{_fmt_pct(baseline_kpi.winrate)}**",
        f"- Expectancy: **{_fmt_r(baseline_kpi.expectancy)}**",
        f"- Max drawdown: **{baseline_kpi.max_drawdown_r:.2f}R**",
        "",
        "### Gate-evaluatie",
    ]
    for gate in gate_results:
        icon = "✅" if gate.passed else "❌"
        if "winrate" in gate.name:
            value = _fmt_pct(gate.value)
            threshold = _fmt_pct(gate.threshold)
        elif "expectancy" in gate.name:
            value = _fmt_r(gate.value)
            threshold = _fmt_r(gate.threshold)
        else:
            value = f"{gate.value:.2f}" if gate.name == "max_drawdown_r" else f"{gate.value:.0f}"
            threshold = f"{gate.threshold:.2f}" if gate.name == "max_drawdown_r" else f"{gate.threshold:.0f}"
        lines.append(f"- {icon} `{gate.name}`: {value} {gate.comparator} {threshold}")

    lines.extend(
        [
            "",
            "## 3) OOS-validatie (chronologische split)",
            f"- Split: **{int(args.train_ratio * 100)}% / {int((1 - args.train_ratio) * 100)}%**",
            f"- In-sample: trades={in_kpi.trades}, winrate={_fmt_pct(in_kpi.winrate)}, expectancy={_fmt_r(in_kpi.expectancy)}, max_dd={in_kpi.max_drawdown_r:.2f}R",
            f"- Out-of-sample: trades={out_kpi.trades}, winrate={_fmt_pct(out_kpi.winrate)}, expectancy={_fmt_r(out_kpi.expectancy)}, max_dd={out_kpi.max_drawdown_r:.2f}R",
            "",
            "## 4) Beslisformat",
            f"- Status: **{'GO' if passed else 'NO-GO'}**",
            (
                "- Advies: door naar volgende iteratie met dezelfde filters."
                if passed
                else "- Advies: filters/risico aanscherpen en baseline opnieuw draaien."
            ),
            "",
            "## Per-dataset KPI",
            "| Dataset | Trades | Winrate | Expectancy | Max DD (R) | Top combo |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    lines.extend(dataset_rows)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written: {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline v2 with gates + OOS report")
    parser.add_argument("--allowlist", default="configs/baseline_allowlist_v2.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--min-trades", type=int, default=80)
    parser.add_argument("--min-winrate", type=float, default=0.55)
    parser.add_argument("--min-expectancy", type=float, default=0.50)
    parser.add_argument("--max-drawdown-r", type=float, default=3.00)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
