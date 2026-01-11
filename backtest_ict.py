"""Run gated ICT backtest variants with equity simulation guards."""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import run_backtest  # noqa: E402
from src.data import load_candles_csv, load_candles_xlsx  # noqa: E402
from src.models import Candle, Trade  # noqa: E402
from src.reporting import leakage_contribution_report  # noqa: E402
from src.risk import GuardedEquitySimulation, simulate_equity_with_guards  # noqa: E402

RISK_PCT_MIN = 0.01
RISK_PCT_MAX = 0.03


@dataclass(frozen=True)
class GatingConfig:
    allow_entry_types: list[str] | None
    require_tradable_ob: bool
    allow_phases: list[str] | None


@dataclass(frozen=True)
class Variant:
    code: str
    label: str
    gating: GatingConfig


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


def _dataset_path(symbol: str, timeframe: int) -> Path:
    normalized = symbol.lower()
    for root in _data_roots():
        for suffix in ("csv", "xlsx"):
            candidate = root / f"{normalized}, {timeframe}.{suffix}"
            if candidate.exists():
                return candidate
    pattern = re.compile(rf"{re.escape(normalized)}.*{timeframe}", re.IGNORECASE)
    for root in _data_roots():
        for candidate in root.glob("*.*"):
            if pattern.search(candidate.stem):
                return candidate
    raise FileNotFoundError(
        f"Unable to locate dataset for {symbol} {timeframe}m in /data or data."
    )


def _load_candles(path: Path) -> list[Candle]:
    if path.suffix.lower() == ".csv":
        return load_candles_csv(path)
    if path.suffix.lower() == ".xlsx":
        return load_candles_xlsx(path)
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def _normalize_label(value: str | None) -> str:
    return str(value or "").strip().lower()


def _apply_gating(trades: list[Trade], gating: GatingConfig) -> list[Trade]:
    allowed_entries = (
        {_normalize_label(entry) for entry in gating.allow_entry_types}
        if gating.allow_entry_types
        else None
    )
    allowed_phases = (
        {_normalize_label(phase) for phase in gating.allow_phases}
        if gating.allow_phases
        else None
    )
    filtered: list[Trade] = []
    for trade in trades:
        entry_type = _normalize_label(trade.entry_method)
        if entry_type == _normalize_label("Continuation Entry"):
            continue
        if allowed_entries is not None and entry_type not in allowed_entries:
            continue
        if gating.require_tradable_ob and trade.ob_tradable is not True:
            continue
        if allowed_phases is not None:
            phase = _normalize_label(trade.mmxm_phase)
            if phase not in allowed_phases:
                continue
        filtered.append(trade)
    return filtered


def _clamp_risk_pct(value: float) -> float:
    return max(RISK_PCT_MIN, min(RISK_PCT_MAX, value))


def _variant_matrix() -> list[Variant]:
    return [
        Variant(
            code="A",
            label="Baseline gated",
            gating=GatingConfig(
                allow_entry_types=["Refinement Entry"],
                require_tradable_ob=True,
                allow_phases=None,
            ),
        ),
        Variant(
            code="B",
            label="Baseline gated + Manipulation",
            gating=GatingConfig(
                allow_entry_types=["Refinement Entry"],
                require_tradable_ob=True,
                allow_phases=["Manipulation"],
            ),
        ),
        Variant(
            code="C",
            label="Risk+Refinement",
            gating=GatingConfig(
                allow_entry_types=["Refinement Entry", "Risk Entry"],
                require_tradable_ob=True,
                allow_phases=None,
            ),
        ),
        Variant(
            code="D",
            label="Risk+Refinement + Manipulation",
            gating=GatingConfig(
                allow_entry_types=["Refinement Entry", "Risk Entry"],
                require_tradable_ob=True,
                allow_phases=["Manipulation"],
            ),
        ),
        Variant(
            code="E",
            label="Confirmation test",
            gating=GatingConfig(
                allow_entry_types=[
                    "Refinement Entry",
                    "Risk Entry",
                    "Confirmation Entry",
                ],
                require_tradable_ob=True,
                allow_phases=["Manipulation"],
            ),
        ),
    ]


def _maybe_add_custom_variant(
    variants: list[Variant],
    *,
    allow_entry_types: list[str] | None,
    require_tradable_ob: bool,
    allow_phases: list[str] | None,
) -> None:
    if allow_entry_types is None and allow_phases is None and not require_tradable_ob:
        return
    variants.append(
        Variant(
            code="Custom",
            label="Custom gating",
            gating=GatingConfig(
                allow_entry_types=allow_entry_types,
                require_tradable_ob=require_tradable_ob,
                allow_phases=allow_phases,
            ),
        )
    )


def _print_variant_header(symbol: str, timeframe: int, variant: Variant) -> None:
    print(
        "\n"
        f"=== {symbol.upper()} {timeframe}m | Variant {variant.code}: {variant.label} ==="
    )


def _summarize_guarded(simulation: GuardedEquitySimulation) -> None:
    result = simulation.simulation
    print(
        "results: "
        f"trades_taken={len(result.trade_results)}, "
        f"end_equity={result.final_equity:.2f}, "
        f"total_return_pct={result.return_pct:.2f}%, "
        f"max_drawdown_pct={result.max_drawdown_pct:.2f}%, "
        f"stopped_by={simulation.stopped_by}"
    )


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Invalid {name} env var value: {raw}") from None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Invalid {name} env var value: {raw}") from None


def _run_sanity_checks() -> None:
    assert _clamp_risk_pct(0.005) == RISK_PCT_MIN
    assert _clamp_risk_pct(0.04) == RISK_PCT_MAX
    assert _clamp_risk_pct(0.02) == 0.02

    dd_trades = [
        Trade(
            entry_time="2024-01-01 09:00",
            entry_price=100.0,
            stop_price=90.0,
            direction="bullish",
            mmxm_phase="Expansion",
            entry_method="Risk Entry",
            ob_tradable=True,
            ob_id=1,
            exit_time="2024-01-01 10:00",
            exit_price=90.0,
            pnl_r=-2.0,
        ),
        Trade(
            entry_time="2024-01-01 11:00",
            entry_price=100.0,
            stop_price=90.0,
            direction="bullish",
            mmxm_phase="Expansion",
            entry_method="Risk Entry",
            ob_tradable=True,
            ob_id=2,
            exit_time="2024-01-01 12:00",
            exit_price=110.0,
            pnl_r=1.0,
        ),
    ]
    dd_sim = simulate_equity_with_guards(
        dd_trades,
        initial_equity=10_000.0,
        risk_pct=0.02,
        max_dd_pct=0.03,
        daily_loss_limit_pct=0.0,
        max_trades_per_day=0,
    )
    assert dd_sim.stopped_by == "max_dd"
    assert len(dd_sim.simulation.trade_results) == 1

    daily_trades = [
        Trade(
            entry_time="2024-01-01 09:00",
            entry_price=100.0,
            stop_price=90.0,
            direction="bullish",
            mmxm_phase="Expansion",
            entry_method="Risk Entry",
            ob_tradable=True,
            ob_id=1,
            exit_time="2024-01-01 10:00",
            exit_price=90.0,
            pnl_r=-1.0,
        ),
        Trade(
            entry_time="2024-01-01 11:00",
            entry_price=100.0,
            stop_price=90.0,
            direction="bullish",
            mmxm_phase="Expansion",
            entry_method="Risk Entry",
            ob_tradable=True,
            ob_id=2,
            exit_time="2024-01-01 12:00",
            exit_price=110.0,
            pnl_r=1.0,
        ),
        Trade(
            entry_time="2024-01-02 09:00",
            entry_price=100.0,
            stop_price=90.0,
            direction="bullish",
            mmxm_phase="Expansion",
            entry_method="Risk Entry",
            ob_tradable=True,
            ob_id=3,
            exit_time="2024-01-02 10:00",
            exit_price=110.0,
            pnl_r=1.0,
        ),
    ]
    daily_sim = simulate_equity_with_guards(
        daily_trades,
        initial_equity=10_000.0,
        risk_pct=0.02,
        max_dd_pct=0.5,
        daily_loss_limit_pct=0.02,
        max_trades_per_day=0,
    )
    assert daily_sim.stopped_by == "daily_loss"
    assert len(daily_sim.simulation.trade_results) == 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gated ICT backtest variants.")
    parser.add_argument("--symbol", default="FX_SPX500", help="Instrument symbol.")
    parser.add_argument(
        "--tfs",
        type=int,
        nargs="+",
        default=[15, 30],
        help="Timeframes in minutes.",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=_env_float("INITIAL_EQUITY", 10_000.0),
        help="Starting equity for equity simulation.",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=_env_float("RISK_PCT", 0.02),
        help="Risk percent per trade (fraction).",
    )
    parser.add_argument(
        "--max-dd",
        type=float,
        default=_env_float("MAX_DD_PCT", 0.03),
        help="Max peak-to-trough drawdown percent (fraction).",
    )
    parser.add_argument(
        "--daily-loss",
        type=float,
        default=_env_float("DAILY_LOSS_LIMIT_PCT", 0.02),
        help="Daily loss limit percent (fraction). Use 0 to disable.",
    )
    parser.add_argument(
        "--max-trades-day",
        type=int,
        default=_env_int("MAX_TRADES_PER_DAY", 3),
        help="Max trades per day. Use 0 to disable.",
    )
    parser.add_argument(
        "--allow-entry-types",
        nargs="+",
        default=None,
        help="Optional custom allowed entry types.",
    )
    parser.add_argument(
        "--allow-phases",
        nargs="+",
        default=None,
        help="Optional custom allowed phases.",
    )
    parser.add_argument(
        "--require-tradable-ob",
        action="store_true",
        help="Require tradable order blocks for the custom variant.",
    )
    args = parser.parse_args()

    _run_sanity_checks()

    risk_pct = _clamp_risk_pct(args.risk)
    if risk_pct != args.risk:
        print(f"Clamped risk_pct from {args.risk:.4f} to {risk_pct:.4f}.")

    variants = _variant_matrix()
    _maybe_add_custom_variant(
        variants,
        allow_entry_types=args.allow_entry_types,
        require_tradable_ob=args.require_tradable_ob,
        allow_phases=args.allow_phases,
    )

    for timeframe in args.tfs:
        path = _dataset_path(args.symbol, timeframe)
        candles = _load_candles(path)
        result = run_backtest(candles, timeframe=f"{timeframe}m")

        for variant in variants:
            _print_variant_header(args.symbol, timeframe, variant)
            gated_trades = _apply_gating(result.trades, variant.gating)
            simulation = simulate_equity_with_guards(
                gated_trades,
                initial_equity=args.initial_equity,
                risk_pct=risk_pct,
                max_dd_pct=args.max_dd,
                daily_loss_limit_pct=args.daily_loss,
                max_trades_per_day=args.max_trades_day,
            )
            _summarize_guarded(simulation)
            print()
            print(leakage_contribution_report(simulation.simulation.trade_results))


if __name__ == "__main__":
    main()
