from dataclasses import dataclass

from scripts.run_baseline_workflow import KPI, KPIGates, _max_drawdown_r, evaluate_gates, kpi_from_trades


@dataclass
class _Trade:
    pnl_r: float


def test_max_drawdown_r_computes_peak_to_trough_drop() -> None:
    pnls = [1.0, 1.0, -0.5, -2.0, 1.0]
    assert _max_drawdown_r(pnls) == 2.5


def test_kpi_from_trades_uses_pnl_r_distribution() -> None:
    trades = [_Trade(1.0), _Trade(-1.0), _Trade(2.0), _Trade(0.0)]
    kpi = kpi_from_trades(trades)
    assert kpi.trades == 4
    assert kpi.winrate == 0.5
    assert kpi.expectancy == 0.5


def test_evaluate_gates_flags_failures() -> None:
    kpi = KPI(trades=75, winrate=0.52, expectancy=0.4, max_drawdown_r=3.2)
    gates = KPIGates(min_trades=80, min_winrate=0.55, min_expectancy=0.5, max_drawdown_r=3.0)
    results = evaluate_gates(kpi, gates)
    assert all(not r.passed for r in results)
