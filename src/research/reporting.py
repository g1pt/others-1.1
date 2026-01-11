"""Reporting utilities for MMXM research output."""
from __future__ import annotations

from src.analysis import SummaryRow, rank_summaries, select_candidates
from src.models import OrderBlock


def ob_failure_counts(order_blocks: list[OrderBlock]) -> dict[str, int]:
    counts = {
        "total": 0,
        "tradable": 0,
        "nontradable": 0,
        "no_fvg": 0,
        "no_bos": 0,
        "not_near_level": 0,
    }
    for ob in order_blocks:
        counts["total"] += 1
        if not ob.fail_reasons:
            counts["tradable"] += 1
        else:
            counts["nontradable"] += 1
        for reason in ob.fail_reasons:
            if reason in counts:
                counts[reason] += 1
    return counts


def top_candidates(
    rows: list[SummaryRow],
    *,
    min_trades: int = 25,
    min_expectancy: float = 0.0,
    max_dd: float = 12,
    top_n: int = 10,
) -> list[SummaryRow]:
    filtered = select_candidates(
        rows,
        min_trades=min_trades,
        min_expectancy=min_expectancy,
        max_dd=max_dd,
    )
    ranked = rank_summaries(filtered)
    return ranked[:top_n]
