"""Leakage reporting helpers for research runs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.models import Trade
from src.risk import TradeResult


def leakage_report(trades: list[Trade]) -> str:
    """Return a formatted leakage report for a set of trades."""
    total = len(trades)

    entry_counts = Counter(_normalize_entry(trade) for trade in trades)
    phase_counts = Counter(_normalize_phase(trade) for trade in trades)
    tradable_counts = Counter(_normalize_tradable(trade) for trade in trades)

    lines = ["Leakage Check", f"total_trades={total}", "", "By EntryType"]
    lines.extend(_format_counts(entry_counts, total))

    lines.extend(["", "By OB Tradability"])
    lines.extend(_format_counts(tradable_counts, total, order=_TRADABLE_ORDER))

    lines.extend(["", "By Phase"])
    lines.extend(_format_counts(phase_counts, total))
    manipulation = sum(
        count
        for phase, count in phase_counts.items()
        if phase.lower() == "manipulation"
    )
    lines.append(f"Manipulation vs Rest: {manipulation} vs {total - manipulation}")

    lines.extend(["", "Cross: Phase x EntryType"])
    lines.extend(_format_cross(_cross_counts(trades, _normalize_phase, _normalize_entry)))

    lines.extend(["", "Cross: Tradable x EntryType"])
    lines.extend(
        _format_cross(_cross_counts(trades, _normalize_tradable, _normalize_entry))
    )

    lines.extend(["", "Cross: Phase x Tradable"])
    lines.extend(
        _format_cross(_cross_counts(trades, _normalize_phase, _normalize_tradable))
    )

    return "\n".join(lines)


@dataclass(frozen=True)
class ContributionRow:
    key: str
    trades: int
    total_pnl: float
    total_r: float | None
    expectancy: float
    contribution_pct: float


@dataclass(frozen=True)
class ContributionBucket:
    title: str
    rows: list[ContributionRow]


def leakage_contribution_stats(
    trade_results: list[TradeResult],
) -> list[ContributionBucket]:
    total_pnl = sum(result.pnl for result in trade_results)
    has_r = any(result.pnl_r is not None for result in trade_results)
    buckets = [
        ("By EntryType", lambda trade: _normalize_entry(trade)),
        ("By OB Tradability", lambda trade: _normalize_tradable(trade)),
        ("By Phase", lambda trade: _normalize_phase(trade)),
        (
            "By Phase x EntryType",
            lambda trade: f"{_normalize_phase(trade)} | {_normalize_entry(trade)}",
        ),
        (
            "By Tradable x EntryType",
            lambda trade: f"{_normalize_tradable(trade)} | {_normalize_entry(trade)}",
        ),
    ]

    bucket_rows: list[ContributionBucket] = []
    for title, key_fn in buckets:
        grouped: dict[str, list[TradeResult]] = {}
        for result in trade_results:
            key = key_fn(result.trade)
            grouped.setdefault(key, []).append(result)
        rows: list[ContributionRow] = []
        for key, results in grouped.items():
            trades = len(results)
            pnl = sum(result.pnl for result in results)
            total_r = sum(result.pnl_r or 0.0 for result in results) if has_r else None
            expectancy = pnl / trades if trades else 0.0
            contribution_pct = (pnl / total_pnl * 100) if total_pnl else 0.0
            rows.append(
                ContributionRow(
                    key=key,
                    trades=trades,
                    total_pnl=pnl,
                    total_r=total_r,
                    expectancy=expectancy,
                    contribution_pct=contribution_pct,
                )
            )
        rows.sort(key=lambda row: (-row.total_pnl, row.key))
        bucket_rows.append(ContributionBucket(title=title, rows=rows))
    return bucket_rows


def leakage_contribution_report(trade_results: list[TradeResult]) -> str:
    stats = leakage_contribution_stats(trade_results)
    has_r = any(result.pnl_r is not None for result in trade_results)
    lines = ["Leakage Contribution Report"]
    header = (
        "key | trades | total_pnl | total_R | expectancy | contribution%"
        if has_r
        else "key | trades | total_pnl | expectancy | contribution%"
    )

    for bucket in stats:
        lines.extend(["", bucket.title, header])
        if not bucket.rows:
            lines.append("No trades")
            continue
        for row in bucket.rows:
            if has_r:
                total_r = f"{row.total_r:.2f}" if row.total_r is not None else "NA"
                lines.append(
                    f"{row.key} | {row.trades} | {row.total_pnl:.2f} | "
                    f"{total_r} | {row.expectancy:.2f} | {row.contribution_pct:.1f}%"
                )
            else:
                lines.append(
                    f"{row.key} | {row.trades} | {row.total_pnl:.2f} | "
                    f"{row.expectancy:.2f} | {row.contribution_pct:.1f}%"
                )
    return "\n".join(lines)


_TRADABLE_ORDER = ("Tradable", "NonTradable", "Unknown")


def _normalize_entry(trade: Trade) -> str:
    raw = getattr(trade, "entry_method", None)
    value = str(raw).strip() if raw else "Unknown"
    return value or "Unknown"


def _normalize_phase(trade: Trade) -> str:
    raw = getattr(trade, "mmxm_phase", None)
    value = str(raw).strip() if raw else "Unknown"
    return value or "Unknown"


def _normalize_tradable(trade: Trade) -> str:
    raw = getattr(trade, "ob_tradable", None)
    if raw is True:
        return "Tradable"
    if raw is False:
        return "NonTradable"
    return "Unknown"


def _format_counts(
    counts: Counter[str],
    total: int,
    *,
    order: tuple[str, ...] | None = None,
) -> list[str]:
    entries = list(counts.items())
    if order:
        entries = [(key, counts.get(key, 0)) for key in order]
        extras = [pair for pair in counts.items() if pair[0] not in order]
        entries.extend(sorted(extras, key=lambda item: (-item[1], item[0])))
    else:
        entries = sorted(entries, key=lambda item: (-item[1], item[0]))
    return [
        f"{key}: {count} ({_percent(count, total):.1f}%)"
        for key, count in entries
    ]


def _percent(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total * 100


def _cross_counts(
    trades: list[Trade],
    left_fn,
    right_fn,
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for trade in trades:
        left = left_fn(trade)
        right = right_fn(trade)
        counts[(left, right)] += 1
    return counts


def _format_cross(counts: Counter[tuple[str, str]]) -> list[str]:
    lines = []
    for (left, right), count in sorted(
        counts.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        lines.append(f"{left} | {right}: {count}")
    if not lines:
        lines.append("No trades")
    return lines
