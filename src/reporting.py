"""Leakage reporting helpers for research runs."""
from __future__ import annotations

from collections import Counter

from src.models import Trade


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
