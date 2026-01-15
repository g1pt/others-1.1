"""Trade state machine for paper execution."""
from __future__ import annotations

from dataclasses import dataclass

from .models import TradeStatus


@dataclass(frozen=True)
class TradeStateMachine:
    """Enforces allowed trade status transitions."""

    transitions: dict[TradeStatus, set[TradeStatus]] = None

    def __post_init__(self) -> None:
        if self.transitions is None:
            object.__setattr__(
                self,
                "transitions",
                {
                    TradeStatus.NEW: {
                        TradeStatus.OPEN,
                        TradeStatus.INVALID,
                        TradeStatus.REJECTED,
                    },
                    TradeStatus.OPEN: {
                        TradeStatus.CLOSED,
                        TradeStatus.CLOSED_TP,
                        TradeStatus.CLOSED_SL,
                        TradeStatus.CLOSED_TIME,
                    },
                    TradeStatus.CLOSED: set(),
                    TradeStatus.CLOSED_TP: set(),
                    TradeStatus.CLOSED_SL: set(),
                    TradeStatus.CLOSED_TIME: set(),
                    TradeStatus.INVALID: set(),
                    TradeStatus.REJECTED: set(),
                },
            )

    def can_transition(self, current: TradeStatus, target: TradeStatus) -> bool:
        return target in self.transitions.get(current, set())

    def transition(self, current: TradeStatus, target: TradeStatus) -> TradeStatus:
        if not self.can_transition(current, target):
            raise ValueError(f"Invalid trade status transition: {current} -> {target}")
        return target
