"""Session time helpers (London/NY)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start_hour: int
    end_hour: int


LONDON = SessionWindow(name="London", start_hour=7, end_hour=16)
NEW_YORK = SessionWindow(name="New York", start_hour=12, end_hour=21)


def is_in_session(timestamp_hour: int, session: SessionWindow) -> bool:
    """Return True if the timestamp hour is inside the session window."""
    return session.start_hour <= timestamp_hour < session.end_hour
