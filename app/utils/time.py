from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_date_str(now: datetime | None = None) -> str:
    current = now or utc_now()
    return current.date().isoformat()
