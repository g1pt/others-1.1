from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from uuid import uuid4


@dataclass
class Order:
    id: str
    received_utc: datetime
    symbol: str
    timeframe: str
    setup: str
    direction: str
    entry_price: float
    sl: float
    tp: float
    rr: float
    sl_pct: float
    risk_per_trade: float
    equity_snapshot: float
    size: float
    status: str = "OPEN"
    source: str = "tradingview"

    @classmethod
    def create(
        cls,
        *,
        received_utc: datetime,
        symbol: str,
        timeframe: str,
        setup: str,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        rr: float,
        sl_pct: float,
        risk_per_trade: float,
        equity_snapshot: float,
        size: float,
    ) -> "Order":
        return cls(
            id=str(uuid4()),
            received_utc=received_utc,
            symbol=symbol,
            timeframe=timeframe,
            setup=setup,
            direction=direction,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            rr=rr,
            sl_pct=sl_pct,
            risk_per_trade=risk_per_trade,
            equity_snapshot=equity_snapshot,
            size=size,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["received_utc"] = self.received_utc.isoformat()
        return payload
