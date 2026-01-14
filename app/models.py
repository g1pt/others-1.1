from typing import Any, Optional, Union

from pydantic import BaseModel


class TVWebhook(BaseModel):
    secret: str
    symbol: str
    timeframe: Union[int, str]
    timestamp: str
    setup: str
    entry_type: str
    phase: str
    ob_tradability: str
    direction: Optional[str] = None
    side: Optional[str] = None
    price: float
    htf_bias: Optional[str] = None
    session: Optional[str] = None
    entry_variant: Optional[str] = None
    level_type: Optional[str] = None
    near_level_dist: Optional[float] = None
    sweep: Optional[str] = None
    bos: Optional[str] = None
    fvg_size: Optional[float] = None
    sim_outcome_r: Optional[float] = None


class Order(BaseModel):
    id: str
    symbol: str
    timeframe: str
    setup: str
    side: str
    entry_price: float
    stop_price: float
    tp_price: float
    risk_pct: float
    qty: float
    status: str
    opened_at: str
    closed_at: Optional[str] = None
    pnl_r: Optional[float] = None
    pnl_cash: Optional[float] = None
    meta: dict[str, Any]


class LedgerState(BaseModel):
    equity: float
    high_watermark: float
    max_dd_pct: float
    daily_date: str
    daily_start_equity: float
    daily_dd_pct: float
    trades_today: int
    consec_losses: int
