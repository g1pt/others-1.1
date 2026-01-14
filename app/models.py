from datetime import datetime
from typing import Optional, Union

from pydantic import BaseModel


class TVWebhook(BaseModel):
    secret: str
    symbol: str
    timeframe: Union[int, str]
    timestamp: Optional[str] = None
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


class PaperOrder(BaseModel):
    id: str
    symbol: str
    timeframe: str
    setup_id: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_pct: float
    risk_cash: float
    position_size: float
    status: str
    opened_utc: str
    closed_utc: Optional[str] = None
    pnl_cash: Optional[float] = None
    pnl_r: Optional[float] = None


class EquityState(BaseModel):
    equity: float
    high_watermark: float
    max_dd_pct: float
    daily_date: str
    daily_start_equity: float
    daily_dd_pct: float
    trades_today: int
    consec_losses: int


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
