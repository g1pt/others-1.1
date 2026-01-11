from pydantic import BaseModel


class TVWebhook(BaseModel):
    secret: str
    symbol: str
    timeframe: str
    timestamp: str
    setup: str
    entry_type: str
    phase: str
    ob_tradability: str
    direction: str
    price: float
