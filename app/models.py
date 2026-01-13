from typing import Union

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
    direction: str
    price: float
