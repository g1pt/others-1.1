import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = BASE_DIR / "logs"

EXECUTION_MODE = os.getenv("EXECUTION_MODE", "LOG_ONLY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "CHANGE_ME")

SYMBOL_RULESETS = {
    "SP500": {
        "allowed_timeframes": {"30"},
        "require_setup": "MMXM_4C_D",
        "entry_type": "Refinement",
        "phase": "Manipulation",
        "ob_tradability": "Tradable",
        "enabled": True,
    },
    "EURUSD": {
        "allowed_timeframes": {"30"},
        "require_setup": "EURUSD_4C_D",
        "entry_type": "Refinement",
        "phase": "Manipulation",
        "ob_tradability": "Tradable",
        "enabled": False,
    },
}

SYMBOL_MAP = {
    "SP500": "SP500",
    "SP:SPX": "SP500",
    "OANDA:SPX500USD": "SP500",
    "FOREXCOM:SPXUSD": "SP500",
    "FX:SPX500": "SP500",
    "CAPITALCOM:US500": "SP500",
    "EURUSD": "EURUSD",
    "OANDA:EURUSD": "EURUSD",
    "FX:EURUSD": "EURUSD",
    "FOREXCOM:EURUSD": "EURUSD",
    "PEPPERSTONE:EURUSD": "EURUSD",
}


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


INITIAL_EQUITY = _get_float("INITIAL_EQUITY", 10000.0)
RISK_PCT_BASE = _get_float("RISK_PCT_BASE", 0.005)
RISK_PCT_MAX = _get_float("RISK_PCT_MAX", 0.01)
HARD_MAX_DD_PCT = _get_float("HARD_MAX_DD_PCT", 0.03)
DAILY_MAX_DD_PCT = _get_float("DAILY_MAX_DD_PCT", 0.02)
MAX_TRADES_PER_DAY = _get_int("MAX_TRADES_PER_DAY", 1)
MAX_CONSEC_LOSSES = _get_int("MAX_CONSEC_LOSSES", 2)
HIGH_WATERMARK_RISK_REDUCE_PCT = _get_float(
    "HIGH_WATERMARK_RISK_REDUCE_PCT",
    0.005,
)
SPREAD_SLIPPAGE_R_PENALTY = _get_float("SPREAD_SLIPPAGE_R_PENALTY", 0.05)
