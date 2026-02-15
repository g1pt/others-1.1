import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

EXECUTION_MODE = os.getenv("EXECUTION_MODE", "LOG_ONLY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "CHANGE_ME")


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
RISK_PCT_DEFAULT = _get_float("RISK_PCT_DEFAULT", 0.005)
RISK_MODE = os.getenv("RISK_MODE", "fixed_per_trade")
DAILY_RISK_BUDGET_PCT = _get_float("DAILY_RISK_BUDGET_PCT", 0.02)
MIN_RISK_PER_TRADE_PCT = _get_float("MIN_RISK_PER_TRADE_PCT", 0.001)
MAX_RISK_PER_TRADE_PCT = _get_float("MAX_RISK_PER_TRADE_PCT", 0.02)
RISK_PCT_MAX = _get_float("RISK_PCT_MAX", 0.01)
DD_SOFT = _get_float("DD_SOFT", 0.02)
REDUCED_RISK_PCT = _get_float("REDUCED_RISK_PCT", 0.0025)
DD_DAILY_STOP = _get_float("DD_DAILY_STOP", 0.02)
DD_HARD_SYSTEM = _get_float("DD_HARD_SYSTEM", 0.03)
MAX_TRADES_PER_DAY = _get_int("MAX_TRADES_PER_DAY", 1)
MAX_CONSEC_LOSSES = _get_int("MAX_CONSEC_LOSSES", 2)
RR_DEFAULT = _get_float("RR_DEFAULT", 2.0)
SL_DISTANCE_MODE = os.getenv("SL_DISTANCE_MODE", "fixed_pct")
SL_FIXED_PCT = _get_float("SL_FIXED_PCT", 0.002)

SYMBOL_MAP = {
    "FX_SPX500": "SP500",
    "SPX500": "SP500",
    "SP500": "SP500",
    "EURUSD": "EURUSD",
}

DEFAULT_EQUITY = _get_float("DEFAULT_EQUITY", 10000.0)
DEFAULT_RISK_PER_TRADE = _get_float("DEFAULT_RISK_PER_TRADE", 0.005)
DEFAULT_RR = _get_float("DEFAULT_RR", 2.0)
DEFAULT_SL_PCT = _get_float("DEFAULT_SL_PCT", 0.002)

BE_TRIGGER_EURUSD = _get_float("BE_TRIGGER_EURUSD", 0.0010)
BE_TRIGGER_SP500 = _get_float("BE_TRIGGER_SP500", 6.0)

# Two-stage protective stop policy.
# FX defaults: +10 pips -> BE, +15 pips -> lock +5 pips.
BE_TRIGGER_FX = _get_float("BE_TRIGGER_FX", 0.0010)
BE_LOCK_TRIGGER_FX = _get_float("BE_LOCK_TRIGGER_FX", 0.0015)
BE_LOCK_OFFSET_FX = _get_float("BE_LOCK_OFFSET_FX", 0.0005)

# Index defaults (points): +10 -> BE, +15 -> lock +5.
BE_TRIGGER_INDEX = _get_float("BE_TRIGGER_INDEX", 10.0)
BE_LOCK_TRIGGER_INDEX = _get_float("BE_LOCK_TRIGGER_INDEX", 15.0)
BE_LOCK_OFFSET_INDEX = _get_float("BE_LOCK_OFFSET_INDEX", 5.0)
