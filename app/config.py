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
