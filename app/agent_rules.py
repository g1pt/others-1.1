SYMBOL_RULESETS = {
    "SP500": {
        "allowed_timeframes": {"30"},
        "setup_id": "MMXM_4C_D",
        "entry_type": "Refinement",
        "phase": "Manipulation",
        "ob_tradability": "Tradable",
        "enabled": True,
    },
    "EURUSD": {
        "allowed_timeframes": {"30"},
        "setup_id": "EURUSD_4C_D",
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
