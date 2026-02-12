# others-1.1

Research sandbox for ICT backtesting experiments.

## Layout

```
others-1.1/
  src/
    data.py
    sessions.py
    ict_features.py
    backtest.py
    risk.py
    report.py
  runs/
```

## Notes
- `src/ict_features.py` and `src/risk.py` are the primary research modules.
- `src/backtest.py` should stay stable so experiments are isolated.

## Documentation
- `docs/mmxm_checklist_tradingplan.md` — MMXM checklist + tradingplan (NL).
- `docs/mmxm_research_spec_v1.md` — MMXM research specification (v1, NL).
- `docs/smart_money_entry_types_questions.md` — Smart Money Entry Types research questions (NL).
- `docs/codex_filtering_strategy.md` — Codex filter prompt for expectancy/trade-count screening (NL).
- `docs/scalping_bot_roadmap.md` — roadmap voor snellere scalping-bot + testplan (NL).

## CLI
- Run research: `python -m scripts.run_mmxm_research`
- Run tests: `pytest -q`

## TradingView Webhook Receiver (FastAPI)

Install (venv):
  pip install fastapi uvicorn pydantic

Run (cmd):
  set WEBHOOK_SECRET=CHANGE_ME
  uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

Test (PowerShell):
  $body = @{
    secret="CHANGE_ME"
    symbol="SP500"
    timeframe="30"
    timestamp="2026-01-11T20:00:00Z"
    setup="MMXM_4C_D"
    entry_type="Refinement"
    phase="Manipulation"
    ob_tradability="Tradable"
    direction="buy"
    price=5000.0
  } | ConvertTo-Json
  Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/webhook/tradingview" -ContentType "application/json" -Body $body

Health check:
  Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"


## Risk mode (paper engine)
- `RISK_MODE=fixed_per_trade|daily_budget`
- `DAILY_RISK_BUDGET_PCT=0.02` (bijv. 2% equity per dag)
- `MIN_RISK_PER_TRADE_PCT=0.001`
- `MAX_RISK_PER_TRADE_PCT=0.02`

Als `RISK_MODE=daily_budget`, dan verdeelt de engine het dagbudget over de resterende trade-slots van die dag.
