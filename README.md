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

## CLI
- Run research: `python -m scripts.run_mmxm_research`
- Run tests: `pytest -q`

## TradingView Webhook Receiver (FastAPI)

Install (venv):
  pip install fastapi uvicorn pydantic

Run (cmd):
  set WEBHOOK_SECRET=CHANGE_ME
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

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
