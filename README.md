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
- `docs/stappenplan_volgen.md` — concreet stappenplan om setup, research en webhook-flow te volgen (NL).
- `docs/stappenplan_uitvoering.md` — uitgevoerd runverslag van het stappenplan + eventuele blockers (NL).

## CLI
- Run research: `python -m scripts.run_mmxm_research`
- Run tests: `pytest -q`
- Run baseline v2 (fixed allowlist + KPI-gates + OOS report): `python -m scripts.run_baseline_workflow`
  - adjust thresholds: `python -m scripts.run_baseline_workflow --min-trades 100 --min-winrate 0.58 --min-expectancy 0.7 --max-drawdown-r 2.5`
  - override dataset set: `python -m scripts.run_baseline_workflow --allowlist configs/baseline_allowlist_v2.json`
- Run only selected datasets (PowerShell):
  - `@'
[
  "FX_SPX500, 2.csv",
  "FX_SPX500, 15 (1).csv",
  "OANDA_GBPUSD, 15.csv"
]
'@ | Set-Content .\keep_datasets.json`
  - `python -m scripts.run_mmxm_research --all-datasets --live-mode --dataset-allowlist .\keep_datasets.json`
  - alias also supported: `--dataset_allowlist`
  - optional env fallback: set `MMXM_DATASET_ALLOWLIST` to the same JSON path

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
- `DAILY_PROFIT_LOCK_PCT=0.0` (bijv. `0.02` = stop met nieuwe trades na +2% dagwinst)

Als `RISK_MODE=daily_budget`, dan verdeelt de engine het dagbudget over de resterende trade-slots van die dag.

Als `DAILY_PROFIT_LOCK_PCT` > 0, dan blokkeert de engine nieuwe entries zodra de dagwinstdrempel is gehaald.

## SP500 lot-size referentie (CFD)
- Bij veel CFD-brokers geldt vaak:
  - `1.0 lot ≈ $10 per punt`
  - `0.1 lot ≈ $1 per punt`
  - `0.01 lot ≈ $0.10 per punt`
- Dit verschilt per broker/symbool. Controleer altijd in MT5:
  - Right click op symbool → `Specification` → `Contract Size`.


## Waarom deze manier van bouwen sterk is
Deze repo bouwt bewust in kleine, testbare stappen met risk-gates en duidelijke logging. Dat heeft directe voordelen:

- **Minder regressies:** elke wijziging krijgt gerichte tests, zodat bestaande flow stabiel blijft.
- **Sneller leren met minder schade:** nieuwe ideeën eerst in paper-sim + gates, pas daarna opschalen.
- **Profitability beschermen:** drawdown-limieten, loss-streak stops en profit-locks beperken overtrading en winst-terugval.
- **Transparantie:** rejections en trade-events worden gelogd met redenen, waardoor beslissingen uitlegbaar zijn.
- **Snelle tuning:** gedrag is via env/config aanpasbaar (`RISK_MODE`, budget, drawdown, profit-lock) zonder codewijziging.
- **Schaalbare software-opbouw:** losse modules (risk, ledger, engine, reporting) houden het systeem onderhoudbaar.

Praktisch betekent dit: je kan meer vragen, sneller itereren en tegelijk je downside beter beheersen.
