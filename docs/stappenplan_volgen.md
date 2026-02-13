# Stappenplan (praktisch, volgorde om te volgen)

Dit plan helpt je om **gestructureerd** van setup naar evaluatie te gaan.

## 1) Omgeving klaarzetten
1. Maak/activeer een virtual environment.
2. Installeer dependencies:
   - `pip install -r requirements.txt`
   - (optioneel voor dev) `pip install -r requirements-dev.txt`
3. Draai basischecks:
   - `pytest -q`

## 2) Keuze maken: research of webhook-flow
Kies eerst één primaire focus, zodat je niet alles tegelijk doet:
- **Research-track**: hypotheses testen met backtests.
- **Webhook-track**: signalen valideren via FastAPI + paper logic.

## 3) Research-track uitvoeren
1. Controleer de uitgangspunten in `docs/mmxm_research_spec_v1.md`.
2. Draai een baseline run:
   - `python -m scripts.run_mmxm_research`
3. Vergelijk resultaten met checklist/filters:
   - `docs/mmxm_checklist_tradingplan.md`
   - `docs/codex_filtering_strategy.md`
4. Log per iteratie:
   - welke parameter je wijzigde,
   - impact op expectancy,
   - impact op trade count.

## 4) Webhook-track uitvoeren
1. Zet secret en start API:
   - `WEBHOOK_SECRET=CHANGE_ME uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`
2. Verifieer health endpoint:
   - `curl http://127.0.0.1:8000/health`
3. Stuur een test webhook naar `/webhook/tradingview`.
4. Bevestig dat execution mode/risk guardrails correct reageren (zie `AGENT.md`).

## 5) Risk-instellingen expliciet vastleggen
Voor elke testdag noteer:
- `RISK_MODE`
- `DAILY_RISK_BUDGET_PCT`
- `MIN_RISK_PER_TRADE_PCT`
- `MAX_RISK_PER_TRADE_PCT`

Gebruik telkens dezelfde set tijdens vergelijkingen; pas pas na een volledige evaluatie aan.

## 6) Dagelijkse ritme (aanbevolen)
1. **Start dag**: health + quick smoke test.
2. **Midden dag**: 1 gecontroleerde wijziging.
3. **Einde dag**: resultaten samenvatten in 3 bullets:
   - wat werkte,
   - wat niet werkte,
   - wat je morgen test.

## 7) Definition of Done (wanneer "klaar" is)
Een iteratie is pas klaar als:
- tests groen zijn,
- je wijziging reproduceerbaar is,
- je een korte beslissing hebt: *houden*, *rollback*, of *verder testen*.

## 8) Volgende concrete actie (nu meteen)
Begin met:
1. `pytest -q`
2. `python -m scripts.run_mmxm_research`
3. noteer baseline metrics in 1 kort logbestand.
