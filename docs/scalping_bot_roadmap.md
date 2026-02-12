# Scalping bot roadmap (others-1.1 backbone)

## Doel
Van de huidige webhook + paper-engine backbone een snellere, robuuste scalping-bot maken met beter risicobeheer op **dagbudget-niveau (2% equity per dag)** in plaats van alleen vast risico per trade.

## Fase 1 — Basis hardening (nu)
- Voeg configureerbare risicomodus toe:
  - `fixed_per_trade` (huidig gedrag)
  - `daily_budget` (nieuw): verdeelt dagbudget over resterende trade-slots.
- Nieuwe env variabelen:
  - `RISK_MODE`
  - `DAILY_RISK_BUDGET_PCT`
  - `MIN_RISK_PER_TRADE_PCT`
  - `MAX_RISK_PER_TRADE_PCT`
- Gebruik bestaande safety rails:
  - max trades/day
  - stop na verliesreeks
  - daily drawdown stop
  - hard max drawdown stop

## Fase 2 — Entry kwaliteit omhoog
- Filter alleen high-probability condities:
  - sessie/killzone filter
  - minimale impulssterkte
  - minimale OB-kwaliteit
- Variant tests per entry-type (`Risk`, `Refinement`, `Confirmation`) met expectancy + drawdown ranking.

## Fase 3 — Trade management versnellen
- Time-stop voor scalps (sneller kapitaal vrijmaken).
- Partial TP + break-even policy.
- Volatility-aware SL/TP routing.

## Fase 4 — Validatie & optimalisatie
- Walk-forward backtests op meerdere perioden.
- Monte Carlo shuffle op trade-sequence.
- Parameter sweep:
  - `DAILY_RISK_BUDGET_PCT` (bijv. 1.0% / 1.5% / 2.0% / 2.5%)
  - `MAX_TRADES_PER_DAY`
  - entry gating combinaties

## Aanbevolen testprotocol
1. **Baseline** draaien met `fixed_per_trade` (huidige setup).
2. **Daily budget mode** aanzetten met 2% dagbudget.
3. Vergelijk KPI's:
   - expectancy
   - max drawdown
   - winrate
   - profit factor
   - stability per week
4. Kies profiel:
   - Conservatief: 1.0–1.5% dagbudget
   - Gebalanceerd: 2.0% dagbudget
   - Agressiever: 2.5% dagbudget + striktere stop-after-losses

## Praktische startconfig (voor jouw vraag)
- `RISK_MODE=daily_budget`
- `DAILY_RISK_BUDGET_PCT=0.02`  (2% van equity per dag)
- `MIN_RISK_PER_TRADE_PCT=0.003`
- `MAX_RISK_PER_TRADE_PCT=0.01`
- `MAX_TRADES_PER_DAY=3`
- `MAX_CONSEC_LOSSES=2`

Dit geeft ruimte om soms iets meer risico te nemen, maar begrensd en met daglimiet-controle.
