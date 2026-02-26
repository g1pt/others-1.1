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
- Time-stop voor scalps (sneller kapitaal vrijmaken, lagere kapitaalblokkade).
- Partial TP + break-even policy verder uitbouwen.
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

## Eerstvolgende concrete sprint (aanbevolen volgorde)
Om de roadmap praktisch uitvoerbaar te maken én testbaar op te bouwen, nemen we de volgende volgorde:

1. **Partial TP mechaniek afronden**
   - definieer schaal-uit regels (bijv. 50% op TP1, rest op TP2 of BE-trailing),
   - eenduidige ledger-boekingen per partial fill,
   - compatibel maken met bestaande time-stop.
2. **Volatility-aware SL/TP toevoegen**
   - routeer SL/TP-afstanden op basis van volatiliteitsregime (bijv. ATR buckets),
   - behoud harde risk caps zodat risicopercentage nooit buiten limiet valt,
   - monitor impact op gemiddelde R-multiple en duur per trade.
3. **Daarna pas uitgebreide validatie (walk-forward + Monte Carlo)**
   - eerst stabiliteit van execution-logic borgen,
   - daarna robuustheid testen op sequence-risico en regimeverschuivingen.



## Chart-gedreven testritme (blijven testen)
Omdat de marktstructuur en liquiditeitsdynamiek continu verschuiven, blijft chart-validatie onderdeel van elke iteratie:

1. **Weekly chart review-set samenstellen**
   - per instrument een vaste set voorbeelden van win/loss/no-trade,
   - label HTF-context (D1/H4 bias), liquiditeitspool-type en sessie.
2. **Na elke strategy-wijziging opnieuw replayen**
   - controleer of dezelfde situaties nu consistenter worden afgehandeld,
   - log expliciet waar de engine te vroeg, te laat of onterecht niet instapt.
3. **Koppel chart-observaties terug naar filters/templates**
   - vertaal patronen naar parameter- of template-aanpassingen,
   - geen losse meningen: alleen wijzigingen met herhaalbare chart-evidence.

## Edge-adaptatie: Thinker → Trainer → Trader laag
Om de edge niet statisch te houden, bouwen we een expliciete feedback-loop tussen analyse, training en uitvoering.

### 1) Thinker (hypothese-laag)
- Detecteert waar de edge afbrokkelt (regime-shift, afwijkende liquiditeit, sessiegedrag).
- Formuleert wijzigingsvoorstellen als **templates**:
  - entry-template,
  - HTF-confirmatie-template,
  - management-template (partial/BE/SL-TP routing).

### 2) Trainer (validatie-laag)
- Test template-varianten op recente én historische slices.
- Rankt varianten op expectancy, drawdown, stability en trade-count.
- Markeert alleen varianten als deployable wanneer validatiegates gehaald worden.

### 3) Trader (executie-laag)
- Draait uitsluitend met vrijgegeven templates/versies.
- Logt per trade welke template actief was (traceability naar PnL).
- Valt terug op laatst stabiele template bij degradatie-signaal.

### 4) AI deploy-agent (governance)
- Beheert versiepromotie van `candidate` → `staging` → `live-paper`.
- Forceert rollback-regels bij overschrijding van DD- of kwaliteitsdrempels.
- Houdt changelog bij: welke edge-aanpassing, waarom, en met welk effect.

## HTF + liquiditeit als verplichte communicatielaag
Voor jouw doel (“alles rond edge laten communiceren”) maken we deze velden verplicht in research én execution logging:
- HTF-bias (bijv. bullish/bearish/neutral),
- type liquiditeitsevent (equal highs/lows sweep, session high/low raid, enz.),
- entry-type + bevestigingsstatus,
- actieve template-versie,
- uitkomst in R en close-reason.

Zo kun je achteraf niet alleen zien **of** een trade won/verloor, maar ook **welke edge-aanname** werkte of faalde in een veranderende markt.


## Testing-focus voor deze sprint (belangrijk)
Dit onderdeel is kritisch: **eerst correctheid, daarna optimalisatie**.

### A) Functionele tests (deterministisch)
- Partial TP:
  - partial exit triggert op juiste level,
  - resterende positie-size klopt,
  - realized/unrealized PnL wordt correct bijgewerkt,
  - break-even activatie gebeurt alleen na geldige partial TP.
- Volatility routing:
  - juiste bucket-selectie per volatiliteitsniveau,
  - SL/TP-afstanden binnen ingestelde min/max-risicobounds,
  - fallback-gedrag als volatiliteitsinput ontbreekt.

### B) Integratietests (research + paper simulatie)
- Vergelijk baseline (zonder nieuwe regels) versus nieuwe trade-management regels op:
  - expectancy,
  - max drawdown,
  - profit factor,
  - gemiddelde trade-duur,
  - equity-curve stabiliteit per week.

### C) Validatiegates vóór Fase 4
Ga pas door naar uitgebreide walk-forward/Monte Carlo wanneer:
- trade accounting 100% consistent is,
- regressietests groen blijven op bestaande datasets,
- verbeteringen niet alleen uit incidentele lucky runs komen.
