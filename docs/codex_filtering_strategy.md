# Codex filter: positieve expectancy en ≥20 trades

Deze notitie beschrijft een filter voor Codex-analyses die alleen statistisch zinvolle
strategiecombinaties doorlaat. De kern is: positieve expectancy, minimaal 20 trades,
en aanvullende risicofilters op drawdown en equity-curve stabiliteit.

## Kerncriteria

* **Expectancy > 0** (verwachte opbrengst per trade is positief).
* **Trades ≥ 20** (minimale steekproef voor betrouwbaarheid).
* **Max drawdown < 10%** (beperkt diepe equity-dalingen).
* **Stabiliteit > 0,4** (voorkeur voor gelijkmatige equity-groei).

## Codex prompt

Gebruik onderstaande prompt om combinaties te filteren en te rapporteren.

```python
"""
Deze prompt filtert combinaties op basis van minimale betrouwbaarheidseisen
voor strategietests: positieve expectancy, voldoende trades, lage drawdown,
en stabiele equity-curve. Kan worden gebruikt voor automatische analyse
door Codex.
"""

prompt = """
Analyseer alle combinaties met Risk Entry. Filter combinaties met:
- Positieve expectancy (expectancy > 0)
- Minimaal 20 trades (trades >= 20)

Toon per geldige combinatie de volgende statistieken:
- Winrate (in %)
- Expectancy (in R of %)
- Max drawdown (in %)
- Stabiliteitsscore (tussen 0 en 1)

Bouw vervolgens een beslisregel die alleen combinaties toelaat waarbij:
- Drawdown < 10
- Stabiliteit > 0.4

Markeer combinaties die falen met een toelichting waarom (bijv. stabiliteit te laag).
Geef een aparte lijst van toegelaten combinaties.

Outputformaat: gestructureerde JSON met velden:
[
  {
    "combo": "Phase|Entry|OBType",
    "expectancy": float,
    "drawdown": float,
    "stability": float,
    "winrate": float,
    "trades": int,
    "valid": true/false,
    "reason": "..."
  }
]
"""
```
