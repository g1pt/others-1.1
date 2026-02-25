# Uitvoering stappenplan (runverslag)

Dit bestand legt vast wat er effectief is uitgevoerd uit `docs/stappenplan_volgen.md`.

## Uitgevoerd

### 1) Basischeck tests
Command:

```bash
pytest -q
```

Resultaat:
- **Geslaagd**: `41 passed`.

### 2) Baseline research-run
Command:

```bash
python -m scripts.run_mmxm_research --all-datasets --live-mode
```

Resultaat:
- **Geslaagd**: datasets worden gevonden en verwerkt (o.a. `ICMARKETS_EURUSD` en `OANDA_GBPUSD` files in repo-root).
- In deze run leverde de huidige entry-quality filter (`killzone + impulse + OB-range`) **0 gehouden setups** op per dataset, waardoor de baseline en varianten op **0 trades** uitkwamen.

## Baseline metrics (huidige data + default filters)
- trade count: `0`
- expectancy: `0.000 R`
- winrate: `0.00%`
- max drawdown: `0.00%`
- gebruikte parameterset: standaard instellingen + `--all-datasets --live-mode`

## Conclusie
- De pipeline werkt weer end-to-end met aanwezige data.
- Volgende optimalisatiepunt is filterkalibratie (killzone/impulse/OB-range) om voldoende trade count te krijgen voor statistisch bruikbare evaluatie.

## Volgende actie
1. Draai één iteratie met soepelere filters (bijv. lagere `--min-impulse-pct` en/of `--min-ob-range-pct`).
2. Vergelijk KPI's met deze baseline:
   - trade count,
   - expectancy,
   - max drawdown.
3. Leg per wijziging vast of het effect stabiel blijft over meerdere datasets/timeframes.
