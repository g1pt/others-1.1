# Profitability recheck (2026-02-25)

## Commandos
- `python -m scripts.run_mmxm_research --baseline`
- `python -m scripts.run_mmxm_research --baseline --all-datasets`

## Default baseline (huidige default set)
- Dataset-runs: **2**
- Trade count (totaal): **15**
- Winrate (gewogen op trades): **46.67%**
- Expectancy (gewogen, R/trade): **0.400R**
- Drawdown (gemiddelde max drawdown): **2.50R**
- Drawdown (slechtste max drawdown): **3.00R**
- Drawdown (gemiddelde max_drawdown_pct): **2.46%**
- Drawdown (slechtste max_drawdown_pct): **2.97%**
- Positieve dataset-runs (total_return_pct > 0): **2**
- Negatieve/flat dataset-runs (total_return_pct <= 0): **0**

## All-datasets baseline
- Dataset-runs: **18**
- Trade count (totaal): **140**
- Winrate (gewogen op trades): **60.00%**
- Expectancy (gewogen, R/trade): **0.800R**
- Drawdown (gemiddelde max drawdown): **1.78R**
- Drawdown (slechtste max drawdown): **4.00R**
- Drawdown (gemiddelde max_drawdown_pct): **1.73%**
- Drawdown (slechtste max_drawdown_pct): **3.97%**
- Positieve dataset-runs (total_return_pct > 0): **17**
- Negatieve/flat dataset-runs (total_return_pct <= 0): **1**

## Korte conclusie
- Beide runs blijven positief op gewogen expectancy (>0).
- Default set is momenteel klein (2 datasets, 15 trades) en daardoor minder robuust.
- All-datasets geeft breder beeld (18 datasets, 140 trades) en blijft positief met 0.800R expectancy.
