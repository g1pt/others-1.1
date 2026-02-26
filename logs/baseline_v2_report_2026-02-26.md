# Baseline V2 report (2026-02-26)

## 1) Reproduceerbare baseline
- Allowlist: `configs/baseline_allowlist_v2.json`
- Datasets gevonden: **12**

## 2) KPI-gates
- Min trades: **80**
- Min winrate: **55.00%**
- Min expectancy: **0.500R**
- Max drawdown: **3.00R**

### Baseline KPI (alle trades)
- Trades: **89**
- Winrate: **64.04%**
- Expectancy: **0.921R**
- Max drawdown: **3.00R**

### Gate-evaluatie
- ✅ `min_trades`: 89 >= 80
- ✅ `min_winrate`: 64.04% >= 55.00%
- ✅ `min_expectancy`: 0.921R >= 0.500R
- ✅ `max_drawdown_r`: 3.00 <= 3.00

## 3) OOS-validatie (chronologische split)
- Split: **70% / 30%**
- In-sample: trades=68, winrate=61.76%, expectancy=0.853R, max_dd=3.00R
- Out-of-sample: trades=17, winrate=82.35%, expectancy=1.471R, max_dd=2.00R

## 4) Beslisformat
- Status: **GO**
- Advies: door naar volgende iteratie met dezelfde filters.

## Per-dataset KPI
| Dataset | Trades | Winrate | Expectancy | Max DD (R) | Top combo |
|---|---:|---:|---:|---:|---|
| ICMARKETS_EURUSD, 1.csv | 8 | 50.00% | 0.500R | 3.00 | Manipulation|Refinement Entry|Tradable |
| ICMARKETS_EURUSD, 15 (1).csv | 5 | 60.00% | 0.800R | 1.00 | Manipulation|Refinement Entry|Tradable |
| ICMARKETS_EURUSD, 2.csv | 3 | 100.00% | 2.000R | 0.00 | Manipulation|Refinement Entry|Tradable |
| ICMARKETS_EURUSD, 3.csv | 10 | 70.00% | 1.100R | 1.00 | Manipulation|Refinement Entry|Tradable |
| ICMARKETS_EURUSD, 30 (1).csv | 3 | 33.33% | 0.000R | 2.00 | Manipulation|Refinement Entry|Tradable |
| ICMARKETS_EURUSD, 5 (1).csv | 11 | 45.45% | 0.364R | 3.00 | Manipulation|Refinement Entry|Tradable |
| OANDA_GBPUSD, 1.csv | 6 | 66.67% | 1.000R | 2.00 | Manipulation|Refinement Entry|Tradable |
| OANDA_GBPUSD, 15.csv | 8 | 87.50% | 1.625R | 1.00 | Manipulation|Refinement Entry|Tradable |
| OANDA_GBPUSD, 2.csv | 5 | 100.00% | 2.000R | 0.00 | Manipulation|Refinement Entry|Tradable |
| OANDA_GBPUSD, 3.csv | 11 | 54.55% | 0.636R | 3.00 | Manipulation|Refinement Entry|Tradable |
| OANDA_GBPUSD, 30.csv | 9 | 55.56% | 0.667R | 1.00 | Manipulation|Refinement Entry|Tradable |
| OANDA_GBPUSD, 5.csv | 10 | 70.00% | 1.100R | 2.00 | Manipulation|Refinement Entry|Tradable |
