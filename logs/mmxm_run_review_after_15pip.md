# MMXM review after 15-pip/15-point stop update

## Quick verdict
- The run **did not materially improve the baseline block** (`[4C-A]`) versus prior results.
- Baseline still contains multiple negative datasets (e.g. `FX_SPX500 1m`, `FX_SPX500 30m`, `ICMARK 1m`, `ICMARK 30m`).
- So the current implementation likely improves **RR-sweep simulations in specific slices**, but not the baseline path enough to change overall behavior.

## What did improve
- Some RR sweep rows now show reduced damage / near-BE behavior in places where median R shifted from deep negatives toward ~0 in specific markets/timeframes.
- This matches BE/lock logic intent, but effect is **regime-dependent** and not universal.

## Why user still sees weak results
1. Baseline strategy quality dominates outcome on weak datasets.
2. BE/lock helps only if price first moves favorably by trigger distance.
3. Several symbols/timeframes still show persistent negative expectancy even after stop management.

## Recommended next step (high impact)
- Add a **hard market-timeframe allowlist** and only trade combinations that are consistently positive in both:
  - RR expectancy, and
  - early/mid/late robustness.

## Candidate keep-list from this run (best relative pockets)
- `FX_SPX500 2m`
- `FX_SPX500 3m`
- `ICMARK 2m`
- `GBPUSD 2m` and `GBPUSD 5m` remain weak in RR and should be filtered out.

## Actionable implementation suggestion
- Introduce a pre-trade filter config (JSON/YAML) to allow only selected `(symbol, timeframe)` pairs.
- Re-run `--all-datasets --live-mode` and compare:
  - win/loss count,
  - expectancy,
  - max drawdown,
  - equity endpoint.
