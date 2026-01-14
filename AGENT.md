# Agent Notes

## Execution Modes
- `LOG_ONLY` (default) accepts valid triggers and logs them, but never creates paper orders.
- `PAPER_SIM` creates simulated orders and updates the ledger based on deterministic outcomes.
- `LIVE_BROKER` is intentionally disabled; the API returns an error if selected.

## Symbol Whitelist
- `SP500` is enabled by default with the `MMXM_4C_D` ruleset.
- `EURUSD` is disabled by default and requires its own ruleset ID (`EURUSD_4C_D`) to be enabled.

## Risk Guardrails
Triggers may be accepted without a trade if risk rules block execution.
