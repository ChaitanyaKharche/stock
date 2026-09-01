"""Live paper-trading laboratory.

A prospective, out-of-sample research instrument for QQQ/SPY 0DTE options.

**This package never places an order.** There is no broker client, no credential, and no
order-routing code path anywhere in it. Positions are hypothetical and priced from
observed NBBO only. That is a design boundary, not a phase-1 limitation.

It imports nothing from `trade_analysis.live_trading` (an Alpaca-based package that is
non-functional and evaluates signals on the currently-forming bar) and nothing from
`trade_analysis.data_sources`, so a change to the historical research client cannot
silently alter live behaviour.

Modules:
    indicators  pure, dependency-free indicator maths
    feed        ThetaData live + historical access, with outage handling
    session     bar admission (closed bars only) and the immutable evaluation Context
    setups      the 13 frozen setups
    options     ATM / ATM-1 / ATM+1 selection, derived IV and delta
    positions   hypothetical positions, exits, MFE/MAE, the underlying twin
    store       append-only JSONL with the decision-before-price fsync guarantee
    runner      the live loop
    replay      offline lookahead probe and signal-frequency calibration
    dashboard   read-only reporting

Specification: research/live_lab_specification.md
"""

__all__ = ["indicators", "feed", "session", "setups", "options",
           "positions", "store", "runner", "replay", "dashboard"]

PLACES_ORDERS = False
