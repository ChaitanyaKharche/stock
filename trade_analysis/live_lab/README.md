# Live Lab — prospective paper-trading laboratory

**This package never places an order.** No broker client, no credential, no order-routing
code path. Positions are hypothetical and priced from observed NBBO only.

Specification: [`research/live_lab_specification.md`](../../research/live_lab_specification.md)
Config hash of the current frozen definitions: printed by `--print-config`.

---

## Why this exists

Every number in this project so far is **in-sample and backward-looking**. Nine
pre-registered tests found no predictive edge, and the binding constraint turned out to be
sample size, not method — the trader's own QQQ tape sits at t ≈ 1.0 and needs roughly 4x
more observations to separate from zero.

This lab generates the only thing that can settle it: **a clean prospective record**,
starting from the first trade.

## Run it

```bash
python -m trade_analysis.live_lab.runner --symbols QQQ SPY
```

```bash
python -m trade_analysis.live_lab.replay --days 20
```

```bash
python -m trade_analysis.live_lab.dashboard
```

```bash
python -m trade_analysis.live_lab.dashboard --today
```

The runner is safe to kill with Ctrl-C: it flattens open positions at the last observed
quote, checkpoints, and writes the daily summary. Restarting reloads `positions_open.json`
and resumes — a three-hour crash loses nothing.

## Prerequisites

- **Theta Terminal running on 127.0.0.1:25503.** Set `host = "127.0.0.1"` in its
  `config.toml`; the default binds `0.0.0.0` and exposes a paid feed to your LAN.
- **Stock Data subscription** — required for underlying quotes and bars.
- **Option Data subscription** — required for every contract price. **This cancels
  2026-09-05.** Without it the runner still records signals and the underlying twin, but
  every fill is `SKIPPED: feed`. The options half simply stops.

## The ordering guarantee

The one thing worth reviewing carefully, in `runner._open()`:

```python
signal_id = self.store.write_decision(...)   # fsync'd to disk
# ---- only now may a price be requested ----
chain = self._chain(sym, expiration)
```

The decision is durable on disk **before** any option price is requested. If the process
dies in between, the signal still exists with no knowledge of any price. This is why
`write_decision` is a separate method from `event()` and why it fsyncs.

Supporting invariants:

| rule | where |
|---|---|
| only closed bars are ever admitted (`T + 60s + settle`) | `session.SessionState.accept_bars` |
| a Context cannot hold a bar past its own instant | `session.SessionState.context`, asserted every evaluation in `replay` |
| 5m buckets need all five minutes or they do not exist | `session._rebuild_5m` |
| VWAP is cumulative from 09:30, never full-session | `indicators.session_vwap` + session ownership |
| RVOL is time-of-day normalised | `indicators.rvol_time_of_day` |
| prior-day levels are RTH-only | `feed.minute_bars` filters to 09:30–15:59 |
| setups cannot see each other | each gets only its own `Context`; no shared mutable state |

## What the replay gate proves — and what it does not

It **cannot** prove "no lookahead" by feeding a setup the future, because the architecture
makes that impossible: a `Context` is built from a buffer that ends at the evaluation bar,
so future bars are not merely unused, they are **absent**. The structural assertion checks
that invariant on every evaluation.

What the one-shot comparison catches is the residual class the invariant cannot:
order-dependence and state leaked across calls.

On its first run the gate flagged a mismatch that turned out to be a bug **in the probe** —
1m bars carry the vendor's OPEN stamp while 5m buckets are stamped at their CLOSE, so the
naive truncation handed the probe one extra bar. Worth knowing if you extend it.

## Files written

```
live_lab_data/
  config/<hash>.json     frozen definitions + git sha + spec version
  events.jsonl           append-only, monotonic seq
  signals.jsonl          DECISION / FILL / SKIP — every signal, including rejected
  positions_open.json    atomic-replace snapshot for crash recovery
  trades.jsonl           completed, one line per (signal x strike arm)
  outages.jsonl          disconnects, stale quotes, degraded bars
  daily/<date>.json      session summary
```

Changing any setup parameter changes the **config hash**, which forks the setup into a new
identity with a fresh history. Old trades are never retro-fitted to a new definition.

## Reading results honestly

The dashboard ranks by **expectancy with a CI and n**, never by raw P&L, because setups
fire at wildly different rates.

A setup prints `*** PROMISING ***` only if **all four** hold:

1. n ≥ 200 completed ATM trades
2. clears Holm within the family of 13
3. |effect| exceeds its own achieved MDE
4. first half and second half agree in sign

Anything else prints `INSUFFICIENT` or a specific failure, no matter how good the P&L looks.

**Four setups are labelled `[SLOW]`** — `EMA_9_20_Pullback`, `ThreeBarPlay`,
`PDH_PDL_FailedBreak`, `Gap_Fade`. Their signal rate makes a verdict unreachable in a
reasonable horizon. They are kept and recorded — deleting them is exactly the selective
reporting this design forbids — but an early reading from them is never evidence.

**Roughly three independent bets, not thirteen.** The ORB/Crabel/PDH family are all
early-session directional breakouts; the two fades are correlated; the two
continuation-after-impulse setups are correlated. Correlated positives will look like
confirmation and are not.

## Relationship to `trade_analysis/live_trading/`

**None.** This package imports nothing from it. That package is Alpaca-based, currently
non-functional (`alpaca` is not installed, and its credentials are absent), contains real
`submit_order` calls, and evaluates signals on the **currently-forming bar** — the specific
failure this lab is built to avoid. It is left in place as a record, not reused.
