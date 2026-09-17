# Live lab — 50/100-trade DIAGNOSTIC checkpoint

**Run 2026-09-16. Diagnostic only: pipeline health, data quality, outages. No effect
claims.** `FREEZE.json` and `forward_test_preregistration.md` §3 define both checkpoints in
those words; the first serious evaluation is **200 per setup**, where the bar is four
conditions, not one. Nothing here ranks, tests, promotes or drops a setup.

Reproduce with `python -m trade_analysis.live_lab.checkpoint_diagnostic`.

## It is late, and that is part of the record

Both checkpoints were passed without being recorded. The primary (ATM) arm has closed
**173** trades — 517 across all three strike arms — so 50 and 100 are long behind us. This
is the same pattern as `momo_v2.py` sitting pre-registered and unrun for nine days: the
checkpoint was specified and then not executed. Recorded so the omission is visible rather
than tidied away.

## Verdict on the apparatus

| check | result |
|---|---|
| provenance, both arms | **PASS** |
| pre-registered `bars_failed` pooling guard | **PASS** |
| summary-vs-log reconciliation | **FAIL — 4 sessions** |
| fills skipped, both arms, whole record | **0** |

The primary records are intact and authoritative. The failures are all in the **summary
layer**, and none of them lose data.

## 1. Provenance — PASS

Every session ran under a hash the freeze accepts.

- **Options:** 9 sessions, all `1f7247d7839d9950`.
- **Shares:** 11 sessions — 4 under `53229d6f1f24df10` (QQQ+SPY), 7 under `b53ca8a58aa11718`
  (15 names). Both accepted; the pooling justification is in `shares/FREEZE.json`.
- No session precedes either arm's `start_date`.

## 2. The pre-registered pooling guard — PASS

`shares/FREEZE.json` names its own falsification condition: pooling the two shares hashes is
justified only while the feed is not saturating, and *"a sustained rise in `bars_failed`
outages would break this justification and must be treated as such."*

Per session: `09-10: 63`, `09-14: 17`, `09-16: 12`, every other session **0**. Earlier half
12.6/session, recent half 7.2/session — **no sustained rise**. The pooling justification
holds.

But the shape is worth noting: `bars_failed` is **bursty, not trending**. One session
carries 63 of 92. **2026-09-10 is also the shares arm's worst session (−$1,734.28)**, and
both facts sit on the same day. That is not evidence of causation — 09-09 lost $1,339.77
with zero `bars_failed` — but it is the pairing to watch if it recurs.

## 3. Coverage

- **Options: a 4-session hole — 2026-09-08, 09-09, 09-10, 09-11.** The span 08-28 → 09-16
  is **9 sessions, not 14**, and must never be quoted as continuous.
- **Shares: no gaps** across 09-01 → 09-16 (11 sessions).
- 2026-09-07 is Labor Day and is correctly absent from both; the diagnostic carries an NYSE
  holiday list so a closed market is not reported as missing data.

## 4. Summary-vs-log reconciliation — FAIL on 4 sessions

The `daily/*.json` summaries disagree with `trades.jsonl` on four sessions. Two distinct
causes, both benign for the data and both dangerous for anyone reading the summaries.

### 4a. A schema change mid-experiment — 3 sessions

Sessions **2026-09-01, 09-02, 09-03** record a field named **`net`**; every other session
records **`atm_net`**. They are not the same quantity:

| date | field | recorded | true ATM | all 3 arms |
|---|---|---|---|---|
| 09-01 | `net` | −1,907.88 | **−614.29** | −1,907.88 |
| 09-02 | `net` | −365.60 | **−126.53** | −365.60 |
| 09-03 | `net` | +3,659.68 | **+1,246.23** | +3,659.68 |

`net` equals the **sum over all three strike arms**, exactly. `FREEZE.json` sets
`"primary_arm": "ATM"`, so any tool summing "the daily net" across the record mixes
primary-arm figures with ~3× inflated all-arms ones. **On 09-03 alone that is +$2,413 of
phantom profit.**

Nothing is lost — `trades.jsonl` carries per-arm records and the true ATM values are exactly
recoverable. **The rule that follows: sum `trades.jsonl` filtered by arm; never sum the
daily summaries.**

The first version of the diagnostic itself fell for this, via
`r.get("atm_net", r.get("net", ...))`. That fallback silently substitutes a *different
quantity* when the first key is absent, and printed an all-arms number in an ATM column with
no warning. Same silent-fallback shape as the NaN placebo in the sweep audit — the code had
an answer for the missing case and the answer was wrong. The diagnostic now computes from
the log and *checks* the summary against it, never the reverse.

### 4b. Summaries sealed before EOD reconstruction — 1 session

Shares **2026-09-04**: summary `net` = −92.82, log = −103.84. The $11.02 gap is exactly
three trades with `exit_reason = "eod_reconstructed"` (−5.57, +0.91, −6.36) appended after
the daily record was written. **`trades.jsonl` is the more complete record**, which is the
reassuring direction.

## 5. `eod_reconstructed` — the finding that matters most

Some positions are closed not by a live EOD flatten but by a later reconstruction pass.
Counts: **options ATM 10, shares 13.** They are concentrated:

| session | options ATM | shares |
|---|---|---|
| 2026-09-02 | −$8.32 | +$15.23 |
| **2026-09-03** | **+$1,474.52** | **+$324.77** |
| 2026-09-04 | — | −$11.02 |

**2026-09-03 is the options arm's best session (+$1,246.23 ATM), and $1,474.52 of it sits in
six reconstructed exits.** Without them the session is **−$228.29**.

**The marks themselves look economically correct.** The largest, a QQQ 713 call entered at
09:40 for 1.65 and marked 4.48 at 15:55: the underlying finished at 717.57, so intrinsic is
4.57. A 0DTE ATM call finishing $4.57 ITM *should* mark near 4.5x. +171% is not an artifact;
it is what that instrument does on a trend day. The reconstruction did not manufacture the
P&L — the positions really were held all day into a rally.

**But reconstructed exits are not interchangeable with live ones, and the difference leans
one way:**

| | last live bid | reconstructed | diff |
|---|---|---|---|
| mean over 10 trades | | | **+$0.282** |
| direction | | | **9 of 10 favour the position** |

Total mark effect **+$282** on an arm whose cumulative is −$675.97.

This is mechanically expected rather than sinister: the arm is always long premium, the last
live bid is a *stale* observation from before the loop stopped, and both affected sessions
rallied into the close, so a later mark is legitimately higher. And the 10 trades span only
two sessions, so 9-of-10 is roughly n≈2, not n=10 — the apparent significance is illusory.

### Why they needed reconstructing — answered, and already fixed

This checkpoint originally left "why did six positions need reconstructing at all?" as an
open thread. **It was not open.** `reconstruct_eod.py` documents the cause in its own
header:

> `autostart.supervise()` terminated both arms at 15:55:02 ET -- the exact minute they were
> due to flatten.

A supervisor race killed the arms at precisely the minute `_flatten_all` was due to run, so
every position still alive at the close was never written. The event log confirms the chain
exactly: **09-04 09:35:07 `recovery_rejected`, 18 orphaned positions from 09-03** — and
18 ÷ 3 strike arms = the 6 ATM trades.

**Fixed.** `autostart.py:71` now sets `RUNNER_DONE_AFTER = 15:55` with a comment recording
that terminating at 15:55 "destroyed the EOD flatten and the daily summary for three
straight" sessions, plus `HARD_STOP_GRACE_SEC = 240` so a waking runner flattens before it
can be killed. Sessions 09-14/15/16 show live `eod` exits (3, 7, 1) and zero reconstructions.

**The reconstruction is the remedy, not the defect** — and the header names the reason it
mattered so much:

> The loss is NOT random. It removes exactly one exit type: `eod` ... A hole that deletes
> one exit category biases the sample far more than a larger random one would.

That is the correct framing and it is stronger than the one this document reached for.

### The residue: 2026-09-01 has a permanent hole

`reconstruct_eod.py` states that 09-01's snapshot "was overwritten by the following session
before the archive existed. Those positions are gone permanently and that session keeps its
hole." The exit table confirms it: **09-01 has zero `eod` AND zero `eod_reconstructed`.**

So one session in the frozen record is missing an entire exit category with no way to
recover it. Per the tool's own argument that is worse than a random hole of the same size.
It cannot be excluded — the clock rule forbids dropping a session — so it must simply be
**carried as a known defect and stated whenever 09-01 is included**.

| session | `eod` | reconstructed | status |
|---|---|---|---|
| 2026-08-28 | 0 | 0 | 2 `shutdown` (day 1) |
| 2026-09-01 | 0 | 0 | **permanent hole, unrecoverable** |
| 2026-09-02 | 0 | 4 | supervisor race, reconstructed |
| 2026-09-03 | 0 | 6 | supervisor race, reconstructed |
| 2026-09-04 | 0 | 0 | 3 `shutdown`, feed down at close |
| 2026-09-14/15/16 | 3 / 7 / 1 | 0 | live flatten working |

## 6. Funnel and outages

- **`fills_skipped` = 0 across the entire record, both arms.** Every decided signal became a
  trade, so there is no fill-selection channel to worry about.
- **`degraded_bars`: options 0, shares 25.**
- Shares outage mix: `stale_bar` 2,108, `feed` 591, `tick` 272, `bars_failed` 92,
  `future_quote` 45, `degraded_bar` 25, `quote_failed` 22, `stale_quote` 20.
- Feed retry rate is low and stable except **shares 09-04 (27.65/1k)** and **09-16
  (10.68/1k)**.

## 7. What is explicitly NOT claimed here

No setup is ranked, tested, promoted or dropped. P&L appears because §3 says reading it is
"not forbidden — pretending otherwise would be theatre", but no conclusion is drawn from it.

**The first serious evaluation is 200 per setup.** MOMO_CHASE shares stands at 193/200 and
will likely cross within one session. When it does, the bar is all four conditions of §4 —
n ≥ 200, Holm within the family of 13, effect exceeding its own achieved MDE, and split-half
sign agreement — not a P&L reading.

## 8. Actions

1. **Never sum `daily/*.json` nets.** Sum `trades.jsonl` filtered by `arm`. §4a.
2. ~~Investigate why 09-03 left six positions unflattened.~~ **Answered and already
   fixed** — a supervisor race terminated the arms at the 15:55 flatten minute; see §5.
   The live successor: **2026-09-01 carries a permanent, unrecoverable hole in the `eod`
   exit category** and must be flagged wherever it is included.
3. **Quote the options arm as 9 sessions with a 4-session hole**, never as 08-28 → 09-16
   continuous. §3.
4. Watch `bars_failed` against shares session quality if 09-10's pairing recurs. §2.
