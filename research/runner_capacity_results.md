# Results — Runner capacity, and broadening the shares arm to 15 names

Executed 2026-09-05. Code: `trade_analysis/live_lab/feed.py`,
`shares_runner.py`, `bar_cache_test.py`, `backfill_daily.py`.
No strategy definition changed. No setup was added, removed or tuned.

**Headline: the feed loop was the binding constraint on everything, and it is gone.
Sustained capacity went from ~5 symbols to ~29, the shares arm now runs 15 names, and the
80 trades already collected are kept rather than discarded.**

---

## 1. The problem was two constraints wearing one name

`spread_survey_results.md` identified the feed loop as a prerequisite for broadening and
proposed a per-minute bar cache. That was half right. There are two independent limits and
the cache only fixes one:

**SUSTAINED** — wire seconds consumed per minute of wall clock. A 1-minute bar changes once
a minute; the loop refetched the whole session every 5-second poll. At 477–756 ms/call that
is 11.45 s of wire time per symbol per minute, so **five symbols saturate the loop
completely**. Past that the backlog is unbounded — which is exactly the 40-minute stale
fills of 2026-09-01 that cost ~$500 across both arms, not a jitter.

**PEAK** — the cost of the one tick a minute when the new bar lands. **The cache does not
help here at all**, and this is the part the survey missed: every symbol waits on the *same*
minute boundary, so their refetches collide on a single tick. Sequentially that tick costs
N × 2 × 477 ms — at 15 symbols the last name in the list fills ~14 s after its bar closed.
These are momentum entries, so a delayed fill is *adversely selected*, not merely noisy.

Only fixing both makes 15 symbols safe. The cache alone would have moved the failure from
"permanently behind" to "14 seconds late every minute", which is quieter but still eats a
meaningful slice of a 3.34 bp edge.

## 2. The cache is an equality, not an approximation

`feed._serve_from_cache` serves cached bars **only when doing so is provably identical to
refetching**. A bar stamped T covers [T, T+60s) and `SessionState.bar_is_complete` refuses
it before T + 60s + settle, so at wall clock `now` the newest admissible bar is stamped
`floor_minute(now − 60s)`. If the cache reaches that stamp it holds every bar the session
could admit, and `accept_bars` sees an identical set.

Written as an invariant rather than a timer, two behaviours fall out free:
a bar the vendor publishes **late** is polled for on every tick until it arrives; and during
a network outage the cache **stops answering within 60 s**, so it cannot hide a dead link.
It deliberately ignores `settle` rather than adding it, which keeps the guarantee one-sided
and true for any `settle_ms >= 0`.

## 3. Proof, not assertion

`bar_cache_test.py` replays a full session tick-by-tick through two independent
`SessionState`s — one on the uncached path, one on the cached path — and asserts the
admitted sequences match **on stamp and on tick**. Same tick matters more than same set:
admission time is fill time, so a bar admitted one tick later is a different trade at a
different price. The simulated vendor is hostile on purpose: per-bar delays, one bar
published 95 s late, two never published, and a 10-minute link outage.

```
vendor lag    wire calls   cached   saved   calls/min   identical
0-0  s              4753      534     89%        1.37       YES
0-5  s              4753      911     81%        2.33       YES
0-15 s              4753     1281     73%        3.28       YES
0-30 s              4753     1851     61%        4.74       YES
```

Both arms also produce identical `degraded_bars` (11:18 from the late bar, 13:42 from the
halt) and identical 5-minute buckets. **Config hashes are unchanged:
options `1f7247d7839d9950`, shares-at-2-symbols `53229d6f1f24df10`.**

## 4. Measured against the real gateway

15 symbols, best of 3, warm — this is a **measurement, not a model**:

```
workers    1      2      4      6      8      12     15
seconds    6.76   3.38   2.40   2.09   1.59   1.93   2.55
```

**It gets worse past 8.** The Theta Terminal is one local Java process and begins contending
with itself, so sizing the pool to the universe would have slowed the tick down. `workers=8`
is now the documented default with that table beside it.

```
                                          uncached    cached
wire seconds per symbol per minute           11.45s     2.04s
sustained symbol capacity                       5.2      29.4
boundary tick at 15 symbols, sequential       14.31s
boundary tick at 15 symbols, fanned out                  1.59s
```

Per-symbol failures are isolated: one unreachable symbol no longer blanks the other
fourteen, which is what the old bare loop did by propagating the first exception.

## 5. The fork costs nothing — a claim I had made and had not checked

The survey said widening the universe *"forks the hash and restarts the shares count at
zero"*. The hash does fork (`53229d6f1f24df10` → `b53ca8a58aa11718`). **The count does
not restart.** The shares arm has no cross-symbol coupling anywhere:

- `max_per_day` counts against `(setup_id, symbol)`
- `max_per_direction` against `(setup_id, symbol, direction)`
- the one-open-per-setup rule filters `p.symbol == sym` before collecting setup ids
- there is no global position cap and no shared capital constraint

So a QQQ signal is evaluated, capped and filled identically whether the universe holds 2
names or 15. QQQ trades under either hash are draws from the same distribution and pool
validly. **The 80 trades already collected are kept.**

The residual coupling is operational, not definitional: the symbols share one feed
connection. Measured sustained load at 15 symbols is 51%, and per-symbol failures are
isolated — but a sustained rise in `bars_failed` outages would void this justification, and
`FREEZE.json` says so explicitly.

## 6. Five defects found along the way

**The autostart gate would have aborted every session from 2026-09-08.** This is the
serious one. `autostart` decides whether to abort by reading preflight's stdout, and
classified a line as non-fatal with `"REACHABLE from" not in line` — a match against the
check's *English*. Commit `6985258` (2026-09-04 16:25, after that day's session had
finished) rewrote the exposure check to query Windows Firewall instead of self-connecting,
changing the wording to `"N inbound ALLOW rule(s) match …"`. Nothing failed and nothing
warned; the string simply stopped matching, so a deliberately non-blocking security warning
became `ABORT: structural preflight failed`. Every session from the next trading day would
have been silently lost.

Compounding it, preflight's `--ignore-exposure` flag had been declared since the beginning
and **never read** — `fails += line.startswith(FAIL)` counted exposure regardless — so the
one mechanism that should have prevented this did nothing.

Fixed three ways: preflight now honours the flag, emits a stable `EXPOSURE:` token instead
of relying on prose, and `autostart` both passes `--ignore-exposure` and keys on the token.
`preflight_gate_test.py` asserts the contract, including that the *live* exposure check
still emits the token — the assertion that would have caught this.

**A quote timestamped in the FUTURE passed the staleness check.** The test was one-sided
(`age > STALE_QUOTE_SEC`), so a quote stamped ahead of the local clock had a negative age
and sailed through. Live this can only mean clock skew, and this lab is unusually exposed:
the machine runs MST with no DST and every reading is converted to ET through a fixed
offset in `clock.py`, so a wrong offset presents *exactly* as future-dated quotes. The
failure is silent and total — every fill priced off a quote from another time. Now rejected
and logged as a `future_quote` outage in both arms. Surfaced by the smoke test, which
filled 148 trades at closed-market spreads (XLI bid 159.04 / ask 191.49 — **1,694 bp**)
and lost $104k in a simulated session; with the guard it correctly takes zero.

For contrast, live spreads on the current names are median **0.28 bp**, max 0.83, with zero
fills above 10 bp — so this cannot happen during RTH. No spread filter was added, and
deliberately: the backtest fills at real historical NBBO and pays whatever the spread was,
so a live-only filter would break the very comparison the forward test exists to make.

**The shares arm had no freeze manifest at all.** `live_lab_data/shares/FREEZE.json` did not
exist, so `dashboard.build` applied *no* config filter and the clock rule was unenforced on
the one arm with a live positive setup. Written now, at the moment the universe widens,
rather than later when the choice could be shaped by results.

**The shares arm could not be reported by its own dashboard.** `dashboard.py` already
special-cased `arm != "SHARES"` for fill-per-signal accounting, but `SHARES` was missing
from the argparse choices, so the command errored out. Also fixed a hardcoded `"total ATM
trades"` label that printed `ATM` whatever arm was being reported.

**Session coverage misdiagnosed three sessions as "never ran".** `autostart` tees one log
per day for both arms at the lab *root*, so a sub-arm store has no `logs/` of its own;
coverage looked only in `store.root` and found nothing. It now searches the parent too, and
cross-checks against trades — a session that left trades behind demonstrably ran. Those
three sessions (2026-09-01…03) are the ones the supervisor killed at 15:55, before
`write_daily`. Their trades were always intact (reconciliation: 0 lost); only the rollup was
missing. `backfill_daily.py` rebuilds them from the durable record, marked
`"backfilled": true`, leaving `still_open` and `feed` as `null` rather than guessing.
Both arms now report **0 unaccounted sessions**.

## 7. What is live now

| | options arm | shares arm |
|---|---|---|
| config hash | `1f7247d7839d9950` (unchanged) | `b53ca8a58aa11718` (forked) |
| universe | QQQ, SPY | 15 names |
| why not broader | no single-name entitlement, no 0DTE on these names, and widening would fork the frozen hash for nothing | — |

Universe: SPY QQQ IWM DIA · XLK XLY XLI XLV XLP XLE XLF · NVDA AAPL GOOGL WMT.
11 of 15 are ETFs deliberately — ETFs have no earnings, and four frozen setups
(`PDH_PDL_Breakout`, `PDH_PDL_FailedBreak`, `Gap_Fade`, `Crabel_Stretch`) behave
pathologically the session after an earnings gap, which the arm has no filter for.

`--symbols` is no longer forwarded to both arms; `autostart` takes `--share-symbols`
separately, and preflight now checks the **union** of both universes — checking only the
options universe would have let the shares arm start on 13 names nothing had verified, and
a missing warmup session there fails silently rather than loudly.

## 8. What this does and does not buy

**Buys:** ~7.5× the trades per session. **Information ×2–3, not ×7.5** — same-day
observations across correlated names are not independent, and with 15 correlated names the
design effect is far larger than it was with 2. Trade count now overstates information by
considerably more than before. n is not evidence; effective n is.

**Measured, not estimated.** `replay.py` over 75 symbol-days puts IMB at **2.03 signals per
symbol-day**, and IMB's `max_per_day = 4` never binds — **0 cap skips** in the live record,
against 3,219 for ORB_15min and 443 for Crabel_Stretch. So essentially every IMB signal
becomes a trade, and 15 symbols yield roughly 22–32 IMB trades per session. The 15-symbol
smoke test produced 18 IMB trades in one session, inside that range.

**Raw n=377 therefore arrives in 12–17 sessions — about three weeks. That does not mean a
verdict in three weeks, and the distinction is the important part of this section.**

n=377 was computed for a 2-symbol arm where trades were close to independent. SPY, QQQ, DIA,
IWM and XLK are largely the same bet on the same day, and every interval here resamples
DATES, so correlated same-day trades collapse into one cluster. **Confidence-interval width
is driven by the number of sessions, not the number of trades.** Broadening multiplies raw
trades by ~7.5 and information by perhaps 2–3.

So a new checkpoint was added to `FREEZE.json`: **`min_sessions: 60`**, and the dashboard
now prints it as *"the checkpoint that governs"* beside the raw counts. It was added
**before any broadened data existed**, and it makes the bar strictly harder — the only
direction a pre-registered bar may move. The honest expectation for a powered IMB verdict is
**3–4 months**: not the 5–7 estimated in the spread survey, and not the three weeks the raw
counter is about to suggest.

**Does not buy:** anything about whether the edge is real. IMB reads +$3.55/trade at n=12
with a CI of [−7.27, +29.37] — consistent with the +$3.34 backtest, with zero, and with
−$5. Broadening is a way to reach n faster. It is not a new hypothesis, and it does not make
any current number more believable.

## 9. Path-independence holds on the new names

`replay.py` re-ran its structural probe across all 15 symbols, 75 symbol-days: for every bar
it evaluates each setup twice — once against the session built incrementally, once against
the full day truncated to that bar. **PASS: every setup returned an identical signal**, so
none of the 13 carries hidden state or order-dependence on the new names. Startup was
verified too: all 15 warm up with 780 prefix 1-minute bars and prior-day levels present, and
a full simulated session runs at 341 ms/tick with an 11,655 / 15 cache hit-to-miss ratio.
