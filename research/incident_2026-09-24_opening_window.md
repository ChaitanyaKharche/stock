# Incident — both arms blind to the open, 2026-08-31 to 2026-09-24

**Every session since 2026-08-31 started after the open, so neither arm has been testing
the opening setups as they are defined.** Found 2026-09-23 while reading the September
logs; fixed 2026-09-24 after the close, effective from the 2026-09-25 session.

No crash, no error, a green log every morning: this project's usual bug shape.

## What happened

`autostart` held both runners until a **post-open** freshness preflight passed. That check
ran at 09:33 and took up to 2m16s, so the runners started at 09:33–09:35. The reason given
in the code (`autostart.py`, `FRESHNESS_AT`) was:

> Costs nothing: no setup can signal before 09:36 because the 09:30-09:35 opening range
> is not formed.

**That was false.** `Crabel_Stretch` trades from 09:31. `PDH_PDL_Breakout` and
`TTM_Squeeze` can fire on the 09:35 bucket. `VWAP_Reclaim` has no window at all.

| dates | runner start | what the late start did |
|---|---|---|
| 2026-08-28 | 09:20 | nothing. Day one had no post-open gate (added in `4956008`). |
| 2026-08-31 → 09-09 | 09:33 → 09:35 | the missed opening bars were evaluated in one batch and **entered late**, at the current price. This is part of the 43-trade stale-signal contamination found on 09-09 (`977f4ae`). |
| 2026-09-10 → 09-24 | 09:35 | the shares arm now refused stale bars, so it was **blind** instead: it processed nothing until ~09:41. The options arm had no such guard and **still entered late**. |

### Why the shares arm was blind until ~09:41, not 09:35

Warmup. The shares arm loads 24 prior sessions for each of 15 symbols, one symbol every
25–40 s. On 2026-09-23 it started at 09:35:18, finished SPY at 09:36:07 and QQQ at
09:36:47, and processed its first bar at ~09:41. By then every bar from 09:30 to 09:38 was
more than 3 minutes old, so the stale-bar guard threw it away, correctly by its own rule:

| date | bars suppressed | first check | blind through |
|---|---|---|---|
| 2026-09-10 | 173 | 09:40:28 | 09:37 |
| 2026-09-14 | 120 | 09:40:55 | 09:37 |
| 2026-09-16 | 135 | 09:41:18 | 09:38 |
| 2026-09-22 | 135 | 09:41:13 | 09:38 |
| 2026-09-23 | 135 | 09:41:13 | 09:38 |

135 = 15 symbols × 9 minutes. The same every day.

### Why being blind did not just skip trades, but changed them

The live `Crabel_Stretch` rule is *"a 1-minute close beyond the level, 09:31–11:00, at
most once a day"*. A runner that cannot see 09:31–09:38 does not skip the day. It enters
on **the first bar it can see** that is still beyond the level. On 2026-09-23 the shares
arm opened Crabel_Stretch on **11 of 15 names in the same minute, 09:41**. That is not a
breakout signal; it is a list of the names that had already moved.

## Measured impact

**Shares arm, 2026-09-10 → 09-24 (1,509 trades):**
- **Crabel_Stretch: 97 of 150 trades entered at 09:38–09:41**, the first minutes the runner
  could see. Net +$161.35. Before 09-10, 12 of 25 Crabel entries came before 09:38.
- Other setups whose window includes the blind minutes, entered in the first visible
  minutes: `ORB_5min` 16 trades (+$218.40), `VWAP_Reclaim` 26 trades (+$64.64).
- These trades are **at risk, not proven wrong**. The lost bars are needed to say which
  entries the rule would have taken earlier or not at all. See follow-up 1.
- `IntradayMomentumBoundary` is **unaffected**: its first checkpoint is 10:00.

**Options arm, whole record (271 ATM trades):**

11 entries were filled more than 3 minutes after the newest bar behind their signal
opened. Net −$245.89.
- **Six are from 2026-08-31 and 09-01**, 40–108 minutes late, after feed recoveries.
- **Five are from 2026-09-10 on**, all at the open. Net +$259.60:

| date | setup | symbol | signal bar | filled | net |
|---|---|---|---|---|---|
| 09-16 | TTM_Squeeze | SPY | 09:35 (5m) | 09:37:02 | −21.08 |
| 09-17 | Crabel_Stretch | SPY | 09:31 | 09:36:41 | −98.08 |
| 09-22 | Crabel_Stretch | SPY | 09:31 | 09:36:52 | −80.08 |
| 09-22 | Crabel_Stretch | QQQ | 09:31 | 09:36:52 | +158.92 |
| 09-23 | Crabel_Stretch | QQQ | 09:31 | 09:36:49 | +299.92 |

**The two arms disagreed on identical bars.** On those days the shares arm refused the
09:31 Crabel signal as stale, and the options arm took it five minutes late.

**The ledger was wrong in the other direction.** It called any start after 09:35:00
PARTIAL, and the runners started 9–43 seconds after that. So every session from 09-10 to
09-24 read PARTIAL. A label that is always on cannot warn about anything, and a 09:34
start, which did lose real signals, would have read clean.

## What changed (2026-09-24)

1. **The runners start before the open.** `START_AT` moved from 09:20 to 09:05. The runners
   now start straight after the structural preflight (~09:07), finish warmup by ~09:15,
   and idle until 09:30. They see 09:31 on time, as `replay.py` does.
2. **The freshness gate still runs, but while the runners are live.** At 09:33,
   `supervise()` runs the preflight once, on the options symbols only. A DELAYED verdict
   stops every arm and records the day ABORTED; an options-only failure stops the options
   arm. `_gate_verdict` keeps the old precedence exactly. The two-symbol preflight keeps
   its load off the terminal during the minutes the runners need it.
3. **Starting early is safe only because staleness is now refused where it is used.**
   - The underlying quote was already checked on both arms (`STALE_QUOTE_SEC`). The
     options arm now measures its age at receipt (`recv_ts`), as the shares arm has since
     `977f4ae`.
   - **Option quotes were never checked by the runner.** The post-open gate was the only
     defence, which is why the runners were held behind it. `runner.py` now skips any fill
     whose ATM quote is older than `preflight.DELAYED_THRESHOLD_SEC` (120 s, imported so
     the two cannot drift). The reason is recorded as `stale_option_quote`.
   - So a delayed feed in the minutes before the 09:33 verdict produces SKIPs, never fills.
4. **The options arm got the stale-bar guard** (`STALE_BAR_SEC = 180`). It is measured
   exactly as the shares arm measures it: from the open stamp of the newest 1-minute bar,
   so a 5-minute bucket is not given 60 s of extra leeway. The skip is written per signal
   as `stale_bar`.
5. **`PARTIAL_AFTER` is now 09:30.** With a pre-open start, the honest bar is the open.

## What this does not change

- **Config hashes are unchanged:** options `1f7247d7839d9950`, shares `b53ca8a58aa11718`.
  Every guard sits outside the hash, by the precedent of `STALE_QUOTE_SEC`. No setup
  definition, fill rule, cap or fee changed.
- **No trade is removed or relabelled.** The clock rule stands. The affected trades above
  stay in the record and are quoted with the split below.
- **Past ledger lines are not rewritten.** 09-10 → 09-24 stay PARTIAL, which was correct
  in spirit: those sessions did lose the open.

## How to quote the affected sessions

Use the same split discipline as 2026-09-11 and 2026-09-15:
- **Shares, 09-10 → 09-24:** report Crabel_Stretch, ORB_5min and VWAP_Reclaim with their
  first-visible-minute entries shown separately. Do not report them as a clean opening
  test.
- **Options:** report the 11 late entries as degraded: net −$245.89, of which +$259.60 is
  since 09-10.

## Verification

- `trade_analysis/live_lab/options_runner_guards_test.py` has 10 tests and
  `trade_analysis/live_lab/opening_window_test.py` has 18.
- **All the defect tests fail against `e4dfaf6`**, the code before this fix. The
  `main()`-driven tests use a clock that moves realistically (a preflight costs 2m15s),
  because a stub that took no time let the old sequencing pass.
- Full suite: 213 passed. All 10 house-style checks exit 0. Both hashes were re-printed under the lab's own
  Python 3.12.10 and are unchanged.

## Follow-ups

1. **Measure the shares counterfactual exactly.** Replay the shares arm for 2026-09-10 →
   09-24 with the opening bars present (`replay.py`). That says which of the 97 Crabel
   entries the rule would have taken at a different minute, or not at all.
2. **Watch the 2026-09-25 log.**
   - Runners should start ~09:07 and warmup should finish before 09:30.
   - The first `[shares]` bar should be processed at ~09:31.
   - `shares/outages.jsonl` should show roughly zero `stale_bar` lines before 09:40.
   - The 09:33 line should read `freshness re-check clean`.
3. **Preflight's option-age check samples `now` early.** It reported a median option-quote
   age of **−53 s** at 09:33 on 09-23. This is the same one-sided-reference defect `977f4ae`
   fixed for the underlying and never ported to options. It is harmless today because it
   errs toward "fresh", but it would hide a delay of up to ~50 s.
4. **CLAUDE.md is stale on options data.** It says ThetaData options lapsed on 2026-09-08
   and nothing new can be measured. The options arm has had a working chain since
   2026-09-14 (logs show 366 usable QQQ 0DTE contracts and real bid/ask). The trader should
   confirm what changed before the file is edited.
