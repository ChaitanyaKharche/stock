# Replay — every live-lab session so far, re-run as if the connection never broke (2026-09-29)

**Bottom line:**
- **Not one of the 22 sessions from 2026-08-28 to 2026-09-29 ran clean.** Adding it up, the
  lab was blind for about 10 hours of market time. On top of that, options were blocked for
  4 whole days.
- A clean replay of the same days does **not** rescue the result.
  - **Options:** almost all of the gap is the 4 blocked days.
  - **Shares:** the clean replay still loses about $10.4k. The outages cost about $1.1k of
    the shares loss, after the exit corrections; the rest is the strategy.

**This is a BACKFILL, not evidence.**
- It was computed after the fact.
- It lives in `live_lab_backfill/`, outside `live_lab_data/`, and is never counted toward a
  checkpoint.
- Every record carries `"provenance": "BACKFILL"`.

## 1. What was run

- `backfill.py --arm both` ran every session through the **real runner code**:
  - options: `LiveLab`, QQQ + SPY, 0DTE ATM / ATM±1
  - shares: `SharesLab`, 15 names
- It used a history-backed feed and a simulated clock that steps once a minute at HH:MM:02.
  Warmup is at 09:07, the six lines load at 09:31, and the session flattens at the close.
- **Each day now has a complete session log:**
  - `live_lab_backfill/<arm>/logs/<day>.log`: the runner's own console output
  - `daily/<day>.json`
  - `trades.jsonl`
  - `signals_archive/<day>.jsonl.gz`: every decision and skip, one file per session
- Comparison files:
  - `report.json`: live vs replay by day
  - `audit.json` (`replay_audit.py`): why each trade differs
- **The replay runs today's config on every day.** The "clean" columns below filter it back
  to what live actually ran that day:
  - options: the six lines only from 09-25
  - shares: QQQ/SPY until 09-04, and the 15 names from 09-05
- **Reproducible:** re-running reproduced the 2026-09-25 options replay to the cent, on all
  19 days it covered.

## 2. What went wrong live, and what a clean run shows

"Clean" means the replay filtered to that day's live scope. "Live" includes the exit
corrections in `trade_corrections.jsonl`. All figures are dollars.

| day | what went wrong live | options live | options clean | shares live | shares clean |
|---|---|---:|---:|---:|---:|
| 08-28 | — | -1,201 | -1,364 | — (not open) | — |
| 08-31 | blind until 09:39, feed down 3m | -1,012 | -1,340 | — (not open) | — |
| 09-01 | blind until 09:39, feed down 41m | -614 | -664 | -279 | -197 |
| 09-02 | blind until 09:39 | -127 | +91 | +6 | +47 |
| 09-03 | blind until 09:39 | +1,246 | +1,540 | +215 | +240 |
| 09-04 | blind until 09:39, feed down 14m | -467 | -737 | -104 | -137 |
| 09-08 | **options blocked**, shares blind until 11:40 | +0 | -374 | -373 | -713 |
| 09-09 | **options blocked**, blind until 09:59 | +0 | -727 | -1,340 | -1,427 |
| 09-10 | **options blocked**, blind until 09:41, feed down 9m | +0 | -1,177 | -1,734 | -1,195 |
| 09-11 | **options blocked**, blind until 09:42, asleep 6m | +0 | -1,474 | -1,647 | -1,597 |
| 09-14 | blind until 09:41, feed down 1m | +857 | +1,214 | +2 | +152 |
| 09-15 | blind until 09:41, feed down 6m | +553 | +321 | +396 | +434 |
| 09-16 | blind until 09:41, feed down 2m | +88 | -521 | -1,317 | -451 |
| 09-17 | blind until 09:41, asleep 11m, feed down 29m | -729 | -1,047 | -980 | -414 |
| 09-18 | blind until 09:41, asleep 65m, feed down 5m | -503 | -450 | -1,013 | -1,041 |
| 09-21 | blind until 09:41, feed down 3m | +3,281 | +3,317 | +552 | +178 |
| 09-22 | blind until 09:41 | +185 | +641 | +49 | -114 |
| 09-23 | blind until 09:41 | +902 | +1,235 | +994 | +938 |
| 09-24 | blind until 09:41, asleep 31m, feed down 2m | +1,038 | +1,107 | +527 | +703 |
| 09-25 | asleep 20m, feed down 4m | -964 | -957 | -2,844 | -2,788 |
| 09-28 | asleep 10m, frozen 11m, feed down 1m | +983 | +184 | -2,250 | -2,033 |
| 09-29 | asleep 4m, frozen 6m, feed down 4m | -650 | -592 | -311 | -948 |
| **22 sessions** | | **+2,868** | **-1,772** | **-11,451** | **-10,363** |

How to read the columns:
- **"blind until"** is the late start (`incident_2026-09-24_opening_window.md`): the runner
  started at about 09:35 and saw its first bar about 6 minutes later.
- **"asleep"** is a `host_suspend` window, counted in RTH minutes only.
- **"frozen"** is the catch-up freeze (`network_resilience.md`).
- **"feed down"** is a run of failed ticks.

Total blind RTH minutes, summed across days:

| cause | options | shares |
|---|---|---|
| late start | 332 min over 18 days | 332 min over 18 days |
| asleep | 149 min over 7 days | 149 min over 7 days |
| feed down | 112 min | 103 min |
| frozen | 17 min | 15 min |

**Options:**
- **The 4 blocked days are the whole story.** The clean replay loses **−$3,752** on 09-08 →
  09-11. Live never traded those days.
- On the 18 days live did trade, clean made **+$1,980** and live **+$2,868**. Live was
  ahead by $888, which is noise at 18 days.

**Shares:**
- **Live lost $1,088 more than clean** (−$11,451 vs −$10,363).
- The worst days were 09-16 (−$866 vs clean), 09-17 (−$566) and 09-10 (−$539).
- Live beat clean on 09-29 (+$637) and 09-21 (+$374).
- **The loss is the strategy, not the outages.** In the clean replay, QQQ + SPY make
  −$526 and the other 13 names −$13,053 (current config, all 22 days). This is the same
  split as the live record.

## 3. What each trade that did not pair was caused by (`replay_audit.py`)

Each live trade is paired with the replay trade on the same signal (setup, symbol,
direction, entry bar). The bins:

| bin | options (ATM) | shares |
|---|---|---|
| same signal, same exit | 169 | 850 |
| same signal, different exit | 85 | 418 |
| same trade, entry bar ≤ 3 min apart | 22 | 191 |
| live only | 59 | 851 |
| clean replay only | 151 | 1,043 |

Every unpaired trade gets the first cause that live itself recorded:

| cause | options n / $ | shares n / $ |
|---|---|---|
| replay only: **options blocked** | 73 / −3,752 | — |
| replay only: **late start** (blind open) | 3 / +396 | **413 / −1,260** |
| replay only: feed down | 5 / −376 | 13 / −204 |
| replay only: host asleep | — | 13 / −102 |
| replay only: stale bar refused, quote too old, degraded bar | 1 / −153 | 24 / −47 |
| replay only: live skipped it (a cap already used) | 19 / +25 | 37 / −105 |
| live only: **late entry** (old bar filled minutes later) | 3 / −278 | 39 / −506 |
| live only: the replay skipped it (a cap already used) | 20 / −403 | 103 / −308 |
| replay only: no live signal | 50 / −1,319 | 543 / −2,528 |
| live only: no replay signal | 36 / +1,308 | 709 / −3,872 |

**The last two rows are mostly not outage damage.** Some are knock-on effects of an earlier
difference (a daily cap spent at another time). The rest is the vendor's bar corrections,
covered in §4.

## 4. Can a replay reproduce a live day? In totals yes, trade for trade no

**The clean-hours check.** Take the three sessions that started before the open (09-25,
09-28 and 09-29) and use only the trades that entered before that day's first problem.

| | trades with a replay twin | exact twins |
|---|---|---|
| options | 30 / 39 (77%) | 12 (31%) |
| shares | 307 / 405 (76%) | 177 (44%) |

- **The dollar gap on those clean hours is small:** $1.32 per trade on shares, 265 round
  trips.

**Why not 100%: the vendor corrects its bars after the close.**
- On the **same** decisions, live and replay computed a VWAP that differed by a median
  **1.6 bp (about 12 cents on QQQ)**, up to 3.6 bp. VWAP here is the volume-weighted
  average price.
- That is enough to flip VWAP-cross signals.
- The replay can only ever see the corrected bars.

**It depends on the setup** (clean hours, shares, share of trades on only one side):

| setup | one-sided |
|---|---|
| Crabel_Stretch | 0% |
| Gap_Fade | 8% |
| IntradayMomentumBoundary | 19% |
| Six_Lines / Six_Lines_NoCap | 25% |
| MOMO_CHASE | 54% |
| ORB_5min | 72% |
| VWAP_Reclaim | 79% |

- **Fill timing.** The replay fills on the NBBO at the minute mark (NBBO = best bid and
  offer); live filled a median 1.5 s later.
  - That is usually nothing.
  - At 15:55 it can matter. On 09-22, QQQ's bid fell from $747.70 to $747.37 between 15:55:00
    and 15:55:01; the underlying 1m history quote was checked against the 1s quotes, and
    there is no lookahead.

**So the replay answers "what would a runner with a perfect connection have logged?"** Its
day and month totals can be compared to live; a single trade's P&L cannot.

## 5. Corrections added to the live record today

`gap_recovery.py` was run for the three old sleep days that pre-date the automatic
7-day recovery. Corrections are append-only and the original rows are untouched.

| day | arm | written | effect |
|---|---|---|---|
| 09-17 | shares | 46 | −$1,311 → −$980 |
| 09-17 | options | 18 | −$789 → −$729 (ATM) |
| 09-18 | shares | 44 | −$931 → −$1,013 |
| 09-18 | options | 9 | −$506 → −$503 (ATM) |
| 09-11 | shares | **refused** | the replay reproduced only 80% of checkable exit rules, gate is 90% |

## 6. Fixed 2026-09-30: the signals file outgrowing GitHub

- **The problem.** The live `shares/signals.jsonl` had reached 37.7 MB and grew 3–4.5 MB
  a session, almost all of it `max_per_day` SKIP rows. It would have passed GitHub's
  100 MB per-file limit, and broken the nightly push, in about four weeks.
- **Nothing was dropped.** The shares pre-registration (item 6) requires every capped
  signal to be written as a SKIP, so the fix changes only where the rows are stored, not
  what is recorded.
- **How it works.** Each runner, at start-up and under its lock, moves every row from
  earlier sessions into `signals_archive/<date>.jsonl.gz` (`store.roll_signals`). The
  active file then holds only today. Every reader goes through `store.read_signals`:
  the store, the daily summary, the checkpoint funnel, gap recovery, the dashboard and
  the replay audit.
- **Done once on the real files tonight, with a before/after fingerprint of every row.**

  | | rows (identical) | before | after |
  |---|---|---|---|
  | live options | 5,256 | 1.5 MB | 0.3 MB in 19 files |
  | live shares | 153,447 | 37.7 MB | 5.2 MB in 21 files |
  | replay shares | 317,763 | 83.7 MB | 9.3 MB in 22 files |

  `report.json` and `audit.json` came out byte-identical when rebuilt from the archive.
- **Tests:** `signals_archive_test.py`, each verified to fail on the broken version.
  - nothing lost, reordered or altered
  - no duplicates after a double roll or a crash mid-roll
  - the sequence counter still resumes above archived rows
  - a torn last line stays with its session
  - both runners roll before writing anything
