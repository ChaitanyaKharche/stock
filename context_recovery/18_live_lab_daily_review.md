# 18 — Every live lab session, 2026-08-28 to 2026-10-15

Written 2026-10-15 at about 18:00 MST, after that day's session. Read-only. Sources, all in
the main checkout's `live_lab_data/`: the 37 daily log files, `session_ledger.jsonl`,
`trades.jsonl` and `trade_corrections.jsonl` for both arms, the `daily/` summaries,
`outages.jsonl`, and Windows' own power-event log for 2026-10-15.

## The short version

- **35 log days, 34 market sessions** (2026-09-07 was Labor Day and the lab correctly stood down).
- **Only 6 sessions ran start to finish without a real problem:** 09-30, 10-01, 10-09, 10-12,
  10-13 and 10-14. Every other session lost something: the open, an arm, part of the
  afternoon, or the whole day.
- **Running total (paper money):** options −$958.88 on 580 trades; shares −$13,436.24 on 4,204
  trades; together **−$14,395.12**.
- **Since 10-08** (the last figure from the old chats): options +$53.20 and shares +$1,790.29.
  10-14 was the best combined day of the whole test: +$2,910.11.
- **Today, 10-15, was lost.** The laptop's battery ran down and Windows put it to sleep at
  00:29 MST. It did not wake until 10:46 MST, so the 05:30 start never happened. The lab started
  at 13:55 ET with two hours left and took **zero trades**.

### How the money is counted

- Options means the at-the-money strike only, the same headline the lab's daily summaries use.
- Shares means every trade.
- Live gap-recovery corrections that were never retracted are applied, latest one wins.
- **Check that passed:** counted this way through 10-08, the totals are options −$1,012.08 and
  shares −$15,226.53. Those match the old session's running totals (−$1,012.07 and −$15,226.54)
  to the cent.
- Without the corrections, the totals are −$1,018.88 and −$15,460.22.
- On every day that has a daily summary file, the trade rows add up to exactly that summary.
  The one exception is shares on 09-04, where three exits were reconstructed and added later.

## What kept going wrong, by cause

| cause | sessions | what it cost |
|---|---|---|
| **Missed the open** (the opening-window bug: the lab could not see 09:30–09:41) | every session from 08-28 to 09-24, 21 in all | The first ten minutes of every session. Fixed 09-24; from 09-25 the arms start at about 09:07 ET. |
| **Options data switched off** (ThetaData returned "403 entitlement" for options) | 09-08, 09-09, 09-10, 09-11 | Shares arm only for four sessions. Options were back on 09-14. |
| **Laptop went to sleep mid-session** | 09-11, 09-17, 09-18, 09-24, 09-25, 09-28, 09-29, 10-08 | 4 to 80 minutes each, often across the close. 09-25 lost about 39 minutes, including the close: the worst shares day, −$2,843.87. |
| **Network or data terminal died and stayed dead** | 10-02 (about 2 h), 10-05 (14:10 to close), 10-06 (12:26 to close) | The supervisor restarted the terminal six times, then logged "leaving it for a human". 10-05 and 10-06 have no daily summary; their exits were filled from history the next morning. |
| **Whole lab died with no trace** | 10-07 (about 15:30 ET) | No summary, no commit. Exits filled from history on 10-08. Ledger says `INTERRUPTED`. |
| **Shares arm crashed** | 09-15 (10:51 ET) | A `PermissionError` saving open positions. Per the old chats, a monitor the assistant had running was holding the file. Three restarts; 58 open positions recovered. |
| **Laptop asleep at start time** | 10-15 | The whole day. |
| **Did not push to GitHub** | 09-21 to 09-24 (main checkout behind GitHub), 10-05 and 10-06 (no internet), 10-07 (no commit at all) | Since resolved: `main` matches GitHub as of 10-13. |

Two lines are printed at every start and never change: 19 shares positions from 09-08 and 32
from 09-10 have no exit and are "left for a human".

## Every session

The net columns use the counting method above; "—" means the arm did not trade. Times are ET.

| date | ledger says | arms started | options trades | options net | shares trades | shares net | what happened |
|---|---|---|---|---|---|---|---|
| 08-28 | COLLECTED | (before the field existed) | 27 | −$1,201.18 | — | — | First day. Options arm only. Ran until 16:42. |
| 08-31 | COLLECTED | — | 15 | −$1,012.21 | — | — | Options only. |
| 09-01 | COLLECTED | — | 16 | −$614.29 | 16 | −$278.81 | Shares arm's first day. The supervisor killed both arms at 15:55, the minute they close positions; options end-of-day exits for this day were never recovered. |
| 09-02 | COLLECTED | — | 19 | −$126.53 | 22 | +$5.77 | Same 15:55 kill; 4 + 4 exits rebuilt later. A second runner collided on the lock at 15:00. |
| 09-03 | COLLECTED | — | 22 | +$1,246.23 | 22 | +$214.65 | Same 15:55 kill; 6 + 6 exits rebuilt. Rebuilding turned this options day from a reported loss into a gain. |
| 09-04 | COLLECTED | — | 20 | −$466.62 | 20 | −$103.84 | Shares arm mistook a dead network for an early close and flattened early. |
| 09-07 | NOT_A_SESSION | — | — | — | — | — | Labor Day. Preflight saw quotes 3,693 minutes old and refused; correct. |
| 09-08 | PARTIAL | — | — | — | 65 | −$373.14 | Terminal would not answer; three aborts. Options 403. Shares alone from 11:34. |
| 09-09 | PARTIAL | 09:53 | — | — | 130 | −$1,339.77 | Options 403; shares only, started late. |
| 09-10 | PARTIAL | 09:35 | — | — | 106 | −$1,734.28 | Options 403. An arm was still alive at 17:35 and was terminated. |
| 09-11 | PARTIAL | 09:36 | — | — | 135 | −$1,323.80 | Options 403. Laptop asleep 15:53–16:23, across the close. |
| 09-14 | PARTIAL | 09:35 | 15 | +$856.79 | 143 | +$1.71 | Options back. |
| 09-15 | PARTIAL | 09:35 | 20 | +$553.38 | 139 | +$395.71 | Shares arm crashed at 10:51; terminal also down; three restarts. |
| 09-16 | PARTIAL | 09:35 | 19 | +$88.46 | 155 | −$1,316.51 | Late open only. |
| 09-17 | PARTIAL | 09:35 | 13 | −$729.05 | 134 | −$980.05 | Laptop asleep 15:48–17:08, across the close. |
| 09-18 | PARTIAL | 09:35 | 14 | −$506.13 | 118 | −$1,035.57 | Three sleeps between 14:21 and 15:55, about 65 minutes. |
| 09-21 | PARTIAL | 09:35 | 13 | **+$3,280.95** | 131 | +$552.01 | Best options day of the test. Push to GitHub failed. |
| 09-22 | PARTIAL | 09:35 | 16 | +$184.71 | 151 | +$49.45 | Push failed. |
| 09-23 | PARTIAL | 09:35 | 20 | +$902.38 | 143 | +$993.83 | Push failed. |
| 09-24 | PARTIAL | 09:35 | 22 | +$872.22 | 154 | +$478.88 | Two sleeps, 14:55–15:27. Push failed. The opening-window bug was found and fixed this day. |
| 09-25 | COLLECTED | **09:06** | 20 | −$963.62 | 165 | **−$2,843.87** | First on-time start. Three sleeps from 15:39 to 16:19, across the close. Worst shares day. |
| 09-28 | PARTIAL | 09:06 | 25 | +$884.98 | 193 | −$2,301.27 | 11-minute sleep at 11:04, replayed from history. |
| 09-29 | PARTIAL | 09:06 | 19 | −$508.54 | 168 | −$342.80 | 4-minute sleep; a 16-second terminal drop. |
| 09-30 | COLLECTED | 09:06 | 26 | −$659.10 | 203 | +$117.63 | Clean. |
| 10-01 | COLLECTED | 09:06 | 29 | −$581.34 | 206 | −$1,119.56 | Clean. |
| 10-02 | PARTIAL | 09:06 | 23 | **−$1,268.86** | 144 | −$2,476.60 | Terminal died at 11:51; six restarts; given up at 12:16. Delayed-feed aborts. Arms back about 15:08, roughly 2 hours lost. Worst options day. |
| 10-05 | COLLECTED | 09:06 | 24 | +$1,008.06 | 185 | −$1,359.72 | Network died at 14:10 for the rest of the day. No daily summary. Exits filled next morning. The ledger's "COLLECTED" is wrong. |
| 10-06 | COLLECTED | 09:06 | 15 | −$718.21 | 136 | +$624.48 | Network died at 12:26 for the rest of the day. Same as 10-05. |
| 10-07 | INTERRUPTED | 09:06 | 20 | −$1,129.62 | 158 | −$680.79 | The whole lab died about 15:30 with no message. Exits filled on 10-08. |
| 10-08 | PARTIAL | 09:06 | 24 | −$404.94 | 181 | +$949.73 | 18-minute sleep at 15:13; the terminal took 20 minutes to come back. |
| 10-09 | COLLECTED | 09:06 | 20 | −$968.62 | 161 | +$220.55 | A 30-second terminal drop at 11:17; otherwise clean. |
| 10-12 | COLLECTED | 09:07 | 21 | −$144.70 | 172 | +$363.19 | Clean. |
| 10-13 | COLLECTED | 09:06 | 19 | −$548.54 | 167 | +$11.50 | Clean (watched live; see report 17). |
| 10-14 | COLLECTED | 09:06 | 24 | **+$1,715.06** | 181 | **+$1,195.05** | Clean. Best combined day. |
| 10-15 | PARTIAL | **13:55** | 0 | $0.00 | 0 | $0.00 | Laptop asleep 00:29–10:46 MST (sleep reason: battery). Started with two hours left; no trades. |

## 2026-10-15 in detail

| MST | ET | what happened | source |
|---|---|---|---|
| 00:29 | 03:29 | Windows enters sleep. Reason: **battery**. | System event 42 |
| 05:30 | 08:30 | Scheduled start time. The machine is asleep; nothing runs. | — |
| 10:46 | 13:46 | Machine wakes ("resume from hibernate"). | System events 507, 1 |
| 10:52 | 13:52 | Task Scheduler catches up on the missed start; autostart runs. | log |
| 10:53 | 13:53 | Terminal started, answering in 11 s. Preflight passes. | log |
| 10:55 | 13:55 | Both arms start. Ledger: `OPENED`, `late_start true`. | log, ledger |
| 13:00 | 16:00 | Both arms stop. **Options:** 7 signals decided, all 7 fills skipped for `no_underlying_quote`; 385 signals skipped as `stale_bar`. **Shares:** no signal rows at all. Ledger: `PARTIAL`. Committed and pushed. | log, `signals.jsonl` |
| 13:22 onwards | 16:22 onwards | The task keeps re-firing every 30 minutes into the evening, each run exiting at once ("past the 15:30 cutoff"). Harmless. | log |
| 14:29–15:19 | 17:29–18:19 | Laptop sleeps again, battery budget exceeded, then wakes. | System events |

### Why the 13:55 start produced no trades (checked 2026-10-16)

**The laptop's clock was about 7 seconds fast, and the lab throws away any stock quote that
looks more than 5 seconds old.** So it threw away every quote all afternoon. The late start
did not cause this; the clock did.

1. The machine woke from a ten-hour hibernation at 10:46 MST. Windows set the time from the
   laptop's own hardware clock (System event 1: "synchronized with the hardware clock").
2. Windows network time sync was not running. `W32Time` is set to Manual and reports "Last
   Successful Sync Time: unspecified". The clock was not corrected until 15:19:57 MST, after
   the close. At that moment the time service pulled it back by **6.946 seconds**
   (System event 1, "Time Delta: −6946 ms", the same second as event 37 "receiving valid time
   data from time.windows.com").
3. Both arms measure a quote's age as the laptop's clock minus the exchange timestamp, and
   discard it above 5 seconds (`STALE_QUOTE_SEC = 5.0`, `runner.py:52`,
   `shares_runner.py:73`). Every quote measured 6.1–7.7 seconds old. The arms recorded 946
   (options) and 1,770 (shares) `stale_quote` rows, every one of them that afternoon.
4. With its quote discarded, the shares arm skips the symbol entirely
   (`shares_runner.py:442-443`), so it wrote no signal rows at all. The options arm still
   evaluated its setups and decided 7 signals. Each fill was then skipped for
   `no_underlying_quote` (`runner.py:441`).
5. **The preflight check passed anyway.** It logged `[OK] QQQ underlying REAL-TIME (age 6.2s)`
   because its limit is 120 seconds (`DELAYED_THRESHOLD_SEC = 120.0`, `preflight.py:39`). The
   gate allows 120 seconds and the runners allow 5. Between those two numbers the lab can say
   "All checks passed. Safe to run the lab." and then trade nothing. This is the project's usual
   bug shape: no crash, a confident message, nothing measured.
6. The 385 `stale_bar` skips are a different and harmless thing. On start, the runner replayed
   the morning's bars (09:31–11:19) and correctly refused to act on bars hours old.

The same mechanism, smaller: a stale-quote burst on 2026-09-24 (163 + 135 rows) and on 10-13
(67 + 47). On 2026-10-16 at 00:56 MST the clock was +0.36 s from internet time. Fine for now,
but Windows still reports the time service as "not synchronized", so the next long sleep can
repeat 10-15.

## Things the logs show that are worth knowing

- **The `COLLECTED` label overstates.** 10-05 and 10-06 lost hours and are still `COLLECTED`.
  Use this table or the logs, not the ledger alone.
- **A sleeping laptop is the single biggest source of damage**: eight sessions with mid-day
  sleeps, plus a whole lost day today. The lab's sleep guard does not stop Windows sleeping
  on battery.
- **The watchdog gives up after six terminal restarts.** On 10-05 and 10-06 that happened
  within about 20 minutes of the network dropping, and it then logged "leaving it for a human"
  41 and 93 times.
- **The worst shares days (09-25, 09-28, 10-02) all had an outage or a sleep.** Recovered
  exits use history, so they are an estimate of what live trading would have done, not a fill.
