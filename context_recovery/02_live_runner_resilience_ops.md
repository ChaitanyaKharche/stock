# 02 — Live runner: launch, supervision, outage recovery, backfill, replay, daily operations

Written 2026-10-13 about 09:40 ET, while the lab was running a live session from the main
checkout. Read-only work. Code read at commit `6f9c1ab` (worktree and main checkout were on the
same commit; `git worktree list` shows the main checkout on branch `main`).

Conventions used below:

- **ET** = exchange time (New York). **MST** = the laptop's local time (Arizona, no daylight
  saving). Until 2026-10-31, ET is 3 hours ahead of MST. From 2026-11-01 it is 2 hours ahead.
- `path:line` refers to files under `trade_analysis/live_lab/` unless a longer path is given.
- "log 10-12:N" means line N of `C:\Users\chaitanyakharche\Documents\stock\live_lab_data\logs\2026-10-12.log`
  (logs are gitignored, they exist only in the main checkout).
- **(inferred)** = I reasoned it from code or data; I did not see it happen or see it stated.
- "Arm" = one of the two paper-trading runners: **options** (0DTE calls/puts on QQQ and SPY)
  and **shares** (15 stocks/ETFs, $10,000 per trade).

Where the seven slice questions are answered:

| question | section |
|---|---|
| 1. Daily timeline, ET and MST, what repeats | §3.1, §3.2 |
| 2. Every guard in `autostart.py` | §7.1 |
| 3. The outage story and the after-the-fact write rule | §3.4, §6.2 |
| 4. Backfill and replay versus the live record | §3.6 |
| 5. The end-of-session commit | §3.5 |
| 6. Known recovery gaps, checked against current code | §8.3 |
| 7. What a new assistant could damage by accident | §7.3 |

---

## 1. In one paragraph

This slice is everything that lets a paper-trading forward test run by itself on a laptop that
rides in a car. A Windows scheduled task starts `autostart.py` at 05:30 local time; it sleeps
until 09:05 New York time, starts the data terminal, checks the feed, launches the two runners
before the open, watches them all day, restarts what dies, notices when the laptop slept, and
at the close fills in exits the machine could not record live, then commits and pushes the
day's record to git. Around it sit the repair tools (`catchup.py` during the session,
`gap_recovery.py` and `stranded.py` after it), the "what would a perfect connection have
logged" replay (`backfill.py`, `replay_audit.py`), and a few diagnostics. It exists because
the forward test's only value is an unbroken, honest record, and almost every hole in that
record so far came from operations (sleep, network, a killed process), not from strategy code.

## 2. The goal it serves, and how the goal moved

The constant goal: **every decision the lab makes is on disk before any price that could
influence it exists, and nothing is ever quietly edited afterwards** (`store.py:3-5`,
`backfill.py:9-14`).

How the operational goal changed, in order (dates from commit messages and incident notes):

1. **2026-08-28, day one.** "Start the runner once a day." `autostart` ran the runner exactly
   once (`git show 4956008`, item 1).
2. **2026-08-31.** "Keep it alive." Supervision with restarts, single-instance locks, a
   date-stamped recovery file, runner output captured in the log (`4956008`).
3. **2026-09-03.** "Do not destroy the end of the day." The supervisor had been killing both
   runners at 15:55:02, the minute they flatten, for three sessions (`264068e`).
4. **2026-09-08.** "Make gaps readable." Four stacked failures on one morning produced the
   session ledger (`ledger.py:1-22`).
5. **2026-09-10/11.** "Survive the host sleeping." Suspend detection and a grace window
   (`dbf0bba`).
6. **2026-09-16/17.** "Get the record off this disk, automatically." Six sessions existed only
   on the laptop (`research/live_lab_coverage_audit.md`; `47c6835`).
7. **2026-09-17.** "A hotspot switch is not an outage; wait." (`e7bf6de`).
8. **2026-09-24.** "See the open." Both arms had been blind to 09:30–09:41 on every session
   since 08-31 (`research/incident_2026-09-24_opening_window.md`; `88bae1e`).
9. **2026-09-25.** "Recover exits a sleep stole, from history, without editing rows"
   (`1e7aeef`).
10. **2026-09-28.** "Survive a commute mid-session": freeze, catch up, decide once
    (`research/network_resilience.md`; `494c314`).
11. **2026-10-06 to 10-08.** "Positions with no exit row at all" (`53bffa3`), then three
    review passes that narrowed recovery to **only exits that were actually stolen**
    (`7d0feb1`, `31ddee2`, `4114f1b`).

The direction of travel: from "keep collecting" to "never let the record say something the
machine did not observe, and label everything that was reconstructed".

---

## 3. How it works, end to end

### 3.1 The daily timeline

The scheduled task (queried read-only on 2026-10-13):

| property | value |
|---|---|
| name | `LiveLab-QQQ-SPY` |
| action | `C:\Users\chaitanyakharche\Documents\stock\trade_analysis\live_lab\run_lab.cmd` |
| trigger | weekly, Monday–Friday (`DaysOfWeek 62`), start `05:30` local (`-07:00`) |
| repetition | every 30 minutes for 8 hours, `StopAtDurationEnd = True` |
| settings | `WakeToRun True`, `StartWhenAvailable True`, `MultipleInstances IgnoreNew`, `ExecutionTimeLimit 12 h`, runs on battery, `RestartCount 2` / 5 min |
| principal | `LogonType Interactive`, not elevated |

**The task definition is not in the repository.** Nothing in the tree registers it (the only
mention is the comment at `run_lab.cmd:3`). The table above is the only written copy I know of.

`run_lab.cmd:6-7` does `cd /d C:\Users\chaitanyakharche\Documents\stock` and runs the pyenv
Python 3.12.10 by absolute path with
`-m trade_analysis.live_lab.autostart --symbols QQQ SPY --start-terminal`.

A normal day, taken from the 2026-10-12 log, with the code that drives each step. MST column
is valid until 2026-10-31.

| ET | MST | what happens | code | seen in |
|---|---|---|---|---|
| 08:30:02 | 05:30:02 | Task fires. Log file opened, stdout and stderr both tee'd to it. | `autostart.py:1008-1016` | log 10-12:1-2 |
| 08:30:14 | | Weekend check; `ledger.reconcile` fills in days nobody recorded; supervisor takes its lock; wake lock requested. | `autostart.py:1018-1038` | log 10-12:3-5 |
| 08:30:14 | | **Morning stranded pass**: give exits to positions an earlier session never closed. Must finish 3 minutes before 09:05 and needs at least 240 s. | `autostart.py:1040-1045`, `:700-701` | log 10-12:6-7 |
| → 09:05 | → 06:05 | Sleeps until 09:05 **ET**, computed live, so the offset to local time cannot drift it. | `autostart.py:350-367`, `clock.py:27-29` | log 10-12:8-10 |
| 09:05:02 | 06:05 | Terminal not answering → start Theta Terminal (the vendor's local data gateway, port 25503) → wait until it serves *history*, not just a quote. | `autostart.py:1054-1081`, `:327-347`, `:277-324` | log 10-12:11-14 (83 s that day; 9 s on 10-13) |
| 09:06:26 | | Holiday check (see §7.2: cannot fire this early), then the **structural preflight** on all 15 symbols. About 90 s. | `autostart.py:1083-1133`, `preflight.py:231-312` | log 10-12:15-108 |
| 09:07:54 | 06:07 | Ledger line `OPENED`. Both arms started as child processes. They warm up (24 prior sessions per symbol) and idle until the open. | `autostart.py:1167-1185`, `:833-859`; `runner.py:256-283` | log 10-12:109-130 |
| 09:25 | 06:25 | Terminal watchdog becomes active, checks once a minute until 16:00. | `autostart.py:554-559` | — |
| 09:30 | 06:30 | Open. Runners tick every 5 s with positions open, 15 s without. | `runner.py:131`, `:308` | first entries log 10-12:148-153 |
| 09:31 | | Six-line levels loaded (an `events.jsonl` row `levels`). | other slice | `live_lab_data/events.jsonl` tail |
| 09:33:02 | 06:33 | **Freshness re-check** on QQQ and SPY only, while the runners are live. A delayed feed stops every arm and records the day `ABORTED`. About 70 s. | `autostart.py:1143-1164`, `:899-909`, `:773-791` | log 10-12:154-195 |
| all day | | Supervisor loop every 5 s: watchdog, child liveness, restart with backoff, suspend detector. | `autostart.py:896-975` | — |
| 15:30 | 12:30 | A *fresh* `autostart` invocation no longer starts a session after this. | `autostart.py:87`, `:354-356` | — |
| 15:55 | 12:55 | Runners flatten everything (rows print as `15:56 CLOSE ... eod`). Supervisor stops *restarting* but does not kill. | `clock.py:24`; `autostart.py:89-94`, `:940-944` | log 10-12:590-612 |
| 16:00 | 13:00 | Runners leave their loop, write the daily summary, exit. | `runner.py:296-315`, `shares_runner.py:322-346` | log 10-12:613-616 |
| 16:00:14 | | Supervisor: ledger `COLLECTED` or `PARTIAL`; gap recovery (last 7 days); stranded pass; **archive commit and push**; wake lock released. | `autostart.py:1185-1240` | log 10-12:617-622 (commit 16:00:17, pushed 16:00:22) |
| 16:10 + 240 s | 13:10 | Backstop, only if an arm is still alive: wait 240 s, then terminate. | `autostart.py:95`, `:109`, `:917-939` | log 10-05:649-650 (terminated 16:14:02) |
| 16:30 | 13:30 | Last repetition of the task (05:30 + 8 h). Past the cutoff, so it only runs the **evening stranded pass** and exits. | `autostart.py:1047-1052` | log 10-12:623-628 |

**After daylight saving ends on Sunday 2026-11-01**, the same local times map to ET one hour
earlier: the task fires at 07:30 ET and its 8-hour repetition window ends at **15:30 ET**, not
16:30. The start is safe (it is computed in ET). Two things are not obviously safe, and neither
has ever been exercised because the lab has only ever run in summer time:

- There will be no invocation after the close, so the evening stranded pass disappears
  **(inferred)**. The comment at `autostart.py:1048` ("the task repeats until 16:30 ET") is
  true only in summer.
- `StopAtDurationEnd = True` tells Task Scheduler to stop a running instance when the
  repetition window ends. In winter that moment is 15:30 ET, 25 minutes before the flatten.
  Whether it would actually kill the supervisor and both runners is **untested (inferred)**.
  See question 1 in §10. The first affected session is Monday 2026-11-02.

### 3.2 What repeats through the day, and why

| what | how often | why | where |
|---|---|---|---|
| The scheduled task | every 30 min, 05:30–13:30 MST | so a machine that booted late, or a supervisor that died with its whole process tree, still gets the lab running | `autostart.py:1023-1025` |
| …but it is normally refused | — | Task Scheduler's `IgnoreNew` refuses a second instance while the first is alive (last result `0x800710E0` at 06:30 on 10-13). On a clean day the log shows only two `autostart invoked` lines, 08:30 and 16:30. The Python-level lock (`lock.py`) is the second line of defence. | log 10-12:2, :624 |
| Supervisor loop | 5 s | child liveness, restart timers, suspend detection | `autostart.py:975` |
| Terminal watchdog | 60 s, 09:25–16:00 | a terminal that wedges while both arms stay alive was never restarted before | `autostart.py:465-473` |
| Runner tick | 5 s / 15 s | poll bars and quotes | `runner.py:308` |
| Frozen runner retrying its catch-up | every idle poll (15 s) | history may come back any moment | `runner.py:288-294` |
| Checkpoint `positions_open.json` | after every tick | so a restart resumes exactly the book it held | `runner.py:306-307`, `store.py:291-322` |
| `ledger.reconcile` | every invocation | only it can write "the lab never ran that day" | `autostart.py:1028-1029` |
| Gap-recovery lookback | every close, last 7 days | a recovery that failed because the network was still down gets retried | `autostart.py:462`, `:640-641` |
| Stranded pass | 3 windows: 08:30, after close, 16:30 | the network may only return hours later | `autostart.py:1044`, `:1050`, `:1214` |
| Signals roll | each runner start | keep `signals.jsonl` under the git host's 100 MB file limit | `store.py:167-220` |

### 3.3 Start-up checks in detail

1. **Lock** (`lock.py:44-74`). An OS-level byte lock on `<lab_dir>/autostart.lock`, released by
   the kernel however the process dies. Each runner takes its own (`runner.py:644-648`,
   `shares_runner.py:685-689`) and exits with code 4 if it is already held; the supervisor
   treats 4 as "do not restart" (`autostart.py:953-956`).
2. **Terminal** (`autostart.py:212-219`): "up" means `/v3/stock/snapshot/quote` returns 200.
   If not, `start_terminal` first evicts any Java process that holds port 25503 without
   answering (`:222-274`), launches the jar, waits up to 90 s, then waits up to 45 s for
   history (`:327-347`). If it was already up, history is still checked, for up to 180 s
   (`:1060-1081`, `feed.py:50`).
3. **Preflight** (`preflight.py`), run as a subprocess with `--ignore-exposure`
   (`autostart.py:383-389`): firewall exposure, quote freshness per symbol, 0DTE chain per
   symbol, and three prior sessions of at least 300 one-minute bars per symbol
   (`preflight.py:282-300`). The supervisor reads its **text output** and decides:
   - any `[FAIL]` line that is not tagged `EXPOSURE:` or `OPTIONS:` → abort every arm
     (`autostart.py:408-426`, `:1108-1111`);
   - `OPTIONS:`-tagged failures on symbols the options arm trades → run shares alone
     (`:1113-1124`); on other symbols → ignored (`:434-459`, `:1104-1107`);
   - any line containing `DELAYED` → abort (`:1126-1130`);
   - exposure → warning only (`:1131-1133`).
4. Before the open every quote reads "UNVERIFIABLE", so the real freshness verdict comes from
   the 09:33 re-check (`autostart.py:1092-1097`, `:1164`).

### 3.4 The outage story

Four mechanisms, in the order they act. Each one only handles what the previous one could not.

**A. Inside the tick: skip, do not guess.** A failed feed call raises `FeedOutage`; the runner
records an outage row and keeps looping (`runner.py:298-302`). Stale bars, stale quotes and
stale option quotes are refused where they are used (other slice; `runner.py:52`, `:73`).

**B. `catchup.py`: freeze, then replay the missed minutes.** After more than 90 s without a
good tick (`catchup.py:50`, `:87-89`) the runner stops acting: no entry, no exit. The book
stays on disk. When the vendor's history is reachable again the missed minutes are replayed
**one minute at a time through the runner's own `_tick`**, against a feed that serves history
as of each minute (`catchup.py:92-144`). Exits fire at their real minute and price and carry
`"exit_mode": "caught_up_from_history"` plus the window (`:104-108`). **Entries stay frozen**
during the replay: a signal that fired while the runner was away was never decided before its
price existed, so it is not the forward test's to take (`:27-29`; `runner.py:373-376`). If
history is still unreachable it writes a `catch_up_blocked` outage and stays frozen
(`catchup.py:129-131`). A restarted process does the same from the checkpoint's
`managed_through` stamp (`catchup.py:57-71`, `runner.py:270-274`).

**C. `gap_recovery.py`: after the close, re-price exits a gap stole.** Run by the supervisor
in a subprocess (`autostart.py:631-660`). For positions that have a trade row and were open
when an **unmanaged stretch** began. An unmanaged stretch is a host-suspend window minus
whatever catch-up already replayed (`gap_recovery.py:92-149`, `:487-491`). The whole session is
replayed on history through the runner's exit code; positions open at the gap are managed only
from the gap's start (`:267-289`). It first checks itself: exits live *did* record must come
out by the same rule on at least 90% of at least 20 checks (`:196-222`). It writes a correction
only when `verdict()` says the exit was stolen (`:157-193`). Output goes to
`trade_corrections.jsonl`; `trades.jsonl` is never touched (`:554-568`).

**D. `stranded.py`: positions with no exit row at all.** Rebuilds each position from the
runner's checkpoint, manages it on history from the minute after `managed_through`, and
**appends flagged rows to `trades.jsonl`** (`stranded.py:113-170`). Same self-check idea.
Never touches today's book before 16:05 (`:52`, `:95-96`) and does nothing while a runner
still holds its lock (`autostart.py:719-725`).

What happens in each scenario:

| scenario | during the session | at and after the close | evidence |
|---|---|---|---|
| **Network drops, comes back** | Ticks fail; after 90 s both arms freeze. Watchdog may restart the terminal. On return, catch-up replays the gap; exits at their real minute; entries skipped and counted in a `caught_up` event. | Nothing more is needed if catch-up covered it. | log 10-09:399-405 (2-minute gap); `catchup.py:136-138` |
| **Host suspends, wakes before the close** | Nothing runs while asleep. On wake the supervisor sees more than 90 s of wall clock missing from a 5 s sleep and writes a `host_suspend` outage into both arms' `outages.jsonl` (`autostart.py:153-185`). Runners freeze and catch up as above. | Ledger `PARTIAL` (`:1194-1206`). Gap recovery looks only at minutes no catch-up replayed. | log 10-08:601-610; ledger line for 10-08 |
| **Host wakes after 16:00** | — | Catch-up replays up to the close, including the 15:55 flatten at its real minute (`catchup.py:97-98`). Anything still open is closed on the **last in-session mark**, never a fresh after-hours quote, and flagged `_after_close_mark` (`runner.py:565-580`, `shares_runner.py:622-637`). If past 16:10 the supervisor waits 240 s before terminating (`autostart.py:917-939`). Gap recovery then re-prices rows closed at or after 16:00 or on a stale mark (`gap_recovery.py:225-237`). | `research/incident_2026-09-25_host_suspend_at_close.md` |
| **One arm's process dies, supervisor alive** | Restart with backoff 5/15/30/60/120 s, at most 20 times, checking the terminal first (`autostart.py:111-112`, `:946-974`). The new process reloads the book only if its stamped session date is today (`store.py:324-363`), rebuilds the per-day caps from the decision record (`runner.py:230-243`), and catches up from the last managed minute. No restart after 15:55. | Normal. | `research/incident_2026-09-15_shares_arm_gap.md` |
| **Whole process tree dies** | The next 30-minute task repetition starts a fresh `autostart`; before 15:30 ET it relaunches the session, after that it does not. | Positions keep no exit row. Ledger stays `OPENED`, then `INTERRUPTED` the next morning. The stranded pass fills the exits. **No archive commit that day**; the next day's commit carries the files. | log 10-07:553-568; log 10-08:3, :10-14; no "Live lab record for 2026-10-07" commit in `git log` |
| **Network dies before the close and stays down** | Both arms stay frozen. A frozen runner never reaches its own 16:00 exit, because the frozen branch `continue`s before the close check (`runner.py:288-297`). Watchdog burns its 6 restarts and gives up. | Supervisor terminates both arms at about 16:14. **No exit rows, no daily summary.** Ledger still says `COLLECTED`. Archive commits locally; the push fails and waits for the next successful push. Stranded pass fills the exits when the terminal is next reachable. | log 10-05:584-658; `live_lab_data/stranded_recovery.jsonl` |

**What is recovered:** exits only, and only for positions that were genuinely open.

**What is deliberately NOT recovered:**

- **Entries.** A missed signal is never added, anywhere (`catchup.py:27-29`,
  `gap_recovery.py:19-21`, `stranded.py:20`). A "filled-in day" exists only as a replay in
  `live_lab_backfill/`, which is not evidence.
- **Checkpoints written before `managed_through` existed (before 2026-10-02).** Reported on
  every invocation and never resolved (`stranded.py:33-34`, `:106-108`). Today that is shares
  2026-09-08 (19 positions) and 2026-09-10 (32 positions): log 10-13:4-5.
- **2026-09-01's end-of-day positions.** The snapshot was overwritten before the archive folder
  existed (`reconstruct_eod.py:25-27`).
- **Any day whose self-check fails.** Nothing is written; for a stranded session the refusal is
  recorded and not retried (`stranded.py:37-38`, `:89`).
- **A live exit made after the runner was managing again.** That is an observation; a history
  replay may not override it unless the replay exits inside the unmanaged stretch, or by a
  different rule from a real rebuild (`gap_recovery.py:169-193`).

### 3.5 The end-of-session commit (`archive.py`)

Called from `autostart.py:1227-1232` after the ledger line, gap recovery and the stranded pass,
so what it commits is the completed day.

Exactly what it does (`archive.py:150-238`):

1. `git -C <REPO> add -- <lab_dir>` where `REPO` is **the checkout the code itself lives in**
   (`archive.py:70`) and `lab_dir` is `live_lab_data` (absolute path from `store.py:36`). One
   explicit path. Never `git add -A`.
2. If nothing is staged under that path: log "nothing to archive", unstage, return.
3. Commit with message `Live lab record for <day>` and identity
   `live-lab-archive <live-lab-archive@localhost>`, passed per command so a machine with no git
   identity still commits (`:73-74`, `:184-188`).
4. Push `HEAD` to `origin`, to **the branch that is currently checked out** (`:92-96`,
   `:201-210`). Four attempts with 2, 8, 20 s waits. A detached HEAD commits locally and does
   not push.
5. Only if git says the remote moved ahead (non-fast-forward): one `fetch`, one `merge`, one
   more push. Skipped if tracked files are modified; aborted on conflict. Never rebase, force
   or amend (`:116-147`, `:221-227`).
6. Never raises and never changes the session's exit code.

What `git add -- live_lab_data` actually picks up, from `.gitignore:83-104` and `git ls-files`:

| committed | not committed (gitignored) |
|---|---|
| `FREEZE.json`, `config/`, `daily/`, `events.jsonl`, `outages.jsonl`, `signals.jsonl`, `signals_archive/`, `trades.jsonl`, `trade_corrections.jsonl`, `stranded_recovery.jsonl`, `session_ledger.jsonl`, the same under `shares/`, and **`shares/recovery_archive/`** | `logs/`, `positions_open.json` (both arms), `*.lock`, `*.lock.info`, `bars_cache/`, `quote_cache/`, **`live_lab_data/recovery_archive/` (the options arm's)**, a few named research output files |

Two consequences worth knowing:

- **Anything new dropped into `live_lab_data/` that is not gitignored is committed and pushed
  at the next close.** Three research outputs already live there and are tracked
  (`six_lines_QQQ.json`, `breakout_sweep_QQQ.json`, `early_r1_hold_QQQ.json`).
- The options arm's checkpoint archive is ignored while the shares arm's is tracked. Reason
  not recorded. It means the options checkpoints that `stranded.py` depends on exist only on
  the laptop.

**What goes wrong if the main checkout is on the wrong branch.** The lab runs whatever is
checked out in `C:\Users\chaitanyakharche\Documents\stock`; nothing pins a branch
(`run_lab.cmd:6-7`).

- *A feature branch:* that branch's code trades the session, and the day's record is committed
  onto and pushed to that branch, not `main`.
- *A stale branch:* checking out an old commit **rewrites the working copies of the tracked
  record files to their old contents**, because they are tracked **(inferred from git
  behaviour; the memory note `lab-runs-from-main-checkout.md` records exactly this on
  2026-09-24: the tree held `live_lab_data` as of 09-10 and had no `archive.py`)**. A session
  run from there appends to a truncated history and restarts sequence numbers from the old
  maximum (`store.py:136-163`) **(inferred)**.
- *Code edited mid-session:* it becomes live at the next subprocess. That is the 09:33
  preflight (`autostart.py:383-388`), any arm restart (`:848-858`), and at the close the gap
  recovery, the stranded pass and the lazily imported archive (`:646-648`, `:753-755`,
  `:1229`).
- *Someone else pushes to `origin/main` during the day:* the 16:00 push is rejected, and the
  one merge-and-retry **merges that remote work into the lab's checkout** (`archive.py:132`).
  Pushing code to `origin/main` is therefore a deployment to the lab **(inferred)**.

Current state, checked from local refs without fetching: main checkout on `main` at `6f9c1ab`,
level with `origin/main`; the last four closes logged `[archive] pushed to main`.

### 3.6 Backfill and replay: what they are for and how they stay separate

There are three different things with "replay" or "backfill" in the name. They are easy to
confuse.

| tool | what it does | writes where | part of the forward test? |
|---|---|---|---|
| `replay.py` | The **pre-freeze gate**. For every bar, evaluates each setup against a session built bar by bar and against one rebuilt in one shot, and flags any difference (hidden state, order dependence). Also measures signal frequency. Opens no positions, needs no options data. | stdout only | No. A check on the code. |
| `backfill.py` | Runs the **current config** of either arm over past sessions through the **real runner classes**, with only the feed (history as of a simulated clock) and the clock swapped (`backfill.py:27-40`, `:311-395`). Answers "what would a runner with a perfect connection have logged?" | `live_lab_backfill/options/` and `/shares/` | **No.** |
| `replay_audit.py` | Pairs each live trade with its replay twin and names the first recorded cause for every trade that did not pair (late start, host asleep, frozen, feed down, …) (`replay_audit.py:9-35`). Never edits the live record. | `live_lab_backfill/<arm>/audit.json` | No. |
| `backfill_daily.py` | Unrelated to the above. Rebuilds a missing `daily/<date>.json` rollup from `trades.jsonl` for a session that ran but died before writing one. Marks it `"backfilled": true` with a derived cause and the evidence (`backfill_daily.py:44-131`). | `live_lab_data/…/daily/` | It is a summary of the live record, flagged. |
| `walkforward.py`, `sharewf.py` | Historical walk-forward studies over the frozen setups (underlying only; QQQ shares with next-bar fills at real quotes). | caches and result files under `live_lab_data/`, all gitignored | No. "Results here CANNOT change a frozen rule" (`walkforward.py:16-18`). |

Why a replay can never count: the prices already existed when it ran, so the
decide-before-price guarantee cannot hold, however faithfully the code is reused
(`backfill.py:9-14`).

How it differs from live, stated in the file (`backfill.py:42-51`): one-minute quote snapshots
instead of 5-second polls; the vendor's final, revised bars instead of first prints; no
outages; no open interest. The measured consequence: totals are comparable, single trades are
not. On clean hours about 3 in 4 live trades have a replay twin
(`research/replay_22_sessions.md` §4).

What keeps backfilled data from being mistaken for the forward test:

1. **Location.** Output is under `live_lab_backfill/`, outside `live_lab_data/`
   (`backfill.py:73-74`), pinned by `backfill_test.py:95-98`.
2. **A tag on every row.** `"provenance": "BACKFILL"` on every trade, decision, fill, skip,
   event and outage (`backfill.py:275-289`). I counted 1,436 of 1,436 rows tagged in
   `live_lab_backfill/options/trades.jsonl` and zero in either live `trades.jsonl`.
3. **A manifest.** `BACKFILL.json` in each arm's folder says "NOT PROSPECTIVE"
   (`backfill.py:384-394`).
4. **Every replayed log says so** on its first line (`backfill.py:356-357`).
5. **Convention.** The dashboard, the checkpoint diagnostic and the promotion bar read
   `live_lab_data/` only (stated at `backfill.py:12-13`; `checkpoint_diagnostic.py:58`).

**The weak point:** backfill rows carry the **same config hash** as live rows
(`ccd00ac71f8243e9`, `7c6414fc549315f3`; compare `live_lab_backfill/*/BACKFILL.json` with
`live_lab_data/FREEZE.json`). The "accepted config hash" filter, which is the only filter the
forward test ever applies, would let them through. Only the folder and the `provenance` field
separate them. Never concatenate the two, and never point a tool's `--lab-dir` at
`live_lab_backfill/`.

What the committed backfill covers: both arms, 2026-08-28 to 2026-09-29, 22 sessions
(`BACKFILL.json`; 22 logs per arm).

---

## 4. Map

Launch and supervision

| file | what it is for |
|---|---|
| `run_lab.cmd` | The scheduled task's action: cd to the main checkout, run `autostart` on the pyenv Python. |
| `autostart.py` | The supervisor. Start-up checks, child processes, watchdog, suspend detection, ledger, post-close recovery, archive. |
| `lock.py` | OS-level single-instance lock; `held()` probes without relabelling. |
| `preflight.py` | Feed and structure checks whose text output the supervisor gates on. |
| `clock.py` | The single source of "now" in exchange time; session boundary constants. |
| `ledger.py` | One append-only line per session state change; coverage with a calendar denominator. (Read because `autostart` calls it throughout.) |
| `restart_terminal.py` | Manual "nudge the terminal" tool from 2026-09-15; superseded in normal use by the watchdog. |
| `archive.py` | Commit and push `live_lab_data` at the end of the session. |

Recovery

| file | what it is for |
|---|---|
| `catchup.py` | In-session: freeze after 90 s away, replay missed minutes from history, entries frozen. |
| `gap_recovery.py` | Post-session: re-price exits stolen by an unmanaged stretch; append corrections; audit, retract, reinstate. |
| `stranded.py` | Post-session: exits for checkpoint positions that have no trade row. |
| `reconstruct_eod.py` | One-off from 2026-09-03/04: rebuilt the end-of-day exits the supervisor destroyed on 09-02, 09-03 and three on 09-04. **Hard-coded; do not run again** (§7.3). |
| `backfill_daily.py` | Rebuild a missing daily rollup from the trade log, flagged. |

Replay and diagnostics

| file | what it is for |
|---|---|
| `replay.py` | Pre-freeze path-independence gate and signal-frequency measurement. |
| `backfill.py` | Full-session replay through the real runners into `live_lab_backfill/`; also `HistoryFeed` and `SimClock`, which catch-up and both recovery tools reuse. |
| `replay_audit.py` | Explains every live-versus-replay difference from what live recorded. |
| `checkpoint_diagnostic.py` | The 50/100-trade pipeline-health checkpoint. Apparatus only, no effect claims. |
| `attribution.py` | Descriptive funnel: was direction right, and if so did the option pay. |
| `walkforward.py` | Historical walk-forward of the frozen setups on the underlying. |
| `sharewf.py` | Blind walk-forward of QQQ shares with next-bar fills at real quotes. |

Tests (what each guards, from names and docstrings)

| file | guards |
|---|---|
| `autostart_guard_test.py` | The three 2026-09-10 suspend guards: grace window, suspend detection, wake lock. Script style, in CI. |
| `watchdog_test.py` | Watchdog: no restart when offline *while the terminal answers*, restart when wedged, restart cap, idle outside hours. |
| `feed_resilience_test.py` | Upstream-versus-terminal error classification, bounded waiting, outage episodes; three source-inspection checks on `autostart`. |
| `opening_window_test.py` | Runners start before the open; the 09:33 gate still fires once; an abort is not overwritten; `PARTIAL` means started after 09:30. |
| `foreign_options_gate_test.py` | A stale chain on a symbol the options arm does not trade no longer aborts. |
| `preflight_gate_test.py` | The token contract between preflight's output and the supervisor's gate. Script style. |
| `gap_recovery_test.py` | 30 checks: gaps, affected rows, append-only, refusal, every `verdict` branch, retract and reinstate. |
| `catchup_test.py` | Freeze threshold, minute-by-minute replay, labels, entries refused while frozen, decided-once. |
| `stranded_test.py` | Finding stranded positions, today's book untouched, flagged append-once, refusal not retried, all the supervisor hook behaviours. |
| `checkpoint_managed_test.py` | A restart resumes from `managed_through`, not `saved_at`. |
| `archive_test.py` | Real throwaway git repos: only the lab directory is committed, no force or rebase, conflict abort, never raises. |
| `signals_archive_test.py` | Rolling signals loses, duplicates and reorders nothing; sequence resumes above the archive. |
| `backfill_test.py` | `HistoryFeed` never serves the future; output is outside the live record; reconcile bins every trade once. |
| `bar_cache_test.py` | The bar cache admits exactly what an uncached feed would, on the same tick. |
| `replay_audit_test.py` | Each unpaired trade gets the cause live recorded. |

CI (`.github/workflows/tests.yml`) runs `pytest` plus the script-style tests, and prints ledger
coverage read-only (line 97). It never runs `--reconcile`.

---

## 5. Design decisions and the reason for each

| decision | reason, and where it is written |
|---|---|
| Start at a fixed **exchange** time, not a fixed local time | The machine is on MST and Arizona has no daylight saving; a local schedule would drift twice a year (`autostart.py:7-10`, `run_lab.cmd:4-5`). |
| Start the runners **before** the open and check freshness while they run | Holding them behind a post-open check blinded both arms to the open for every session from 08-31 to 09-24. Safe only because staleness is now refused where a quote is used (`autostart.py:72-86`). |
| Detect holidays from data, not a calendar | "a hardcoded calendar that would go stale" (`autostart.py:11-13`). Preflight's warm-up lookback, by contrast, *does* use the exchange calendar after Labor Day broke it (`preflight.py:271-281`). |
| LAN exposure warns, never aborts | It is a security issue, not a data-integrity one; blocking would only lose sessions (`autostart.py:16-19`, `:411-412`). |
| Gate on stable tokens (`EXPOSURE:`, `OPTIONS:`), never on prose | A reworded check once silently turned a warning into an abort (`preflight.py:43-53`, `preflight_gate_test.py:8-18`). |
| An options failure must not stop the shares arm | On 2026-09-08 four option-chain 403s aborted both arms though every underlying was fine (`autostart.py:414-419`). |
| Stop restarting at 15:55 but do not kill until 16:10, then wait 240 s more | Killing at 15:55 deleted exactly one exit type (`eod`) for three sessions; killing one second after a wake cost 32 positions (`autostart.py:89-96`, `:919-924`). "Killing a healthy shutdown is the more expensive mistake." |
| OS file lock, not a PID file | A killed process never cleans up a PID file; the kernel releases a lock however the process dies (`lock.py:7-10`). |
| Recovery tools run in **subprocesses** | The replay patches the store's clock, which must never leak into the supervisor (`autostart.py:636-637`, `:715-716`). |
| Recovery replays through the runner's **own** exit code | "The code that made every other exit today makes these" (`gap_recovery.py:26-30`). |
| Self-check before writing | "A replay that cannot reproduce the exits it can see has no business supplying the ones it cannot" (`gap_recovery.py:31-34`). |
| Gate on the exit **rule**, report the **minute** | The vendor revises bars after the fact, so a stop can fire one bar earlier or later on history. The first gate (rule and minute) measured 59% and refused; rule agreement was 97%. Changed after seeing it fail, and recorded as such (`gap_recovery.py:36-45`). |
| Corrections are appended, never edits; the latest line per position wins | "A record that silently rewrites itself is not a record" (`gap_recovery.py:22-25`, `:591-592`). |
| Only **exact** recoveries apply by default | `apply_corrections(include=("recovered",))`; a best-available view must be asked for and declared (`gap_recovery.py:584-589`). |
| The live day's history is never disk-cached | A cached partial day would later be served to backfill and recovery as the whole session (`backfill.py:111-114`). |
| `managed_through` separate from `saved_at` | A frozen runner keeps saving while managing nothing; on 10-02 that hid an hour of unchecked stops (`store.py:298-303`). |
| After 16:00 never take a fresh quote | An after-hours bid "sold" two XLP longs 6.2% below entry on 09-25 (`runner.py:566-570`). |
| Archive on by default, explicit pathspec, merge only | The manual step produced a six-session hole; a broad add on an unattended machine holding API keys is how a credential reaches a remote (`autostart.py:998-1001`, `archive.py:23-28`). |
| Commit and push are separate outcomes | The commit needs no network and is what makes the record durable; the push is best effort (`archive.py:56-62`). |
| Backfill regenerates whole, never appends | (`backfill.py:338`). Reason beyond the comment not recorded. |
| The watchdog checks served history, not a status string | "a status string is a claim and served data is evidence" (`autostart.py:516-518`). |
| Per-call retry stays about 7 s; long waits only in warm-up | A tick that blocked for 180 s would stall all 15 symbols (`feed.py:45-49`). |

---

## 6. Records, formats, invariants, frozen things

### 6.1 The files this slice writes

| file (under `live_lab_data/`, and again under `shares/`) | written by | notes |
|---|---|---|
| `session_ledger.jsonl` (top level only) | `ledger.record` | One line per state change. States ranked `MISSED < INTERRUPTED < ABORTED < OPENED < PARTIAL < COLLECTED < NOT_A_SESSION` (`ledger.py:85-86`); a day reads as its **best** state. Every call appends, so `attempts` counts real invocations. |
| `outages.jsonl` | runners, and the supervisor for `host_suspend` | `host_suspend` rows carry `gap_sec`, `from_et`, `to_et`, and a `detail` string that `find_gaps` parses with a regular expression (`gap_recovery.py:70`, `:98`). **Do not change the wording at `autostart.py:161-162`.** |
| `events.jsonl` | runners | `caught_up` rows (`since`, `until`, `minutes`, `exits`, `entry_checks_skipped`) are what gap recovery subtracts from a suspend. |
| `trades.jsonl` | runners; `stranded.py`; once, `reconstruct_eod.py` | See §6.2 for the flags. |
| `trade_corrections.jsonl` | `gap_recovery.py` only | Kinds: `gap_recovery` (status `recovered`, `recovered_approximate`, `estimated`, `unrecoverable`, `retracted`) and `gap_recovery_run` (one marker per completed run, which also stops the supervisor redoing a settled day: `autostart.py:617-619`). |
| `stranded_recovery.jsonl` | `stranded.py` | One line per resolved or refused session. |
| `positions_open.json` | runners, atomic replace | `saved_at`, `managed_through`, `session_date`, `positions`. Gitignored. |
| `recovery_archive/positions_open_<session>_<stamp>.json` | `store.load_open_positions` when the stamped date is not today | The only surviving copy of a dead session's book. |
| `daily/<date>.json` | runners at exit; `backfill_daily.py` | A rollup. `trades.jsonl` is authoritative (`checkpoint_diagnostic.py:185-200`). |
| `logs/<date>.log` | the supervisor's tee | Gitignored. Main checkout only. |

Counts as of commit `6f9c1ab` (my own `grep` counts, not a tool's output):

| | options | shares |
|---|---|---|
| rows in `trades.jsonl` | 1,609 | 3,856 |
| `"exit_mode":"caught_up_from_history"` | 6 | 18 |
| `"exit_mode":"reconstructed_from_history"` (stranded) | 42 | 120 |
| `"exit_reason":"eod_reconstructed"` (2026-09 one-off) | 30 | 13 |
| `"reconstructed":true` in total | 72 | 133 |
| corrections `recovered` / `recovered_approximate` / `estimated` | 99 / 0 / 0 | 160 / 101 / 55 |
| corrections `retracted` | 78 | 209 |

### 6.2 The rule for what may be written after the fact

The brief's version was "write only stolen exits; corrections go in `trade_corrections.jsonl`
rather than editing rows". The code confirms the spirit and needs one correction:

1. **No existing row is ever edited or deleted.** True everywhere I read.
2. **An exit that live recorded but a gap corrupted** is corrected in
   `trade_corrections.jsonl`, and only if it was stolen (`gap_recovery.py:157-193`, `:524-535`).
   Retractions and reinstatements are also appended lines (`:682-703`).
3. **Correction to the brief: an exit for a position that has no row at all is appended to
   `trades.jsonl` itself**, not to the corrections file, flagged
   `"exit_mode": "reconstructed_from_history"`, `"reconstructed": true` and a `reconstruction`
   block naming the checkpoint, the window and the self-check (`stranded.py:155-165`). The
   2026-09 one-off did the same with `"exit_reason": "eod_reconstructed"`
   (`reconstruct_eod.py:160-167`). "One filter excludes them" (`stranded.py:31-32`).
4. **Entries are never written after the fact.**
5. **Summaries may be rebuilt** if flagged (`backfill_daily.py:17-18`). **Ledger lines may be
   added** by `reconcile`, on the lab machine only (§7.3).

### 6.3 Things that must not change

- `live_lab_data/FREEZE.json`, `shares/FREEZE.json`, `config/*.json`: the frozen experiment.
  Not this slice's to explain, but every tool here filters on their accepted hashes.
- `--symbols QQQ SPY` in `run_lab.cmd`. The options universe is inside the config hash;
  widening it forks the hash (`autostart.py:839-842`).
- Constants deliberately **outside** the hash that still change what gets recorded:
  `START_AT`, `FRESHNESS_AT`, `HARD_STOP`, `SUSPEND_JUMP_SEC`, `CATCHUP_AFTER_SEC`,
  `MIN_AGREEMENT`, `MIN_CHECKS`, `SETTLED_AT`. Changing one is a change to how the record is
  made and belongs in an incident note, as every past change has been.
- The append-only files. Never rewrite, sort, deduplicate or reformat them.

---

## 7. Failure modes, incidents, traps

### 7.1 Every guard in `autostart.py`

"No test found" means a search of every `*test*.py` in the repository found nothing that
exercises it; it is not a claim that it is broken.

| # | guard | incident or date that caused it | where | test that pins it |
|---|---|---|---|---|
| 1 | Sleep until 09:05 **ET**, computed live | Design: MST machine, no DST | `autostart.py:7-10`, `:64`, `:350-367`; `clock.py:27-29` | `opening_window_test.py::test_the_start_leaves_room_for_the_shares_warmup` pins the margin; no DST test found |
| 2 | Weekend exit, recorded `NOT_A_SESSION` | Design | `:1018-1021` | No test found |
| 3 | Holiday detected from data | Design | `:370-380`, `:88`, `:1083-1090` | No test found. See §7.2 |
| 4 | Do not start a session after 15:30 | Design | `:87`, `:354-356` | `stranded_test.py::test_main_settles_stranded_positions_before_the_wait_and_in_the_evening` (the evening branch) |
| 5 | Start the terminal if it is not answering | 2026-08-31 hardening | `:327-347`, `:1054-1059` | No direct test found |
| 6 | Evict a terminal that holds the port without answering; only ever kill Java | 2026-09-08: a terminal from 09-03 kept the socket; every retry launched a doomed second copy | `:222-274` | No test found (others stub it) |
| 7 | After the terminal answers, wait until it serves **history** | 2026-09-08: answered at 10:30:29, preflight ran at 10:30:30, first two symbols got 503 | `:277-324` | `feed_resilience_test.py::test_the_probe_must_exercise_the_upstream_not_the_terminal` (the feed's probe) |
| 8 | Check the upstream even when the terminal was already up; wait up to 180 s | 2026-08-28 lost a session; 202 of those 503s on 09-01 (`e7bf6de`, 09-17) | `:1060-1081` | `feed_resilience_test.py::test_autostart_checks_the_upstream_even_when_the_terminal_is_already_up` |
| 9 | That wait is suspend-aware | Merge of two branches, 09-18 (`6c4ce64`) | `:313-320` | `feed_resilience_test.py::test_the_upstream_wait_is_suspend_aware` |
| 10 | Preflight verdict read by token, not prose | 2026-09-04 rewording (`6985258`) would have aborted every session from 09-08 | `:57-59`, `:408-426` | `preflight_gate_test.py` checks 1–2 |
| 11 | Exposure never aborts; `--ignore-exposure` passed | 2026-09-04 | `:383-389`, `:1131-1133` | `preflight_gate_test.py` check 3 |
| 12 | Preflight sees the union of both universes | 2026-09-05 broadening to 15 names | `:392-405` | `preflight_gate_test.py` check 4 |
| 13 | Options-only failure disables the options arm, shares continues | 2026-09-08: subscription dropped to `Options: FREE`, both arms aborted | `:414-431`, `:1113-1124` | `opening_window_test.py::test_gate_verdict_keeps_the_old_precedence` |
| 14 | Option checks on symbols the options arm does not trade are dropped from the gate | 2026-10-02: three restarts aborted on "XLY options DELAYED by ~21 min" (`e4174e2`) | `:434-459`, `:1104-1107` | `foreign_options_gate_test.py` (4 tests) |
| 15 | A `DELAYED` feed aborts every arm | Design | `:783-784`, `:1126-1130` | `opening_window_test.py::test_a_delayed_feed_at_the_open_ends_the_day_aborted` |
| 16 | Runners start pre-open; freshness gate runs once at 09:33 while they are live | 2026-08-31 → 09-24 blind open (`88bae1e`) | `:64-86`, `:880-883`, `:899-909`, `:1135-1164` | `opening_window_test.py` (18 tests) |
| 17 | One verdict function for both checks | 09-24 | `:773-791` | `test_gate_verdict_keeps_the_old_precedence` |
| 18 | An abort is not overwritten by `COLLECTED` | 09-24; the ledger ranks by best state | `:1186-1189` | `test_a_delayed_feed_at_the_open_ends_the_day_aborted` (asserts `ABORTED` and `rc == 3`) |
| 19 | Supervise: restart a dead arm, backoff 5–120 s, cap 20 | 2026-08-31 audit (`4956008`): a runner that died at 11:00 silently ended the day | `:111-112`, `:862-979` | No test of the restart path found in the current tree |
| 20 | Exit code 4 (lock held) is not a crash | 08-31 | `:60`, `:953-956` | No test found |
| 21 | After 15:55 stop restarting, do **not** kill | 2026-09-01..03: both arms terminated at 15:55:02 (`264068e`) | `:89-94`, `:914-916`, `:940-944` | `opening_window_test.py::test_options_verdict_leaves_shares_running` (shares survives to the backstop) |
| 22 | 16:10 backstop with a 240 s grace window | 2026-09-10: backstop fired one second after wake; 32 positions lost (`dbf0bba`) | `:95-96`, `:109`, `:917-939` | `autostart_guard_test.py` checks 1 and 2 |
| 23 | Ask Windows not to idle-sleep | 2026-09-10 | `:105-150`, `:1036-1038` | `autostart_guard_test.py` check 4 (Windows only) |
| 24 | A suspend is written into both arms' outage record | 2026-09-10: the only evidence was a Windows event | `:108`, `:153-185`, `:942`, `:975` | `autostart_guard_test.py` checks 3a, 3b |
| 25 | Wake lock released only after the archive push | 09-18 merge | `:143-150`, `:1234-1239` | `feed_resilience_test.py::test_the_wake_lock_is_released_after_the_archive_not_before` |
| 26 | Single instance, lock taken before the wake lock | 08-31 | `:1031-1038`; `lock.py` | `stranded_test.py::test_the_lock_probe_sees_a_holder_and_does_not_relabel_it` (the probe); no test of mutual exclusion found |
| 27 | Ledger line at every exit path; `reconcile` on entry | 2026-09-08 | `:1020`, `:1028-1029`, `:1058`, `:1077`, `:1087`, `:1110`, `:1120`, `:1128`, `:1156`, `:1182`, `:1202-1206` | `ledger_test.py` (the ledger itself) |
| 28 | Values for the ledger computed defensively before the call | 2026-09-09: a `NameError` in an argument list killed the session (`919e47a`) | `:1168-1184` | No test found |
| 29 | stderr tee'd to the log | 2026-09-09: the traceback went nowhere under Task Scheduler | `:1009-1013` | No test found |
| 30 | Child output relayed through the tee | 08-31: runner lines existed only in the console window | `:805-820` | No test found |
| 31 | Each arm gets its own universe; the options one is never widened | 2026-09-05 | `:833-859`, `:984-989` | No test found |
| 32 | Terminal watchdog: restart when not answering twice, or answering with internet up and no history three times; cap 6 | Gap named 09-15 (`2b4e832`), built 09-28 (`494c314`) | `:465-596`, `:895-898` | `watchdog_test.py` (6 tests) |
| 33 | A day with a suspend during market hours is `PARTIAL` | 2026-09-25 read `COLLECTED` | `:603-609`, `:1190-1208` | No test found |
| 34 | Gap recovery after every session, 7-day retry, subprocess, never raises | 2026-09-25 (`1e7aeef`) | `:462-463`, `:612-660`, `:1210-1212` | `gap_recovery_test.py` covers the tool; no test of `_needs_recovery` or `recover_session_gaps` found |
| 35 | Stranded pass at three windows, with a deadline and a minimum time | 2026-10-05/06: 134 positions with no exit row (`53bffa3`) | `:700-770`, `:1040-1051`, `:1213-1214` | `stranded_test.py` hook tests |
| 36 | The stranded pass shuts down a terminal it launched, even one that never answered | 2026-10-07 16:30: left running, later collided (`HTTP 478`) (`1d295bc`) | `:663-697`, `:767-770` | `stranded_test.py::test_hook_kills_a_terminal_that_never_answered` |
| 37 | …and touches nothing while a runner holds its lock; kills only terminals started after its own launch | Review 2026-10-08 (`6405392`) | `:719-725`, `:688-689` | `test_hook_leaves_everything_alone_while_a_runner_holds_its_lock`, `test_kill_terminals_spares_a_terminal_started_before_the_launch` |
| 38 | Archive on by default; its result never reaches the exit code | 2026-09-16: six sessions only on the laptop | `:998-1005`, `:1216-1232` | `archive_test.py::test_autostart_archives_by_default_and_ignores_the_result` |
| 39 | Stale recovery refused: yesterday's book is archived, not reopened | 08-31 | `store.py:324-363` (described at `autostart.py:28-30`) | `checkpoint_managed_test.py::test_another_days_checkpoint_is_ignored` (for the catch-up stamp); none found for `load_open_positions` |

### 7.2 Traps found while reading (things that would bite whoever touches this next)

1. **The wake lock does not stop this laptop sleeping.** The log prints `HELD` every morning
   (log 10-12:5), and the host has still suspended on 09-11, 09-17, 09-18, 09-24, 09-25, 09-28,
   09-29 and 10-08 (13 `host_suspend` rows in `live_lab_data/outages.jsonl`). The machine has
   only Modern Standby, and a power-button or lid sleep cannot be refused from user space
   (`research/network_resilience.md:9-17`; `autostart.py:123-126`). Treat `HELD` as "asked",
   not "guaranteed".
2. **"rc=0" in the log means nothing for an arm that was alive at 15:55.** `_Child.rc` starts
   at 0 (`autostart.py:801`) and is only updated in the restart block (`:949-952`), which the
   loop never reaches after 15:55 (`:940-944`) or at the backstop (`:917-939`). On 10-05 both
   arms were *terminated* and the log still says `finished rc=0` (log 10-05:650-652), and the
   ledger carries `supervisor_rc: 0`. This is the project's standard bug shape.
3. **The watchdog does restart the terminal when the network is down.** Its docstring and its
   test say "never when the network is merely down" (`autostart.py:543`,
   `watchdog_test.py:1-5`). The internet check exists only on the "answering but not serving"
   branch (`:570-572`); the "not answering" branch restarts without it (`:561-565`). After a
   long enough drop the terminal stops answering at all, so on 10-05 the watchdog spent all six
   restarts in 24 minutes with the hotspot dead, then logged "leaving it for a human" until the
   close (log 10-05:584-648). The test passes because its offline case keeps `answering=True`
   (`watchdog_test.py:28-39`).
4. **A terminal launched with the network down may be left running by the watchdog**
   **(inferred)**. `_kill_terminals`, which exists for exactly that case
   (`autostart.py:668-672`), is called only from the stranded pass (`:770`). The watchdog path
   calls `_evict_stale_terminal`, which only finds a process that holds the port (`:592-594`).
5. **The holiday guard cannot fire on the normal path.** `is_holiday` returns `False` before
   09:45 (`autostart.py:372-373`) and the code reaches it at about 09:06. On Labor Day
   2026-09-07 the first invocation ran the session and aborted at 09:34 with "the feed is
   DELAYED"; a later invocation at 13:27 detected the holiday (that day's log). With today's
   code the runners would start, be stopped by the 09:33 gate, and a later repetition would
   record `NOT_A_SESSION` **(inferred)**. The next weekday closure is Thanksgiving, 2026-11-26
   (`trade_analysis/bulk_download/trading_days.py:96`).
6. **Never open `positions_open.json` from another process while a runner is live.** It is
   the target of an atomic replace; on Windows a reader makes the *writer* crash with
   `WinError 5`. That killed the shares arm at 10:51:09 on 2026-09-15 and cost 15 unmanaged
   minutes on 58 positions (`research/incident_2026-09-15_shares_arm_gap.md`). `stranded.py`
   and `reconstruct_eod.py` both read that file (`stranded.py:63-69`,
   `reconstruct_eod.py:56`, `:63-64`, `:129`). The supervisor guards its own call with a lock
   probe (`autostart.py:719-725`); **the command-line entry points do not**
   (`stranded.py:173-179`), so running either by hand during market hours can kill an arm,
   even in report-only mode **(inferred)**.
7. **`written_at` in `trade_corrections.jsonl` is laptop-local time, everything else is ET.**
   `gap_recovery.py:556`, `:581`, `:689` call `datetime.now()`. The automatic run after the
   10-08 close is stamped `2026-10-08T13:01:02` (16:01 ET). `stranded.py` uses exchange time
   (`stranded.py:157`).
8. **`seq` is not unique in `outages.jsonl`.** The supervisor builds a fresh store for each
   suspend row (`autostart.py:167`), and sequence recovery does not read `outages.jsonl`
   (`store.py:144`). Three `host_suspend` rows on 09-25 share `seq` 6390.
9. **The stranded gate is weaker than the gap-recovery gate.** `gap_recovery.gate` needs 20
   pooled checks at 90% (`gap_recovery.py:196-222`). `stranded.resolve` still uses the older
   rule: the exact pool alone decides, with no minimum number of checks
   (`stranded.py:144-152`, `gap_recovery.py:476-477`). The comment that the two "cannot drift
   apart" (`gap_recovery.py:456-457`) is no longer true. Every stranded run so far had 27 to
   141 exact checks (`stranded_recovery.jsonl`), so nothing written to date depends on this.
10. **A day's gap recovery is considered done as soon as either arm has any correction line
    for it** (`autostart.py:617-619`). The tool does shares first, then options
    (`gap_recovery.py:724`). If the options half crashed after the shares half finished, it
    would not be retried **(inferred; not observed)**.
11. **No standard reader applies the corrections.** Only `backfill.report` does
    (`backfill.py:536-544`). The dashboard, `attribution.py` and `checkpoint_diagnostic.py`
    read raw `trades.jsonl`. Any P&L total therefore depends on whether corrections were
    applied and with which `include`. Say which, every time.
12. **Daily summaries go stale.** Rows appended after the runner wrote its rollup (stranded
    exits, the 2026-09 reconstructions) are not in `daily/<date>.json`. Sum `trades.jsonl`
    by arm; never sum the dailies (`research/live_lab_checkpoint_100_diagnostic.md` §4).
13. **`backfill_daily.py` defaults disagree with each other:** `--lab-dir` defaults to the
    options directory and `--arm` to `SHARES` (`backfill_daily.py:136-137`), and it writes a
    field called `net` where the options runner writes `atm_net`.
14. **The preflight option-age check is one-sided.** It measures age against the `now` taken
    at the start of the run (`preflight.py:195`), so it printed "median quote age −51.2s" on
    10-12 (log 10-12:180) and would hide a delay of up to about 50 s. Flagged on 09-24 as
    follow-up 3 and still in the code.
15. **Paths follow the code, not the machine.** `store.DEFAULT_LAB_DIR`, `gap_recovery.LAB`,
    `backfill.ROOT`, `archive.REPO` and others are all "two folders above this file"
    (`store.py:36`, `gap_recovery.py:62-63`, `backfill.py:72`, `archive.py:70`). A tool run
    from a worktree reads and writes **that worktree's copy** of the record, while still
    talking to the **one shared terminal** on port 25503.

### 7.3 What a new assistant could do by accident that would damage the frozen record

In rough order of how easy it is to do without noticing:

1. **Edit, check out, reset, stash or pull in the main checkout.** Whatever is there runs at
   the next subprocess and gets committed to whatever branch is checked out (§3.5). Work in a
   worktree; never leave the main checkout on anything but an up-to-date `main`.
2. **Push to `origin/main` while the lab is running** without realising the 16:00 archive will
   merge it into the lab's checkout (§3.5).
3. **Read `positions_open.json`** (either arm) with any tool, script or watcher during market
   hours (§7.2 item 6).
4. **Run `python -m trade_analysis.live_lab.reconstruct_eod --commit`.** It is a one-off whose
   source list still points at the *live* `positions_open.json` files
   (`reconstruct_eod.py:56`, `:63-64`). Today it would read the current book and append
   `eod_reconstructed` rows for it at that session's 15:55 quotes **(inferred)**.
5. **Run `python -m trade_analysis.live_lab.backfill` with a short date range.** It deletes
   `signals.jsonl`, `trades.jsonl`, `events.jsonl`, `outages.jsonl` and `signals_archive/` in
   the output folder before writing (`backfill.py:336-339`), leaving 22 days of committed
   dailies and logs beside one day of trades. And never call `backfill.run(out=…)` with a
   path under `live_lab_data/`: the same deletes would hit the live record.
6. **Run any writer from a worktree and merge the result.** `ledger --reconcile`,
   `gap_recovery --write`, `stranded --write`, `backfill_daily --apply` all default to the
   record copy next to the code (§7.2 item 15). `--reconcile` in particular must only run on
   the lab machine: on a clone without the daily files it confidently writes `MISSED` for
   sessions that ran (`research/live_lab_coverage_audit.md` §4).
7. **Run `autostart` or `restart_terminal --fix` from a worktree.** The lock is per lab
   directory, so it would not see the live supervisor; it would start a second pair of runners
   and its terminal handling (`/v3/terminal/shutdown`, evict, kill) acts on the terminal the
   live lab is using **(inferred from `autostart.py:527-539`, `:1031`)**.
8. **Drop a file into `live_lab_data/`.** It is committed and pushed at the next close unless
   gitignored (§3.5).
9. **`git clean -x` in the main checkout.** It would delete the logs, both `positions_open.json`
   files and the options arm's `recovery_archive/`, none of which exist anywhere else.
10. **"Tidy" an append-only file**: reformat, sort, deduplicate, or delete the retracted
    correction lines. `apply_corrections` depends on file order (`gap_recovery.py:591-599`).
11. **Reword the suspend outage message** or rename the `caught_up` event fields; gap recovery
    parses both (`gap_recovery.py:70`, `:116-123`).
12. **Add `"provenance"` handling or copy rows between `live_lab_backfill/` and
    `live_lab_data/`.** The config hashes match (§3.6).
13. **Change `--symbols` in `run_lab.cmd`**, or let the options arm see another symbol
    (`autostart.py:839-842`).
14. **Run project code at all on the lab machine during a session** against the shared
    terminal: `backfill`, `gap_recovery`, `replay`, `walkforward` all load it
    (`research/runner_capacity_results.md` §4 measured the terminal contending with itself
    past 8 workers).
15. **Quote a total without saying which view** (raw, exact corrections, best available; with
    or without `reconstructed` rows).

---

## 8. Current state

### 8.1 Done and working (seen in the last week's logs)

- Pre-open start at about 09:07 ET with warm-up finished before the open (log 10-12:109-147).
- Freshness re-check at 09:33 (log 10-12:195; log 10-13:175).
- Watchdog restart and catch-up after a short drop (log 10-09:399-405) and after an
  18-minute suspend (log 10-08:601-610).
- Automatic stranded fill the morning after a dead session (log 10-08:10-14: 3 option
  positions with a 57/57 self-check, 25 share positions with 133/133).
- Automatic commit and push to `main` at about 16:00:20 ET (10-08, 10-09, 10-12).
- The gap-recovery rule after three review passes; 287 over-corrections retracted and 2
  reinstated by appended lines (commits `3d24460`, `8e1eda0`, `1768e8f`); 09-11's 32
  after-hours exits written under the new gate (`f7b4429`).

### 8.2 Half-done

- **2026-10-05, 10-06 and 10-07 have no daily summary in either arm** (`live_lab_data/daily/`
  and `shares/daily/` listings). Their trades and reconstructed exits are in `trades.jsonl`.
- **The ledger says `COLLECTED` for 10-05 and 10-06**, days that lost about 2 and 3.5 hours
  to a dead network and had 65 and 69 positions stranded.
- **Shares 2026-09-08 (19 positions) and 2026-09-10 (32)** are reported three times a day as
  "left for a human" and will be forever unless someone decides.
- **Follow-up 1 of the 09-24 incident** (measure exactly which opening entries the rule would
  have taken) is covered in aggregate by the 22-session replay
  (`research/replay_22_sessions.md` §3, "late start" rows), not trade by trade.

### 8.3 Known gaps in recovery, each checked against current code

| gap | still true? | evidence |
|---|---|---|
| Positions open when the network dies before the close get **no row from the runner** | **Yes** for the runner; **since 2026-10-06 the stranded pass fills them automatically**, flagged. | `runner.py:288-297` (frozen branch never reaches the close check); `stranded.py`; log 10-08:10-14 |
| …and no daily summary is written for such a day | **Yes.** | §8.2 |
| …and the ledger does not mark such a day `PARTIAL` | **Yes.** Only host suspends do. | `autostart.py:1194-1206` |
| `find_gaps` ignores network freezes | **Yes.** It reads only `host_suspend` rows, and there is no command-line way to pass a window. The exposure is narrower than when the note was written: catch-up handles a freeze that ends in-session, `managed_through` (10-02) makes a restart resume from the right minute, and the stranded pass handles a freeze that never ends. What is left is a stretch nobody managed that is neither a suspend nor ended by a catch-up **(the narrowing is inferred)**. | `gap_recovery.py:92-109`, `:712-721` |
| Checkpoints from before 2026-10-02 are never resolved | **Yes**, by design. | `stranded.py:33-34` |
| A frozen arm is killed at about 16:14 even if the network returns minutes later | **Yes.** On 10-05 it returned at 16:26 per the comment at `autostart.py:1048-1049`. | `autostart.py:917-939` |
| The watchdog gives up after six restarts and spends them while offline | **Yes.** | §7.2 item 3 |
| The whole process tree sometimes dies with no trace | **Yes, cause not recorded.** 10-07: the log stops after a 15:29 line with no supervisor message; the checkpoint was last saved at 15:32:42 (`shares/recovery_archive/positions_open_2026-10-07_20261008091202.json`); a fresh instance started at 16:00:02. The memory note records the same on 10-02 at about 13:05. | log 10-07:553-555 |
| Recovered exits use the vendor's revised bars and may sit a bar from where live would have fired | **Yes**, stated. | `gap_recovery.py:43-45` |
| The host cannot be kept awake | **Yes.** | §7.2 item 1 |
| The options arm's checkpoint archive is not in git | **Yes.** | `.gitignore:93` |

### 8.4 Open and outside code

- The paid feed is reachable from the network: the active profile is Public with two inbound
  allow rules for the terminal (log 10-12:21-24, a warning every day).
- Time sync (`W32Time`) was recorded as stopped in a memory note; I did not check it.

---

## 9. Contradictions (document or memory versus code)

| says | where | the code or data says |
|---|---|---|
| "the development machine is MST, two hours behind" | `clock.py:4` | `autostart.py:8` says "currently -3h"; preflight prints `local clock offset -3.0h` (log 10-12:17). Both are right for their season; neither states the rule. |
| "the task repeats until 16:30 ET" | `autostart.py:1048` | The task is 05:30 MST plus 8 hours. That is 16:30 ET only in summer time. |
| Watchdog restarts "never when the network is merely down" | `autostart.py:543`; `watchdog_test.py:1-5` | It restarted six times with the network down on 10-05 (§7.2 item 3). |
| Ledger timestamps "carry their offset" | `ledger.py:40`; `README.md:170` | `now_et()` is naive (`clock.py:29`), so `utc_offset_hours` is `null` on every one of the 76 ledger lines (`ledger.py:158-159`). |
| "nothing in the lab may call `datetime.now()` directly" | `clock.py:9-10` | `gap_recovery.py:556`, `:581`, `:689`; `backfill_daily.py:130`; `lock.py:70`; `backfill.py:600` do. |
| A holiday "is detected by the absence of bars shortly after the open" | `autostart.py:11-13` | Not on the first invocation (§7.2 item 5). |
| `validate` is "shared with stranded.py so the two gates cannot drift apart" | `gap_recovery.py:456-457` | They drifted on 2026-10-08 (§7.2 item 9). |
| `runner exited rc=0` / `supervisor_rc: 0` reports the session | log lines; ledger | Structurally 0 for any arm alive at 15:55 (§7.2 item 2). |
| "Run it: `python -m trade_analysis.live_lab.runner`"; "Option Data subscription … cancels 2026-09-05"; file list with no shares arm, ledger, archive or corrections | `trade_analysis/live_lab/README.md:21-50`, `:92-103` | The lab is launched by `autostart` via `run_lab.cmd`; the options arm ran on 10-12 with 360 usable QQQ 0DTE contracts (log 10-12:48). The README predates most of this slice. |
| ThetaData "lapsed to `Options: FREE` on 2026-09-08", nothing new can be measured | `CLAUDE.md` | The options arm has been in the ledger's `arms` list every session since 09-14 and is trading today. Already raised as follow-up 4 of the 09-24 incident; `CLAUDE.md` is unchanged. |
| The lab "sleeps until 09:20"; hash `1f7247d7839d9950`; 13 setups | memory `live-lab-forward-test.md` | 09:05 (`autostart.py:64`); current hashes `ccd00ac71f8243e9` and `7c6414fc549315f3`, 15 setups (log 10-12:112-113). |
| "the scheduled task has no `WakeToRun`" (early in the same note; corrected later in it) | memory `live-lab-forward-test.md` | `WakeToRun True` on 2026-10-13. |
| The main checkout was left on a `main` 76 commits stale | memory `lab-runs-from-main-checkout.md` (2026-09-24) | Resolved: `main` at `6f9c1ab`, equal to `origin/main` by local refs. The mechanism is unchanged. |
| `backfill.py --start X --end X` "WIPES `live_lab_backfill/`" | memory `live-lab-outage-recovery.md` | It deletes five files and `signals_archive/` in the chosen arm's folder; other days' `daily/` and `logs/` stay (`backfill.py:336-339`). The warning stands; the wording is slightly too broad. |
| `catchup.py` once said gap recovery would fill positions with no row | `catchup.py:35-36` | Corrected in place on 2026-10-06, with the reason. Recorded here as an example of the house rule working. |
| "corrections go in `trade_corrections.jsonl`" | this job's brief | True for re-priced exits. Exits for positions with no row are appended to `trades.jsonl`, flagged (§6.2). |

---

## 10. Questions only the owner can answer

1. **Daylight saving ends 2026-11-01.** The task's repetition window will end at 15:30 ET with
   `StopAtDurationEnd = True`. Do you know whether Task Scheduler will stop the running lab at
   that moment? If not, this needs checking before Monday 2026-11-02. Do you also want an
   invocation after the close in winter, so the evening stranded pass still runs?
2. **Should a session that lost hours to a dead network read `PARTIAL`?** 10-05 and 10-06
   read `COLLECTED`.
3. **Do you want daily summaries rebuilt for 10-05, 10-06 and 10-07?** If so, which field for
   the options arm (`atm_net`, as the runner writes, or `net`, as the rebuild tool writes)?
4. **Shares 2026-09-08 (19 positions) and 2026-09-10 (32):** leave permanently unresolved, or
   resolve by hand under a stated rule? Either way, should the three-times-daily log line stop?
5. **What killed the whole process tree on 10-02 (about 13:05 ET) and 10-07 (about 15:33 ET)?**
   The task runs in your interactive session, so it has a console window; were you at the
   machine, and could the window have been closed or the session logged off?
6. **Is the options arm's `recovery_archive/` meant to be gitignored** while the shares arm's
   is tracked?
7. **Which P&L view is the canonical one** to quote: raw `trades.jsonl`, exact corrections
   only, or best available? And are `reconstructed` rows in or out?
8. **Watchdog:** is restarting the terminal while offline, and giving up after six, what you
   intended? On a commute day the budget is gone in under half an hour.
9. **Is the scheduled task saved anywhere outside Windows?** If the laptop is rebuilt, §3.1 is
   the only description.
10. **Thanksgiving week** (closed Thursday 2026-11-26, early close Friday 11-27): are you happy
    for the first invocation on the holiday to start and be stopped by the 09:33 gate? And does
    the options arm need early-close handling like the shares arm has? (The second half belongs
    to the core-trading slice; I only saw that `runner.py` has no early-close logic.)
11. **The feed is exposed on a Public network** with two inbound allow rules. Known and
    accepted, or still to fix?
12. **`CLAUDE.md` says ThetaData options data lapsed.** The options arm is live. What changed,
    and should the file be corrected?
