# 17 — Live observation of the lab, 2026-10-13 (Tuesday)

**STATUS OF THIS FILE: three parts, three levels of evidence.**

- **Morning (06:34–06:47 MST): watched live** by the observer agent. Sections 1–9.
- **Afternoon (12:30 and 12:44 MST): watched live**, two samples, raw notes at the bottom. The
  observer was cut off by a usage limit before the close and never folded these in.
- **The close (section 10): NOT watched.** Reconstructed by the lead at 18:20 MST from the
  day's log, the ledger, the daily files and `.git\logs\HEAD`, all read-only. It shows what the
  lab logged, not what an observer saw happen.

Sections 7 and 9 promise "details in the final version". That version was never written; the
numbered one-liners are all there is.

Observer rules followed: looked only. No process, task, port, git command or project
Python was touched. Files were opened read-only. This report is the only file written.

Times: the machine is on Arizona time (MST, UTC−7, no daylight saving). Today ET = MST + 3 h.
"ET" below is exchange time. Record files stamp ET with no offset; lock files stamp local MST.

---

## 1. Is it running and healthy

**Morning: yes.** At every morning sample (06:34, 06:36, 06:40, 06:43 MST) the scheduled
task was `Running`, all five lab processes were alive with CPU time rising, the log and
record files were growing, no restart, no outage row, no warning other than the daily
LAN-exposure warning.

## 2. The morning's real timeline

Source: `live_lab_data\logs\2026-10-13.log` (read from the top), the `events.jsonl` /
`signals.jsonl` rows, lock-info files, and process start times.

| MST | ET | what happened | evidence |
|---|---|---|---|
| 05:30:00 | 08:30:00 | Task Scheduler starts `cmd /c run_lab.cmd` (pid 45588) | process start time |
| 05:30:01 | 08:30:01 | `python -m trade_analysis.live_lab.autostart --symbols QQQ SPY --start-terminal` starts (pid 41588) | process start time |
| 05:30:03 | 08:30:03 | log: `autostart invoked \| local 05:30:03 \| symbols ['QQQ', 'SPY']` | log line 2 |
| 05:30:14 | 08:30:14 | supervisor lock taken (`autostart.lock.info`: `pid=41588 since=2026-10-13T05:30:14`); log: `host suspend guard: HELD (system-required + away-mode)` | lock file, log |
| 05:30:15 | 08:30:15 | two `[stranded]` lines: shares 2026-09-08 (19 positions) and 2026-09-10 (32 positions) "have no exit and are left for a human (no_managed_through)" | log |
| 05:50, 05:55, 06:00 | 08:50, 08:55, 09:00 | `waiting 15 / 10 / 5 min until 09:05 ET`. Nothing is logged between 08:30:15 and 08:50:00 | log |
| 06:05:02 | 09:05:02 | `Theta Terminal is not answering on 127.0.0.1:25503` then `starting Theta Terminal ...`; java launcher pid 42540 starts 06:05:02, real terminal java pid 36032 starts 06:05:06 | log, process start times |
| 06:05:11 | 09:05:11 | `terminal is answering` (9 s; yesterday it took 83 s) | log |
| 06:05:12 | 09:05:12 | `historical upstream ready (after 1 probe(s))` | log |
| 06:05:13 | 09:05:13 | `running preflight (structural, pre-open) ...` — covers all 15 symbols | log |
| 06:06:44 | 09:06:44 | preflight output printed (it took 91 s). No `[FAIL]`. `preflight clean; starting runner`, `starting options arm`, `starting shares arm`. Ledger row `OPENED` written (`started_et 09:06:44`, `late_start false`, `arms ["options","shares"]`) | log, `session_ledger.jsonl` |
| 06:06:44 | 09:06:44 | both runner processes start: options pid 37220, shares pid 11380 | process start times |
| 06:06:46 | 09:06:46 | both runner locks taken (`runner.lock.info pid=37220`, `shares\shares_runner.lock.info pid=11380`, same `since=06:06:46.167438`) | lock files |
| 06:06:46–47 | 09:06:46–47 | each runner rolls yesterday's `signals.jsonl` into `signals_archive\2026-10-12.jsonl.gz` (event `signals_rolled`: options 238 rows, shares 14,016 rows, `kept 0`), then writes a `start` event with its config hash | `events.jsonl` both arms, archive file times |
| 06:07:12–06:07:43 | 09:07:12–09:07:43 | options warmup, QQQ then SPY, 25 prior sessions each | options `events.jsonl` |
| 06:07:11–06:12:01 | 09:07:11–09:12:01 | shares warmup, 15 symbols, 25 prior sessions each — about 5 minutes | shares `events.jsonl` |
| 06:12–06:31 | 09:12–09:31 | runners idle; nothing written except the position snapshot | file times |
| 06:30:01 | 09:30:01 | Task Scheduler's 30-minute repeat fires and is refused because the task is already running (`LastTaskResult 0x800710E0`); nothing reaches the log | task info |
| 06:31:02–06:31:04 | 09:31:02–09:31:04 | shares arm: first tick after the open. 15 `levels` events (the six lines per symbol) | shares `events.jsonl` |
| 06:31:15 | 09:31:15 | options arm: `levels` for QQQ and SPY | options `events.jsonl` |
| 06:32:07 | 09:32:07 | **first signals and first entries (shares):** Crabel_Stretch long XLY, XLF, AAPL on the 09:31 bar | shares `signals.jsonl` seq 261262+ |
| 06:33:01 | 09:33:01 | supervisor starts the post-open freshness preflight (child pid 2788, QQQ and SPY only) while both runners keep trading | log, process list |
| 06:34:04 | 09:34:04 | `All checks passed. Safe to run the lab.` / `freshness re-check clean` (63 s) | log |
| 06:36:02 | 09:36:02 | **first exits (shares):** PDH_PDL_Breakout DIA stop, XLI stop, XLP target; TTM_Squeeze XLF stop | log, shares `trades.jsonl` |
| 06:38:09 | 09:38:09 | **first options signal and entry:** Crabel_Stretch QQQ long, three strikes (741/742/743 calls), on the 09:37 bar | options `signals.jsonl` seq 10123–10124 |
| 06:40:02 | 09:40:02 | second options entry: PDH_PDL_Breakout QQQ short (puts 742/743/744) | options `signals.jsonl` seq 10127–10128 |
| 06:40:08 | 09:40:08 | **first options exits:** all three arms of that trade stopped out 5.7 s after entry | options `trades.jsonl` seq 10129–10131 |
| 06:43:06 | 09:43:06 | third options entry: Crabel_Stretch SPY long | options `signals.jsonl` seq 10135–10136 |

There is no "opening range" line in the log. The opening range only appears inside the
`state` of ORB signal rows (`or_high`, `or_low`); the first ORB_5min decision was written at
09:41:03 ET on the 09:40 bar (shares, XLV).

## 3. Process tree (morning)

```
svchost.exe 3856                          Task Scheduler service
└─ cmd.exe 45588          05:30:00   cmd /c "...\trade_analysis\live_lab\run_lab.cmd"
   ├─ conhost.exe 45472   05:30:00   its console window (task is not hidden)
   └─ python.exe 41588    05:30:01   -m trade_analysis.live_lab.autostart --symbols QQQ SPY --start-terminal
      ├─ java.exe 42540   06:05:02   jdk21\...\java.exe -jar ...\thetaterminal\ThetaTerminalv3.jar   (launcher)
      │  └─ java.exe 36032 06:05:06  java ... -jar ...\thetaterminal\lib\202610081.jar              (the terminal; owns the ports)
      ├─ python.exe 37220 06:06:44   -m trade_analysis.live_lab.runner --symbols QQQ SPY --lab-dir ...\live_lab_data --contracts 1.0
      ├─ python.exe 11380 06:06:44   -m trade_analysis.live_lab.shares_runner --symbols SPY QQQ IWM DIA XLK XLY XLI XLV XLP XLE XLF NVDA AAPL GOOGL WMT --lab-dir ...\live_lab_data\shares
      └─ python.exe 2788  06:33:01   -m trade_analysis.live_lab.preflight --ignore-exposure --symbols QQQ SPY   (gone by 06:34:19)
```

- Python: `C:\Users\chaitanyakharche\.pyenv\pyenv-win\versions\3.12.10\python.exe` for
  every Python process. Not the repo venv.
- No command line contained a credential.
- **Every process in the tree runs at `BelowNormal` priority** (task `Priority = 7`, the
  Task Scheduler default).
- Memory (working set / private): supervisor 28–32 MB / ~864 MB; options runner 32 / 44 MB;
  shares runner 32 / 55 MB; terminal 140–180 / 380–400 MB; terminal launcher 32 / 653 MB.
- CPU seconds used, 06:35:48 → 06:43:32 MST: supervisor 7.2 → 10.7, options runner 1.8 → 2.5,
  shares runner 4.3 → 5.3, terminal 21.3 → 24.6. All are mostly asleep.
- Listening ports: **25503 and 25520, both on `[::]` (every interface), owned by terminal
  pid 36032.** Nothing else in the tree listens — there is no dashboard process and no
  dashboard port. Open connections into 25503: options runner 1, shares runner 8.

### How it was launched

`Get-ScheduledTask -TaskName 'LiveLab*'` returned one task:

| field | value |
|---|---|
| name / state | `LiveLab-QQQ-SPY`, `Running` |
| registered | 2026-08-28T01:41:58, author `OMEN\chaitanyakharche` |
| runs as | interactive logon, run level Limited (not elevated) |
| action | `"C:\Users\chaitanyakharche\Documents\stock\trade_analysis\live_lab\run_lab.cmd"`, no arguments, no working directory |
| trigger | weekly, `DaysOfWeek = 62` (Mon–Fri), start `2026-08-31T05:30:00-07:00`, repeat every `PT30M` for `PT8H`, `StopAtDurationEnd = True` |
| settings | `MultipleInstances = IgnoreNew`, `ExecutionTimeLimit = PT12H`, `WakeToRun = True`, `StartWhenAvailable = True`, `DisallowStartIfOnBatteries = False`, `StopIfGoingOnBatteries = False`, `RestartCount = 2` / `RestartInterval = PT5M`, `RunOnlyIfNetworkAvailable = False`, `Priority = 7`, `Hidden = False` |
| info (06:33:58 MST) | `LastRunTime 06:30:01`, `LastTaskResult 0x800710E0`, `NextRunTime 07:00:00`, missed runs 0 |

`run_lab.cmd` (worktree copy, byte-identical to the main checkout's) does `cd /d` to the
main checkout and runs the pyenv Python with `-m trade_analysis.live_lab.autostart
--symbols QQQ SPY --start-terminal`.

The Task Scheduler operational event log returned no events for today, so the meaning of
`0x800710E0` ("the request was refused") as "repeat refused because an instance is already
running" is **(inferred)** from `MultipleInstances = IgnoreNew` plus the timing.

### Which code it is running

- `C:\Users\chaitanyakharche\Documents\stock\.git\HEAD` = `ref: refs/heads/main`.
- `.git\refs\heads\main` = `6f9c1abe70c52885a938b2996d19e0a37ecd802d`, and
  `.git\refs\remotes\origin\main` holds the same hash. Last line of `.git\logs\HEAD`
  (morning): `... 6f9c1ab... live-lab-archive ... commit: Live lab record for 2026-10-12`.
- **Not stale.** 67 files in `trade_analysis\live_lab\` were hashed in both the main
  checkout and this worktree (same commit `6f9c1ab`): 56 identical, 11 differ only by line
  endings (identical after CRLF→LF). `autostart.py`, `runner.py`, `shares_runner.py`,
  `preflight.py`, `store.py`, `ledger.py`, `lock.py`, `feed.py`, `session.py`,
  `run_lab.cmd`, `README.md` are byte-identical. Uncommitted edits elsewhere in the main
  checkout were not checked (that needs `git status`, which was forbidden).

## 4. What gets written, where, how often (morning)

Everything is under `C:\Users\chaitanyakharche\Documents\stock\live_lab_data\` (options arm)
and `...\live_lab_data\shares\` (shares arm). There is **one log for both arms**:
`live_lab_data\logs\2026-10-13.log`. There is no shares log folder.

| file | grows? | measured |
|---|---|---|
| `logs\2026-10-13.log` | yes, a line per entry/exit plus supervisor lines | 17,585 B / 180 lines at 06:34:33 → 21,924 B / 235 lines at 06:43:32 |
| `positions_open.json` (options) | rewritten in place | every ~15 s while flat, every ~5.3 s once a position was open (150 s of mtime polling) |
| `shares\positions_open.json` | rewritten in place | every ~5.0 s, steady |
| `signals.jsonl` (options) | appended; **starts each day empty** | 0 rows until 09:38 ET; 6 rows at 06:40; 11 rows at 06:43 |
| `shares\signals.jsonl` | appended; **starts each day empty** | 36 rows at 06:35; 74 at 06:40; 95 at 06:43 |
| `trades.jsonl` (options) | appended on each exit, never rolled | 2,204,657 B (last write 10-12) → 2,208,449 B at 06:40 (3 rows today) |
| `shares\trades.jsonl` | appended on each exit, never rolled | 2,123,313 B → 2,131,733 B at 06:43 (16 rows today) |
| `events.jsonl` (both arms) | a handful of rows a day, lifecycle only | options 6 rows today (last 06:31:15); shares 32 rows today (last 06:31:04). Did not move after the first tick |
| `session_ledger.jsonl` | one row at start | last write 06:06:44 (`OPENED`) |
| `outages.jsonl` (both arms) | did not move | options last row 2026-10-09; shares last row 2026-10-12 |
| `daily\2026-10-13.json` (both arms) | not present in the morning | — |
| `autostart.lock.info`, `runner.lock.info`, `shares\shares_runner.lock.info` | written once at start | `pid=<n> since=<local MST timestamp>` |

Trap seen directly: at 06:34:19 a directory listing showed `logs\2026-10-13.log` as
**0 bytes, modified 05:30:03**, while the file actually held 180 lines. Windows does not
refresh the directory entry of a file held open for append until something opens it. A
"is the log growing" check that reads the directory size will say the lab is dead.

`events.jsonl` kinds seen all-time: options `start, warmup, levels, signals_rolled,
caught_up, recovered, recovery_rejected, aborted`; shares the same plus `early_close`.
`seq` is one counter per arm shared by every file (options: events 10117–10122, then
signals 10123–10128, then trades 10129–10131).

## 5. Preflight and guard output as actually logged

**Pre-open structural preflight** (header `PREFLIGHT 2026-10-13 09:05:14 (market closed)
local clock offset -3.0h`, printed 09:06:44 ET):

- exposure: `network 'TP-Link_Extender_5GHz' is Public; firewall ON for that profile` then
  `[WARN] EXPOSURE: 2 inbound ALLOW rule(s) match the Theta Terminal binary on the Public
  profile, so 25503 IS reachable from this network.`
- feed: 15 × `[WARN] <SYM> underlying: last quote ... (market closed -- freshness
  UNVERIFIABLE, rerun during RTH)`.
- options: `[OK]` usable 0DTE chains for QQQ (320 contracts), SPY (302), IWM (166), XLF
  (102); `[WARN]` median quote age ~1010 min (market closed) on each; `[WARN] XLF: ATM not
  tradeable right now`; `[WARN] <SYM>: no 0DTE listed for 2026-10-13` for DIA, XLK, XLY,
  XLI, XLV, XLP, XLE, NVDA, AAPL, GOOGL, WMT.
- warmup bars: 15 × `[OK] <SYM>: prior sessions [390, 390, 390] bars`.
- verdict: `No hard failures, but quote freshness is UNVERIFIED while the market is closed.`
  `feed stats: {'calls': 83, 'retries': 0, ... 'outage_episodes': 0, 'upstream_down_now': False}`.
- supervisor: `WARNING: the paid feed is exposed to the LAN. Not blocking the session --
  it is a security issue, not a data-integrity one -- but fix it.` then `preflight clean;
  starting runner`.

**Post-open freshness re-check** (header `PREFLIGHT 2026-10-13 09:33:02 (MARKET OPEN)`,
printed 09:34:04 ET, QQQ and SPY only):

- same exposure warning;
- `[OK] QQQ underlying REAL-TIME (age -0.2s)`, `[OK] SPY underlying REAL-TIME (age -0.2s)`;
- `[OK] QQQ options REAL-TIME (median quote age -54.7s)`, `[OK] SPY options REAL-TIME
  (median quote age -54.5s)`; ATM spreads 0.6% and 1.2% of mid;
- warmup `[390, 390, 390]` for both;
- `All checks passed. Safe to run the lab.` → `freshness re-check clean`.

Other guards: `host suspend guard: HELD (system-required + away-mode)`; the two
`[stranded]` lines; no `[watchdog]`, `[recovery]`, restart, `ABORT` or `HOST SUSPEND` line
in the morning.

## 6. Signals and trades seen so far today (morning counts; no judgement)

At 06:43:32 MST (09:43:32 ET):

- Options arm: 11 signal rows (3 `DECISION`, 3 `FILL status=OPEN`, 5 `SKIP
  skip_reason=max_per_day`), 3 trade rows (one signal × three strikes).
- Shares arm: 95 signal rows; 16 trade rows. At 06:40 the 74 rows were 35 `DECISION`,
  35 `FILL status=FILLED`, 4 `SKIP` (`max_per_direction` 3, `max_per_day` 1).

Example rows (verbatim):

Options `DECISION`, then its `FILL`:
```
{"seq":10123,"signal_id":"aa937cab9dd641848531eb4a4ece76ae","phase":"DECISION","ts":"2026-10-13T09:38:09.649807","bar_ts":"2026-10-13T09:37:00","setup_id":"Crabel_Stretch","config_hash":"ccd00ac71f8243e9","symbol":"QQQ","direction":"long","state":{"buy":740.91909,"sell":738.1369100000001,"stretch":1.3910899999999855,"price_at_bar":741.58,"vwap":739.9516414285249,"bars_1m":8,"bars_5m":1,"dow":"Tue","tf":"1m","slow_setup":false,"dead_setup":false}}
{"seq":10124,"signal_id":"aa937cab9dd641848531eb4a4ece76ae","phase":"FILL","ts":"2026-10-13T09:38:10.008255","status":"OPEN","n_arms":3,"atm_strike":742.0,"strike_distance_pct":0.011456838047490129,"underlying_mid":741.915,"expiration":"2026-10-13","arms":{"ATM":{"strike":742.0,"bid":1.65,"ask":1.66,...},"ATM-1":{"strike":741.0,...},"ATM+1":{"strike":743.0,...}}}
```
Options `SKIP`:
```
{"seq":10125,"signal_id":"2b8085eb6cd94dc9943b521f381195d8","phase":"SKIP","ts":"2026-10-13T09:39:04.223170","bar_ts":"2026-10-13T09:38:00","setup_id":"Crabel_Stretch","config_hash":"ccd00ac71f8243e9","symbol":"QQQ","skip_reason":"max_per_day"}
```
Options trade row (one per strike arm) — fields: `seq, position_id, signal_id, config_hash,
setup_id, setup_version, symbol, arm, direction, right, strike, expiration, contracts,
entry_ts, entry_bar_ts, entry_ask, entry_bid, entry_spread_pct, entry_underlying,
iv_derived, delta_derived, open_interest, stop, target, time_exit_min, bar_exit, trailing,
state, bars_held, mfe_opt, mae_opt, mfe_opt_ts, mae_opt_ts, mfe_und, mae_und, last_bid,
last_underlying, exit_ts, exit_bid, exit_underlying, exit_reason, trigger_level, pnl_gross,
pnl_net, return_pct, hold_minutes, underlying_return, fee_per_contract_side`. Today's first:
`seq 10129, PDH_PDL_Breakout, QQQ, arm ATM, short, put, strike 743.0, entry_ts
09:40:02.417202, exit_ts 09:40:08.157251, exit_reason stop, pnl_net -17.0808`.

Shares `DECISION` + `FILL`:
```
{"seq":261262,"signal_id":"b9221af36ba848db968243bf431c0f40","phase":"DECISION","ts":"2026-10-13T09:32:07.041281","bar_ts":"2026-10-13T09:31:00","setup_id":"Crabel_Stretch","config_hash":"7c6414fc549315f3","symbol":"XLY","direction":"long","state":{"buy":112.5415,"sell":112.0585,"stretch":0.24150000000000063}}
{"seq":261263,"signal_id":"b9221af36ba848db968243bf431c0f40","phase":"FILL","ts":"2026-10-13T09:32:07.053115","status":"FILLED","symbol":"XLY","price":112.7,"shares":88.73114463176574,"spread_bp":1.7746228926349619,"stop":112.0585,"target":null,"time_exit_min":null,"bar_exit":null,"trailing":null,"timeframe":"1m"}
```
Shares trade row:
```
{"seq":261298,"arm":"SHARES","config_hash":"7c6414fc549315f3","setup_id":"PDH_PDL_Breakout","symbol":"DIA","direction":"long","entry_ts":"2026-10-13T09:35:04.835100","exit_ts":"2026-10-13T09:36:02.413744","entry_bar_ts":"2026-10-13T09:34:00","entry_px":521.06,"exit_px":520.97,"shares":19.19164779487967,"notional":10000.0,"exit_reason":"stop","pnl_net":-1.7272,"return_pct":-0.00017272483015375994,"entry_spread_bp":1.343415345640355,"signal_id":"1991be96918745e484d90c84bdb46a1a","hold_minutes":0.96,"decision_lag_min":1.08}
```
`positions_open.json` (both arms): `{"saved_at", "managed_through", "session_date",
"positions":[...]}`; each open position is the trade row with its exit fields null.

## 7. Differences from the README and the autostart docstring (morning list — to be completed)

See the final list below once the close has been observed. Morning findings so far:

1. README never mentions the shares arm; half the process tree and most rows are it.
2. README says a family of 13 setups; both runners log `15 setups`, config `family_size 15`.
3. README says the option subscription "cancels 2026-09-05 ... the options half simply
   stops". Today option chains were served real-time and the options arm filled trades.
4. README "Files written" says `signals.jsonl` holds "every signal". It is emptied at every
   runner start and yesterday's rows go to `signals_archive\<date>.jsonl.gz`.
5. README says the terminal should bind `127.0.0.1`; it listens on `[::]`.
6. README says ledger timestamps "carry their offset"; every row read has
   `"utc_offset_hours": null` and a naive `recorded_at_et`.
7. Docstring says the single-instance lock is what makes the repeating task safe; while the
   first instance lives, Task Scheduler itself refuses the repeat (`IgnoreNew`).
8. README slow-setup list differs from the config on disk.

## 8. Not observed (morning)

The close, EOD flatten, daily summaries, the runners exiting, the ledger's terminal row,
gap recovery, the archive commit and push, the 16:30 ET pass — to be filled in below.

## 9. Anything that looked wrong (morning)

1. Negative quote ages accepted as real-time (`-54.7s`, `-54.5s`, `-0.2s`).
2. PDH_PDL_Breakout rows with `stop == target`, and stops on the wrong side of the fill.
3. `entry_ts` earlier than the `DECISION` row's `ts` in both arms.
4. The two arms stamp the same 5-minute bar differently in `bar_ts`, and stamp log lines
   differently.
5. Whole tree at BelowNormal priority.
6. Stale `shares\runner.lock.info` from 2026-09-11 (pid 16024).
7. Nothing in today's records identifies the code commit the lab ran from.

(Details for each are written up in the final version of this section.)

---

## AFTERNOON RUNNING NOTES (raw, appended sample by sample; folded into the sections above at the end)

The observer was cut off at ~06:50 MST and resumed at 12:30 MST.

### Sample A — 12:30:16 MST / 15:30:16 ET

- Task `Running`; `LastRunTime 12:30:01`, `LastTaskResult 0x800710E0`, next 13:00:00.
- **Same six processes, same pids, same start times as the morning** (cmd 45588, supervisor
  41588, terminal launcher 42540, terminal 36032, options runner 37220, shares runner
  11380). So: no runner restart and no terminal restart between 06:06 and 12:30 MST.
  Still BelowNormal. Ports unchanged (25503, 25520 on `[::]`).
- CPU seconds so far: supervisor 364, shares runner 122, options runner 97, terminal 167.
- Log: 519 lines. Lines 236–519 are **only** `[shares]`/`[lab]` entry and CLOSE lines — not
  one supervisor line, warning, restart or watchdog line after 09:34:04 ET. Last line
  `[shares] 15:12 CLOSE TTM_Squeeze WMT long stop`. Whole-log counts: shares 167 entries /
  120 closes; options 19 entries / 42 closes (each options entry is three strikes).
- Records: options `signals.jsonl` 265 rows (19 DECISION, 19 FILL/OPEN, 227 SKIP:
  max_per_day 153, max_per_direction 66, already_open 8); shares `signals.jsonl` 12,287 rows
  (167 DECISION, 167 FILL/FILLED, SKIP max_per_day 11,720, max_per_direction 233). Trades
  today: options 42 rows, shares 120 rows. Open now: options 15 positions, shares 47.
- `events.jsonl` unchanged since 06:31 MST in both arms. Ledger unchanged (`OPENED`).
- **Things that happened today that are in `outages.jsonl` but NOT in the log:**
  - 11:17:36 ET (08:17:36 MST), options arm: `feed` row `HTTP 503: Unable to resolve host
    mdds-01.thetadata.us` on `/stock/snapshot/quote`, a `tick` row `UpstreamUnreachable(...
    failed after 4 attempts ...)`, then `upstream recovered after 8s and 1 failed call(s)`.
    The shares arm wrote nothing for it.
  - 13:50–13:52 ET (10:50–10:52 MST), both arms: a burst of `stale_quote` rows — options 66
    rows (QQQ/SPY, max age 44.1 s), shares 45 rows (15 a minute, max age 26.6 s). One more
    options row at 13:28 ET.
  - shares: 4 `degraded_bar` rows ("1m gap before HH:MM": XLP 12:49, XLP 12:53, DIA 13:36,
    one at 13:39) and single stale rows at 13:59 and 14:36 ET.
- Git files (before the close): `HEAD = ref: refs/heads/main`; `refs\heads\main =
  6f9c1abe70c52885a938b2996d19e0a37ecd802d` (mtime 10-12 13:00:17); `origin/main` same;
  last line of `.git\logs\HEAD` = `4d91790... 6f9c1ab... live-lab-archive ... commit: Live
  lab record for 2026-10-12`; no `index.lock`.

### Sample B — 12:44:41 MST / 15:44:41 ET

- Task `Running`, same six pids alive. CPU seconds: supervisor 382, shares 128, options 102,
  terminal 171. No extra python/java/git process.
- Log 523 lines (4 new shares CLOSE lines, last `15:40`). Still no supervisor line.
- Options `signals.jsonl` **has not been written since 11:55:07 MST (14:55:07 ET)** — still
  265 rows; options `trades.jsonl` last written 09:55:12 MST. Shares `signals.jsonl` keeps
  growing (12,287 → 12,886 rows in 14 min, about 43 rows and 10 kB a minute — almost all
  `SKIP max_per_day`). Shares trades today 124. Open: options 15, shares 43.
- No daily file yet. Ledger still `OPENED`. `events.jsonl` untouched since 06:31 MST.

---

## 10. The close — reconstructed from the record, not watched

Read at 18:20 MST from `live_lab_data\logs\2026-10-13.log` (597 lines), `session_ledger.jsonl`,
the two `daily\2026-10-13.json` files and the main checkout's `.git\logs\HEAD` (read as a file;
no git command was run there).

| MST | ET | what the record shows | evidence |
|---|---|---|---|
| 12:56 | 15:56 | End-of-day flatten: the last trade lines are all `15:56 CLOSE ... eod` | log, last trade lines |
| 13:00 | 16:00 | Options arm writes its daily summary and stops: 19 signals decided, 227 skipped, 57 trade rows closed, at-the-money net **−$548.54** | log `[lab] daily: {...}`, `[lab] stopped` |
| 13:00 | 16:00 | Shares arm writes its daily summary and stops: 167 trades, net **+$11.50** | log `[shares] daily`, `[shares] stopped` |
| 13:00:16 | 16:00:16 | Supervisor: both arms `finished rc=0 after 0 restart(s)`. Ledger row `COLLECTED`, `supervisor_rc 0` | log, ledger last row |
| 13:00:19 | 16:00:19 | Stranded pass: the same two "left for a human" lines (shares 2026-09-08, 19 positions; 2026-09-10, 32 positions) | log |
| 13:00:21 | 16:00:21 | `[archive] committed 48a76f2  13 files changed` | log; `.git\logs\HEAD` last line: `6f9c1ab → 48a76f2 ... live-lab-archive ... commit: Live lab record for 2026-10-13` |
| 13:00:25 | 16:00:25 | `[archive] pushed to main` | log (the push itself was not verified against the remote) |
| 13:30:04 | 16:30:04 | Last scheduled repeat: `it is already 16:30 ET, past the 15:30 cutoff`, runs the stranded pass, exits | log |

What this confirms from report 02's timeline: flatten at 15:55–15:56, runners exit at 16:00,
ledger `COLLECTED`, stranded pass, archive commit and push within about ten seconds of the
close, one evening pass at 16:30 ET. No restart, no suspend, no watchdog line all day.

What it does not confirm: gap recovery printed nothing today, so whether it ran and found
nothing or did not run is not visible in the log.

Consequences for anyone working in this repository:

- The main checkout's `main` is now at `48a76f2`. The worktree these reports were written in
  is still at `6f9c1ab`, one data commit behind.
- The day had two events the log never mentioned (an 8-second upstream outage at 11:17 ET and
  a burst of stale quotes at 13:50–13:52 ET, both in `outages.jsonl` — see the afternoon
  notes above). The ledger still says `COLLECTED`. A clean log and a `COLLECTED` row do not
  mean an uneventful day.
- The options arm wrote no new signal row after 14:55 ET (afternoon sample B). With 227 of
  246 options signal rows being `SKIP`, mostly `max_per_day`, that is probably the per-day
  caps being reached rather than a stall, but nobody checked.
