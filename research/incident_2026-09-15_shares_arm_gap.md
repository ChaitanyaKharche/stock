# Incident — 15-minute unmanaged window in the shares arm, 2026-09-15

**The 2026-09-15 shares session has a hole and it was caused by a monitoring tool I added,
not by the lab.** Recorded here because the forward test's record has to show it: for
roughly fifteen minutes, 58 open share positions had no stop, target or time exit evaluated.

## Timeline (ET)

| time | event |
|---|---|
| 10:45 approx | an external watcher is armed. It reads `positions_open.json` in **both** arm directories every 60s via `json.loads(path.read_text())` |
| **10:51:09** | shares arm dies: `PermissionError: [WinError 5] Access is denied: '...\shares\tmpXXXX.tmp' -> '...\shares\positions_open.json'` at `store.py:179` |
| 10:51:15 | autostart restart 1/20 |
| 10:55:32 | `aborted: warmup_feed_unreachable` — `ABORT: feed unreachable warming up NVDA` |
| 10:55:35 | restart 2/20. autostart evicts unresponsive terminal pid 25460 (up 1.6h), relaunches, answering 10:56:28 |
| 10:58:47 | `aborted: warmup_feed_unreachable` again, on SPY |
| 10:58:49 | restart 3/20. Terminal evicted again (pid 1208), relaunched, answering 11:00:22 |
| 11:00:25 | shares arm starts for the fourth time |
| **11:06:43** | `recovered: n_positions=58, restored_counts=88` — full recovery |
| 11:06 | 1,410 `stale_bar` outages in one minute: 94 per symbol x 15 symbols, the 15-minute bar-cache hole being reported |
| 11:10 onward | entries resume (MOMO_CHASE XLF, XLV, XLE) |

## Root cause

`store.save_open_positions` writes the checkpoint with `os.replace(tmp, path)`
(`store.py:179`). On Windows `os.replace` fails with **WinError 5** when another process
holds the destination file open — Python's `open()` does not pass `FILE_SHARE_DELETE`. The
watcher opened that exact file every 60 seconds, so overlap was a matter of time.

Evidence it was the watcher and not something pre-existing: this error appears **once in
fourteen session logs** spanning 2026-08-28 to 2026-09-15 — the one occurrence being
10:51:09 on 2026-09-15, roughly six minutes after the watcher was armed.

**The rule that follows is absolute: never open a file that is the destination of an
`os.replace`.** In this lab that is `positions_open.json`, in both arm directories, for any
external process. Not "read it carefully", not "catch the exception" — the failure lands in
the *writer*, which is the thing being protected.

## What it actually cost, and what it did not

**The 15 minutes is not the six minutes the crash itself implies.** Had the arm been in its
steady tick loop it would have survived the feed gap entirely: `runner`/`shares_runner`
catch `FeedOutage` per tick, record it, and keep looping. Warmup does not — it **aborts**.
The crash put the arm into warmup at the exact moment the network was unstable, so it then
failed warmup twice more. **The options arm is the control: same feed gap, never crashed,
still alive throughout.** So the crash converted a survivable outage into a dead arm.

Not lost:
- **All 58 positions recovered** at 11:06:43. `load_open_positions` accepted them because the
  checkpoint was stamped with the current session date.
- **Per-day caps restored** (88 signals) from the DECISION record rather than from the
  recovered positions — the 2026-09-02 lesson, working as designed.
- Options arm untouched. Restart budget 3 of 20.

Lost or degraded, and this is what the record needs:
- **10:51:09 -> 11:06:43: 58 share positions with no exit logic evaluated.** Any stop,
  target or time exit that should have fired in that window did not.
- The 15-minute bar gap, visible as the 11:06 `stale_bar` burst.

## A second error in the diagnosis, recorded so the method is not reused

While diagnosing, a probe of `https://http.thetadata.us/...` returned
`The remote name could not be resolved` and this was reported as "DNS is gone, the vendor is
unreachable". **That hostname was a guess.** The lab talks only to `127.0.0.1:25503`; the
terminal's upstream host is internal to the terminal. The same probe still fails while the
feed demonstrably works, which shows the probe was wrong rather than the network. The
genuine evidence of a feed gap was the two `warmup_feed_unreachable` aborts — those are
real; the DNS conclusion was not.

## Follow-ups

- The watcher was rewritten to read **only append-only files** (`events.jsonl`,
  `outages.jsonl`, `trades.jsonl`) and the process list, which can never be replace targets.
- `trade_analysis/live_lab/restart_terminal.py` (commit `2b4e832`) covers the related gap:
  `supervise()` only re-checks the terminal on the arm-restart path, so a terminal that
  wedges while both arms stay healthy is never restarted. That gap did **not** fire here —
  autostart evicted and relaunched the terminal twice, correctly, because arms were dying.
- Field-level traps for anyone analysing these records are in the `live-lab-record-traps`
  memory note.
