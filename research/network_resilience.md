# Network resilience — surviving a commute mid-session (2026-09-28)

**The machine will now move during sessions:** home WiFi → phone hotspot → work WiFi,
possibly asleep in between. This note records what can go wrong, what the lab now does
about each case, and what it cannot fix.

## What was measured before changing anything

- **2026-09-25 was a power-button sleep, not a network switch.** The Windows event log
  shows Modern Standby entered at 15:39:30 ET, WiFi dropped at 15:43, and the lid opened
  at 16:19:10. The 22-minute "outage" was the host asleep.
- **This laptop has only Modern Standby, no S3.** Desktop processes do not run during it,
  and on battery the network is cut
  ([MS](https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/prepare-software-for-modern-standby)).
  No code can trade through it. What code can do is **resume honestly**.
- **Power settings:** lid close = Do nothing, power button = Sleep, sleep-after = Never.
  `SetThreadExecutionState` cannot stop a power-button or lid sleep.
- **Short WiFi drops (8–27 s) were survived** that day. The terminal recovered on its own
  within 40–60 s.
- **Theta Terminal v3** exposes `/v3/terminal/mdds/status` (CONNECTED / UNVERIFIED /
  DISCONNECTED / ERROR) and `/v3/terminal/shutdown`
  ([docs](https://thetadata.net/docs/System/System.html)). No reconnect behaviour for its
  history link is documented. A terminal that "keeps the port but stops answering" is
  consistent with a TCP connection that is dead but not yet failed: Windows retransmits
  for up to ~240 s, and keepalive probes an idle socket only after 2 h.

## What was broken

| on reconnect or wake | before | now |
|---|---|---|
| missed bars | admitted in one batch | replayed one minute at a time from history |
| shares exits on those bars | fired on old bars, **filled at the reconnect price** | fire at their own minute, at that minute's historical NBBO |
| options exits on those bars | only the **newest** bar was checked, so a stop hit mid-gap was missed | every missed minute is checked |
| entries during the gap | refused (stale-bar guard) | refused (frozen) and counted |
| restart after a crash | the same batch problem, from the checkpoint | catches up from the checkpoint |
| a wedged terminal | never restarted while both arms were alive | watchdog restarts it, gracefully first |
| the same signal decided twice | possible for setups with a cap above one | impossible (`store.decision_keys`) |

## The mechanisms (all tested; each test verified to fail on the old code)

1. **Freeze and catch up** (`catchup.py`).
   - Freeze: after 90 s without a good tick, a runner stops ticking live. Nothing enters,
     nothing exits, and the book stays on disk.
   - Catch up: when history is reachable, the missed minutes replay one at a time through
     the runner's own `_tick`, entries frozen. The history feed is never disk-cached for
     the live day.
   - Every catch-up exit carries `"exit_mode": "caught_up_from_history"`.
   - **End-to-end check on 2026-09-24:** a 27-minute outage (12:25 → 12:52) against an
     uninterrupted run of the same day.
     - All 12 exits that fell inside the outage matched: same minute, rule and bid.
     - All 63 pre-outage positions matched.
     - The 3 entries the uninterrupted run took during the outage were not taken.
2. **Decided once** (`store.decision_keys`). A (setup, symbol, bar) pair that already has
   a DECISION on disk is never decided again: not by a restart, a catch-up, or a bar the
   feed re-publishes. This is the idempotency a broker gives with a unique client order
   id, or FIX with PossDupFlag.
3. **Terminal watchdog** (`autostart.TerminalWatchdog`). Every minute of RTH:
   - **Not answering for ~2 minutes:** restart.
   - **Answering, internet reachable, history not served for ~3 minutes:** restart. This
     is the "wedged after a network change" state.
   - **Internet unreachable:** never restart. A restart cannot help in a car.
   - Restarts try `/v3/terminal/shutdown` first (tested live: port freed in 0.7 s), then
     kill. Capped at 6 per session.
4. **Post-session gap recovery** (`gap_recovery.py`, 2026-09-25). If the network never
   returns before the close, the same replay runs after the session and appends
   corrections.

## What this cannot do

- **Trade while the laptop sleeps.** Entries during a sleep or a dead link are missed by
  design: they were never decided live.
- **Guarantee the catch-up equals what live would have done.** History bars are the
  vendor's revised bars, and they differ from first-print bars by about one bar on ~40% of
  exits (`incident_2026-09-25_host_suspend_at_close.md`). The 2026-09-24 check above, run
  on revised bars both times, isolates the mechanism from that effect.

## Operational, for a commute day

- **Sleeping is fine; being hot is not.** Lid close is set to *Do nothing*, so a closed lid
  in a bag keeps the machine running at full power, which is an overheating risk. Either
  carry it open and ventilated, or put it to sleep with the power button. The lab
  freezes, and catches up on waking.
- **Plug in when you arrive.** On battery, Windows limits wake locks and hibernates once
  standby drains its budget.
- **The first minutes after each network change are frozen, not lost.** Exits are replayed
  and entries are skipped.
