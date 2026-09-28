"""FREEZE AND CATCH UP -- what a runner does after it has been away.

The lab machine travels. On a commute it goes home WiFi -> phone hotspot -> work WiFi, and
may sleep with the lid shut in between. Each of those is a window in which the runner
cannot see the market. What it does in the first tick AFTER that window is what decides
whether the record stays honest, and before this module it did the wrong thing:

  * The feed returned every missed bar in ONE batch. Entries on those bars were already
    refused (stale-bar guard). Exits were not. The shares arm evaluated each old bar's
    stop and target and filled any that fired at the CURRENT quote -- an exit detected at
    10:21 and priced at 10:47.
  * The options arm checked exits against the LAST bar only, so a stop touched at 10:25
    and recovered by 10:47 was never seen at all.
  * A runner restarted after a crash did both of these, from the checkpoint onward.

WHAT HAPPENS NOW
----------------
  1. FREEZE. Once a runner has gone more than CATCHUP_AFTER_SEC without a good tick --
     network drop, hotspot switch, host suspend, crash-and-restart -- it stops ticking
     live. No entry, no exit, nothing priced off a stale or post-reconnect quote. Open
     positions stay on disk exactly as they were (positions_open.json is still written).
  2. CATCH UP. As soon as the vendor's history is reachable again, the missed minutes are
     replayed ONE AT A TIME through the runner's own `_tick`, against a feed that serves
     history as of each minute (backfill.HistoryFeed, never disk-cached for the live day).
     Every stop, target, time exit, trail and 15:55 flatten fires at the minute and price
     it would have had live. Those rows carry "exit_mode": "caught_up_from_history".
  3. ENTRIES STAY FROZEN THROUGH THE CATCH-UP. A signal that would have fired while the
     runner was away was never decided before its price existed, so it is not the
     forward test's to take. It is counted in the "caught_up" event, not traded.
  4. RESUME. The live feed is restored and the next tick is an ordinary one.

If history is still unreachable the runner stays frozen and retries every idle poll. If
the network never returns before the close, the post-session gap_recovery.py pass
resolves the exits instead -- the two are the same method at two different times.

Bars that were admitted BEFORE the gap are never re-managed: the catch-up pre-admits them
silently and manages only minutes the runner did not see, for the same reason
gap_recovery manages open positions only from the gap's start.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .feed import FeedOutage

CATCHUP_AFTER_SEC = 90.0          # longer than any normal poll jitter (5-15s) plus a
                                  # slow tick; shorter than any gap worth replaying
STEP = dt.timedelta(minutes=1)
OFFSET = dt.timedelta(seconds=2)  # just after bar T+60s+1.5s settle, as live
RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)


def checkpoint_time(lab_dir, day: dt.date) -> dt.datetime | None:
    """When the previous process last saved its book, if it was today. A restarted runner
    catches up from here, so the minutes the crash cost are replayed, not batch-admitted."""
    try:
        d = json.loads((Path(lab_dir) / "positions_open.json").read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None
    if d.get("session_date") != day.isoformat() or not d.get("saved_at"):
        return None
    try:
        t = dt.datetime.fromisoformat(d["saved_at"])
    except ValueError:
        return None
    return t if t.time() >= RTH_OPEN else None


class CatchUp:
    """Owned by one runner. The runner reports good ticks; this decides when it is away."""

    def __init__(self, runner, tag: str):
        self.r, self.tag = runner, tag
        self.last_good: dt.datetime | None = None
        self.frozen = False
        self.suppressed = 0          # entry CHECKS skipped while frozen (bars x timeframes, not signals)

    # ------------------------------------------------------------------ state
    def note_good(self, now: dt.datetime) -> None:
        self.last_good = now

    def away(self, now: dt.datetime) -> bool:
        return (self.last_good is not None
                and (now - self.last_good).total_seconds() > CATCHUP_AFTER_SEC)

    # ------------------------------------------------------------------ the replay
    def run(self, now: dt.datetime, day: dt.date) -> bool:
        """Replay (last_good, now] minute by minute. True once caught up; False if the
        vendor's history is still unreachable, in which case the runner stays frozen."""
        from .backfill import HistoryFeed, SimClock
        r, live, start = self.r, self.r.feed, self.last_good
        close = dt.datetime.combine(day, RTH_CLOSE)
        end = min(now, close)
        clock = SimClock()
        hist = HistoryFeed(live, clock, day, disk_cache=False)
        orig_write = r.store.write_trade
        exits = []

        def tagged(trade, _w=orig_write):
            trade = {**trade, "exit_mode": "caught_up_from_history",
                     "caught_up_window": [start.isoformat(), end.isoformat()]}
            exits.append(trade)
            return _w(trade)

        self.frozen, self.suppressed = True, 0
        r.feed, r.store.write_trade = hist, tagged
        steps = 0
        try:
            # Pre-admit everything the runner had already seen, WITHOUT managing it.
            clock.now = start
            for sym, sess in r.sessions.items():
                sess.accept_bars(hist.minute_bars(sym, day), start)
            t = start.replace(second=0, microsecond=0) + STEP + OFFSET
            if t <= start:
                t += STEP
            while t <= end:
                clock.now = t
                if hasattr(r, "_chain_cache"):
                    r._chain_cache.clear()
                r._tick(t, day)
                self.last_good = t
                steps += 1
                t += STEP
        except FeedOutage as exc:
            r.store.outage("catch_up_blocked", f"history unreachable, staying frozen: {exc!r}")
            return False
        finally:
            r.feed, r.store.write_trade = live, orig_write
        self.frozen = False
        gap_min = (end - start).total_seconds() / 60.0
        r.store.event("caught_up", since=start.isoformat(), until=end.isoformat(),
                      minutes=round(gap_min, 1), steps=steps, exits=len(exits),
                      entry_checks_skipped=self.suppressed)
        print(f"[{self.tag}] caught up {gap_min:.0f} missed minute(s) from history "
              f"({start:%H:%M} -> {end:%H:%M}): {len(exits)} exit(s) at their real minute, "
              f"entries frozen ({self.suppressed} entry checks skipped)",
              flush=True)
        self.last_good = now
        return True
