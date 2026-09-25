"""The trader's six lines, built LIVE for the Six_Lines setups. Added 2026-09-25.

Both runners were RTH-only by design: `feed.minute_bars` returns 09:30-15:59 and nothing
in the live path ever read a premarket bar. The six lines (six_lines.py, as the trader
specified them on 2026-09-20) need three sources, two of which are premarket:

    yesterday's PREMARKET high/low     -- known at warmup
    yesterday's MARKET-HOURS high/low  -- known at warmup
    today's PREMARKET high/low         -- known only once 04:00-09:29 is over

So the loader works in two steps. At warmup it fetches yesterday's extended session. At
the first tick at or after LOAD_AT it fetches today's premarket and sets
`SessionState.levels`. The earliest a Six_Lines setup can act is the 09:30-09:39 bar's
close at 09:40, so there are nine minutes of slack.

WHY LOAD_AT IS 09:31, NOT 09:30:00. The 09:29 bar closes at 09:30:00 and the vendor
publishes it a moment later. Fetching the premarket at 09:30:01 could miss that bar, and
a premarket high set in its last minute would then be silently wrong. A minute of margin
costs nothing here.

The lines are built by `six_lines.build_six` itself, not a re-implementation, so the live
levels cannot drift from the definition every six-line research number was computed on.
If they cannot be built -- feed down, a holiday-shortened yesterday, no premarket prints --
the setups see `levels is None` and do not trade that symbol today. That is RECORDED as an
outage, never papered over with a partial set: five lines are not the trader's six.
"""
from __future__ import annotations

import datetime as dt

from .feed import FeedOutage
from .six_lines import build_six

LOAD_AT = dt.time(9, 31)
MIN_RTH_BARS = 300          # six_lines.run's "thin session" bar; yesterday must be a real day


def lines_as_dicts(lines) -> tuple:
    return tuple({"name": l.name, "price": float(l.price), "source": l.source,
                  "side": l.side} for l in lines)


class LevelsLoader:
    """One per runner. Holds yesterday's extended bars per symbol until the open."""

    def __init__(self, store, tag: str = "lab"):
        self.store = store
        self.tag = tag
        self.yday: dict[str, list[dict]] = {}
        self.yday_date: dict[str, dt.date] = {}
        self._reported: set[tuple[str, str]] = set()

    def _once(self, sym: str, kind: str, detail: str) -> None:
        if (sym, kind) in self._reported:
            return
        self._reported.add((sym, kind))
        try:
            self.store.outage(kind, detail, symbol=sym)
        except Exception:                                    # noqa: BLE001
            pass
        print(f"[{self.tag}] six-lines {sym}: {detail}", flush=True)

    def at_warmup(self, feed, sym: str, prior_dates) -> bool:
        """Fetch the most recent prior session with a full market-hours day."""
        for d in sorted(prior_dates, reverse=True)[:5]:
            try:
                bars = feed.extended_bars(sym, d, dt.time(4, 0), dt.time(16, 0))
            except FeedOutage as exc:
                self._once(sym, "levels_warmup", f"yesterday's extended bars failed: {exc!r}")
                return False
            rth = [b for b in bars if dt.time(9, 30) <= b["ts"].time() < dt.time(16, 0)]
            if len(rth) >= MIN_RTH_BARS:
                self.yday[sym], self.yday_date[sym] = bars, d
                return True
        self._once(sym, "levels_warmup", "no full prior session in the last 5 weekdays")
        return False

    def at_tick(self, feed, sess, now: dt.datetime) -> None:
        """Set sess.levels once, at the first tick at or after LOAD_AT. Cheap after that."""
        sym = sess.symbol
        if sess.levels is not None or now.time() < LOAD_AT or sym not in self.yday:
            return
        try:
            pre = feed.extended_bars(sym, sess.day, dt.time(4, 0), dt.time(9, 30))
        except FeedOutage as exc:
            self._once(sym, "levels_fetch", f"today's premarket fetch failed, will retry: {exc!r}")
            return
        lines = build_six(self.yday[sym], pre)
        if not lines:
            self._once(sym, "levels_unavailable",
                       f"cannot build the six lines ({len(pre)} premarket bars) -- "
                       f"Six_Lines will not trade {sym} today")
            sess.levels = ()        # built and empty: stop retrying, setups see no lines
            return
        sess.levels = lines_as_dicts(lines)
        try:
            self.store.event("levels", symbol=sym, yday=self.yday_date[sym].isoformat(),
                             n_premarket_bars=len(pre), lines=list(sess.levels))
        except Exception:                                    # noqa: BLE001
            pass
        print(f"[{self.tag}] six-lines {sym}: " + "  ".join(
            f"{l['name']}={l['price']:.2f}" for l in sess.levels), flush=True)
