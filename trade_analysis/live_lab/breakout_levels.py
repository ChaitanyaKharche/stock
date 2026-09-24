"""The trader's 8-line breakout: prior 3 sessions' extremes plus today's premarket.

Frozen by `research/breakout_options_preregistration.md`. Read that first, especially §1
-- the 1DTE arm of the strategy this serves is permanently unmeasurable, and this module
covers the part that can be measured.

    from .breakout_levels import build_levels, find_breakouts
    levels = build_levels(prior_sessions, premarket_today)
    sigs   = find_breakouts(rth_1m, levels)

WHAT IT IS
----------
Eight horizontal lines:
  * each of the previous 3 COMPLETED sessions contributes a high and a low, measured over
    the full extended session 04:00-20:00 ET  -> 6
  * today's premarket, 04:00-09:29 ET, contributes a high and a low                -> 2

A signal is a 10-minute bar CLOSING beyond the nearest un-breached level, plus a buffer.

WHY CLOSE-BASED, AGAIN
----------------------
Same reason as `orb_veto.py`, and deliberately the same mechanics: two hypotheses read off
the same chart by the same person should not disagree about what a break is. On
2026-09-17 QQQ printed 718 on the session's largest volume bar and closed back inside a
715.0-716.7 band -- a touch rule calls that a breakout, a close rule calls it the failed
breakout it was.

THE LOOKAHEAD SURFACE, WHICH IS LARGER HERE THAN IT LOOKS
---------------------------------------------------------
Three separate ways this could read the future, all closed here and all pinned by tests:

  1. a "prior session" that is today. `build_levels` takes sessions the caller has
     already ended; it never reads the current day's RTH.
  2. today's premarket used before 09:30. It is complete at the open and the signal
     window starts at 09:40, so it cannot be.
  3. a level re-triggering after the move that broke it. Each level fires at most once
     per day, or a trending day manufactures a signal every bar and the "win rate"
     becomes a function of trend length.

DEPENDENCY-FREE, like `indicators.py` and `orb_veto.py`: bar dicts in, signals out. No
pandas, no feed, no network, so the same code runs in the sweep, in a test, and -- if any
of this survives -- in the live runner.
"""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
from typing import Sequence

from .orb_veto import BUFFER, INTERVAL_MIN, resample_10m

RTH_OPEN = dt.time(9, 30)
SIGNAL_FROM = dt.time(9, 40)      # after the opening range, matching the veto's OR_END
SIGNAL_TO = dt.time(15, 30)       # before the 15:55 flatten, so an entry has room to work
PREMARKET_FROM = dt.time(4, 0)
EXTENDED_TO = dt.time(20, 0)
LOOKBACK_SESSIONS = 3
# Underlying move an ATM 0DTE must clear just to pay spread and theta, from
# shares_runner.py. Reported per signal so claim B can be killed on the underlying alone.
COST_FLOOR_BP = 5.0


@dc.dataclass(frozen=True)
class Level:
    name: str            # "D-1 high", "premarket low", ...
    price: float
    side: str            # "up" -> broken by closing above; "down" -> below


@dc.dataclass(frozen=True)
class Signal:
    ts: dt.datetime      # the bar that CLOSED beyond the level
    fill_ts: dt.datetime  # the NEXT bar, where the fill happens
    direction: str       # "long" | "short"
    level: Level
    close: float         # the breaking close
    session_pct: float   # how far into 09:40-15:30 this is, 0..1


def _extreme(bars: Sequence[dict], lo_t: dt.time, hi_t: dt.time):
    hi = lo = None
    for b in bars:
        t = b["ts"].time()
        if not (lo_t <= t < hi_t):
            continue
        hi = b["high"] if hi is None else max(hi, b["high"])
        lo = b["low"] if lo is None else min(lo, b["low"])
    return hi, lo


def build_levels(prior_sessions: Sequence[Sequence[dict]],
                 premarket_today: Sequence[dict] = (),
                 include_premarket: bool = True) -> list[Level]:
    """The 8 lines. `prior_sessions` is oldest-first and must NOT contain today.

    Each prior session is its full extended-hours bar list; the high and low are taken
    over 04:00-20:00, because that is what he marks on the chart -- a level that only
    existed in RTH is not a level he would have drawn.

    `include_premarket=False` gives the 6-line variant (secondary S1 in the
    pre-registration), which is what he literally said. See §2 there for why 8 is primary.
    """
    out: list[Level] = []
    recent = list(prior_sessions)[-LOOKBACK_SESSIONS:]
    # Name by recency so a level is identifiable in the output: D-1 is yesterday.
    for i, sess in enumerate(reversed(recent), start=1):
        hi, lo = _extreme(sess, PREMARKET_FROM, EXTENDED_TO)
        if hi is not None:
            out.append(Level(f"D-{i} high", hi, "up"))
        if lo is not None:
            out.append(Level(f"D-{i} low", lo, "down"))
    if include_premarket:
        hi, lo = _extreme(premarket_today, PREMARKET_FROM, RTH_OPEN)
        if hi is not None:
            out.append(Level("premarket high", hi, "up"))
        if lo is not None:
            out.append(Level("premarket low", lo, "down"))
    return out


def find_breakouts(rth_1m: Sequence[dict], levels: Sequence[Level],
                   buffer: float = BUFFER,
                   signal_from: dt.time = SIGNAL_FROM,
                   signal_to: dt.time = SIGNAL_TO) -> list[Signal]:
    """Every 10-minute close beyond a level, each level firing at most once.

    Returns signals in time order. `fill_ts` is the NEXT 10-minute bar, because a close
    is only knowable once the bar ends -- filling on the signal bar is the one-bar
    lookahead that cost this project 88-95.7% of a measured edge.
    """
    ten = [b for b in resample_10m(rth_1m) if b["ts"].time() >= RTH_OPEN]
    if not ten:
        return []
    fired: set[str] = set()
    out: list[Signal] = []
    span = ((dt.datetime.combine(dt.date(2000, 1, 1), signal_to)
             - dt.datetime.combine(dt.date(2000, 1, 1), signal_from)).total_seconds())

    for i, b in enumerate(ten):
        t = b["ts"].time()
        if not (signal_from <= t <= signal_to):
            continue
        if i + 1 >= len(ten):
            continue                      # no next bar to fill on; not a tradeable signal
        c = b["close"]
        for lv in levels:
            if lv.name in fired:
                continue
            up = lv.side == "up" and c > lv.price * (1.0 + buffer)
            dn = lv.side == "down" and c < lv.price * (1.0 - buffer)
            if not (up or dn):
                continue
            fired.add(lv.name)
            elapsed = ((dt.datetime.combine(dt.date(2000, 1, 1), t)
                        - dt.datetime.combine(dt.date(2000, 1, 1), signal_from))
                       .total_seconds())
            out.append(Signal(ts=b["ts"], fill_ts=ten[i + 1]["ts"],
                              direction="long" if up else "short",
                              level=lv, close=c,
                              session_pct=round(elapsed / span, 4) if span else 0.0))
    return out


def nearest_unbroken(levels: Sequence[Level], price: float, direction: str) -> Level | None:
    """The level a move in `direction` would hit next. For reporting, not for the signal.

    The signal loop fires on ANY level a close clears, because a single 10-minute bar can
    clear two at once (a gap through yesterday's high and the premarket high). Reporting
    which was nearest makes the output legible without changing what fired.
    """
    cands = [l for l in levels
             if (direction == "long" and l.side == "up" and l.price >= price)
             or (direction == "short" and l.side == "down" and l.price <= price)]
    if not cands:
        return None
    return min(cands, key=lambda l: abs(l.price - price))


def move_bp(entry: float, exit_px: float, direction: str) -> float:
    """Signed underlying move in basis points, in the direction of the trade.

    The number claim B lives or dies on: compare against COST_FLOOR_BP before believing
    any option result.
    """
    if entry <= 0:
        return 0.0
    raw = (exit_px / entry - 1.0) * 10_000.0
    return raw if direction == "long" else -raw
